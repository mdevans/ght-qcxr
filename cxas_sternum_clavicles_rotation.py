import torch
import numpy as np
import cv2
import math
import re
from pathlib import Path
import torch.nn.functional as F
from PIL import Image

from cxas.models.UNet.backbone_unet import BackboneUNet

# --- CONFIGURATION ---
WEIGHTS_PATH = Path.home() / ".cxas" / "weights" / "UNet_ResNet50_default.pth"
IMAGE_DIR = Path("data/TB_Chest_Radiography_Database/Normal") 
OUTPUT_DIR_BASE = Path("segment/cxas_sternum_clavicle_masks")

DIR_PASS = OUTPUT_DIR_BASE / "pass"
DIR_REJECT = OUTPUT_DIR_BASE / "reject"
DIR_ERROR = OUTPUT_DIR_BASE / "error"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLAVICLE_LEFT_IDX = 31
CLAVICLE_RIGHT_IDX = 32
STERNUM_IDX = 29

# --- HELPER FUNCTIONS ---

def normalize(array: torch.Tensor) -> torch.Tensor:
    array = (array - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / (
        torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    )
    return array

def load_segmentation_model():
    print(f"🚀 Loading CXAS BackboneUNet (Sternum Mode) on {DEVICE}...")
    model = BackboneUNet(model_name="UNet_ResNet50_default", classes=159)
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def keep_largest_blob_only(binary_mask):
    mask_uint8 = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_uint8
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(clean_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    return clean_mask

def keep_robust_clavicle_blobs(binary_mask, image_width, image_height):
    """
    A strict filter for clavicles that eliminates 'splinters' and tiny noise fragments.
    It enforces both a relative size threshold and an absolute minimum area threshold.
    """
    mask_uint8 = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask_uint8
        
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    
    if max_area == 0: 
        return mask_uint8

    # 1. Dynamic Minimum Area Threshold
    # A true clavicle should take up at least a certain percentage of the image area.
    # 0.05% of the total image area is a very conservative lower bound for a clavicle fragment.
    total_image_area = image_width * image_height
    min_pixel_area = total_image_area * 0.0005 

    clean_mask = np.zeros_like(mask_uint8)
    valid_contours = []
    
    for i, contour in enumerate(contours):
        # 2. The Strict Tests
        # Pass Condition 1: Must be at least 25% the size of the largest clavicle piece
        # Pass Condition 2: Must be larger than the absolute minimum pixel area
        if areas[i] >= (max_area * 0.25) and areas[i] >= min_pixel_area:
            valid_contours.append(contour)
            
    cv2.drawContours(clean_mask, valid_contours, -1, 1, thickness=cv2.FILLED)
    return clean_mask

def get_robust_medial_point(mask, side):
    kernel = np.ones((5,5), np.uint8) 
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    y_coords, x_coords = np.where(eroded_mask > 0)
    
    if len(x_coords) == 0:
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            return (0, 0)

    if side == 'right':
        idx = np.argmax(x_coords)
    else: 
        idx = np.argmin(x_coords)
        
    return (int(x_coords[idx]), int(y_coords[idx]))

# --- MATH & VISUALIZATION ---

def calculate_sternum_rotation(masks, image_width, image_height):
    # Use the new, stricter robust blob filter for the clavicles
    left_clav_mask = keep_robust_clavicle_blobs(masks[CLAVICLE_LEFT_IDX], image_width, image_height)
    right_clav_mask = keep_robust_clavicle_blobs(masks[CLAVICLE_RIGHT_IDX], image_width, image_height)
    
    # Isolate the clean Sternum mask
    sternum_mask = keep_largest_blob_only(masks[STERNUM_IDX])

    metrics = {
        "valid": False, "status": "ANATOMY ERROR",
        "right_medial_pt": (0, 0), "left_medial_pt": (0, 0),
        "spine_line": (0.0, 1.0, 256.0, 256.0), 
        "dist_right": 0, "dist_left": 0, "pixel_diff": 0, "offset_ratio": 0.0
    }

    # 1. Get Medial Points
    rx, ry = get_robust_medial_point(right_clav_mask, 'right')
    lx, ly = get_robust_medial_point(left_clav_mask, 'left')
    metrics["right_medial_pt"] = (rx, ry)
    metrics["left_medial_pt"] = (lx, ly)
    
    if rx == 0 or lx == 0:
        return metrics, left_clav_mask, right_clav_mask, sternum_mask

    # 🚨 ANATOMICAL SANITY CHECK 1: Crossover
    # Patient Right clavicle must be physically to the left of Patient Left clavicle
    if rx >= lx:
        return metrics, left_clav_mask, right_clav_mask, sternum_mask

    # 🚨 ANATOMICAL SANITY CHECK 2: Vertical Disparity
    # Medial tips should not have a massive vertical height difference
    y_diff_ratio = abs(ry - ly) / image_height
    if y_diff_ratio > 0.15:  # 15% of image height is an absurdly un-anatomical gap
        return metrics, left_clav_mask, right_clav_mask, sternum_mask

    # 2. Fit Line to the Sternum
    sternum_y, sternum_x = np.where(sternum_mask > 0)
    if len(sternum_x) == 0:
        return metrics, left_clav_mask, right_clav_mask, sternum_mask
        
    points = np.column_stack((sternum_x, sternum_y)).astype(np.float32)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    angle_from_vertical = math.degrees(math.atan2(abs(vx), abs(vy)))

    if angle_from_vertical < 5.0:
        vx = 0.0
        vy = 1.0
        M_sternum = cv2.moments(sternum_mask)
        if M_sternum["m00"] != 0:
            x0 = M_sternum["m10"] / M_sternum["m00"]

    metrics["spine_line"] = (vx, vy, x0, y0)

    # 🚨 ANATOMICAL SANITY CHECK 3: Midline Proximity
    # If the segmented clavicle tip is extremely far from the sternum line, 
    # the AI probably segmented the shoulder joint instead of the medial tip.
    dist_rx_to_line = abs((rx - x0) * vy - (ry - y0) * vx)
    dist_lx_to_line = abs((lx - x0) * vy - (ly - y0) * vx)
    
    if (dist_rx_to_line / image_width) > 0.30 or (dist_lx_to_line / image_width) > 0.30:
        return metrics, left_clav_mask, right_clav_mask, sternum_mask

    # 3. Final Math
    metrics["dist_right"] = dist_rx_to_line
    metrics["dist_left"] = dist_lx_to_line
    metrics["pixel_diff"] = abs(metrics["dist_left"] - metrics["dist_right"])
    metrics["offset_ratio"] = metrics["pixel_diff"] / image_width
    
    metrics["valid"] = True
    metrics["status"] = "PASS" if metrics["offset_ratio"] <= 0.05 else "REJECT"

    return metrics, left_clav_mask, right_clav_mask, sternum_mask

def visualize_results(img, masks_tuple, metrics, save_path):
    left_clav, right_clav, spine = masks_tuple
    
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_layer = overlay.copy()
    
    mask_layer[spine > 0] = [0, 165, 255] # Orange for Sternum
    mask_layer[left_clav > 0] = [255, 0, 255]
    mask_layer[right_clav > 0] = [0, 255, 255]

    alpha = 0.3
    cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0, overlay)

    if metrics["valid"]:
        rx, ry = metrics["right_medial_pt"]
        lx, ly = metrics["left_medial_pt"]
        vx, vy, x0, y0 = metrics["spine_line"]
        
        h = img.shape[0]
        if abs(vy) > 1e-6: 
            top_x = int(x0 - y0 * (vx / vy))
            bottom_x = int(x0 + (h - y0) * (vx / vy))
            cv2.line(overlay, (top_x, 0), (bottom_x, h), (0, 0, 255), 1, cv2.LINE_AA)

        def get_perpendicular_pt(px, py, vx, vy, x0, y0):
            t = (px - x0) * vx + (py - y0) * vy
            return int(x0 + t * vx), int(y0 + t * vy)

        r_intersect = get_perpendicular_pt(rx, ry, vx, vy, x0, y0)
        l_intersect = get_perpendicular_pt(lx, ly, vx, vy, x0, y0)

        cv2.line(overlay, (rx, ry), r_intersect, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (lx, ly), l_intersect, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.circle(overlay, (rx, ry), 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, (lx, ly), 3, (0, 255, 0), -1, cv2.LINE_AA)
        
        text = f"Offset: {metrics['offset_ratio']:.1%} [{metrics['status']}]"
        color = (0, 255, 0) if metrics["status"] == "PASS" else (0, 0, 255)
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, overlay)

# --- BATCH PROCESSOR ---

def process_batch():
    DIR_PASS.mkdir(parents=True, exist_ok=True)
    DIR_REJECT.mkdir(parents=True, exist_ok=True)
    DIR_ERROR.mkdir(parents=True, exist_ok=True)
    
    model = load_segmentation_model()
    
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    def natural_sort_key(path_obj):
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', path_obj.name)]

    all_files = sorted([
        p for p in IMAGE_DIR.iterdir() 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ], key=natural_sort_key)
    
    if not all_files:
        print(f"❌ No images found in {IMAGE_DIR}")
        return
    
    test_files = all_files[:1000]
    
    print(f"\n🧪 Testing Sternum-Based Rotation on {len(test_files)} Images...\n")
    print(f"{'FILENAME':<30} | {'LEFT DIST':<10} | {'RIGHT DIST':<10} | {'OFFSET %':<10} | {'RESULT'}")
    print("-" * 80)

    counts = {"pass": 0, "reject": 0, "error": 0}

    for image_path in test_files:
        filename = image_path.name
        img_str = str(image_path)
        
        try:
            pil_img = Image.open(image_path).convert(mode="RGB")
            orig_file_size = (pil_img.size[1], pil_img.size[0]) 
            orig_width = pil_img.size[0]
            orig_height = pil_img.size[1] # 🚀 Needed for sanity checks
        except Exception as e:
            print(f"❌ Skipping {filename} - Error loading: {e}")
            continue

        cv_img_raw = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
        if cv_img_raw is None:
            print(f"❌ Skipping {filename} - OpenCV failed to load")
            continue

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cv_img_enhanced = clahe.apply(cv_img_raw)

        array = np.array(pil_img)
        array = np.transpose(array, [2, 0, 1])
        array = torch.tensor(array).float() / 255.0
        array = normalize(array).to(DEVICE)
        
        img_tensor = F.interpolate(array.unsqueeze(0), size=(512, 512))
        input_dict = {"data": img_tensor}

        with torch.no_grad():
            output_dict = model(input_dict)
            
        segmentation = output_dict["segmentation_preds"][0]
        pred = segmentation.float()
        pred = F.interpolate(pred.unsqueeze(0), size=orig_file_size, mode="nearest")
        masks = pred[0].bool().cpu().numpy()

        # Pass orig_height to evaluate vertical disparity
        metrics, left_m, right_m, sternum_m = calculate_sternum_rotation(masks, orig_width, orig_height)

        if not metrics["valid"]:
            status_str = "ANATOMY ERROR"
            out_dir = DIR_ERROR
            counts["error"] += 1
            print(f"{filename:<30} | {'---':<10} | {'---':<10} | {'---':<10} | 🚨 {status_str}")
        else:
            if metrics["status"] == "PASS":
                out_dir = DIR_PASS
                counts["pass"] += 1
                icon = "✅"
            else:
                out_dir = DIR_REJECT
                counts["reject"] += 1
                icon = "⚠️"
            
            print(f"{filename:<30} | {metrics['dist_left']:<10.1f} | {metrics['dist_right']:<10.1f} | {metrics['offset_ratio']:<10.1%} | {icon} {metrics['status']}")

        save_path = out_dir / f"qc_{filename}"
        visualize_results(cv_img_enhanced, (left_m, right_m, sternum_m), metrics, str(save_path))

    print("\n" + "="*40)
    print("🎯 BATCH PROCESSING COMPLETE")
    print("="*40)

if __name__ == "__main__":
    process_batch()