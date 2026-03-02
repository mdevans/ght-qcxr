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
OUTPUT_DIR_BASE = Path("segment/cxas_parallax_masks")

DIR_PASS = OUTPUT_DIR_BASE / "pass"
DIR_REJECT = OUTPUT_DIR_BASE / "reject"
DIR_ERROR = OUTPUT_DIR_BASE / "error"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

THORACIC_SPINE_IDX = 2
STERNUM_IDX = 29

# --- HELPER FUNCTIONS ---

def normalize(array: torch.Tensor) -> torch.Tensor:
    array = (array - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / (
        torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    )
    return array

def load_segmentation_model():
    print(f"🚀 Loading CXAS BackboneUNet (Parallax Mode) on {DEVICE}...")
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

def merge_vertebrae(binary_mask):
    """Bridges the gaps between individual vertebrae."""
    mask_uint8 = binary_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
    merged_spine = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    return merged_spine

def fit_vertical_line(mask):
    """Fits a line to a mask and forces it to perfectly vertical if the angle is small."""
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return None
        
    points = np.column_stack((x_coords, y_coords)).astype(np.float32)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    angle_from_vertical = math.degrees(math.atan2(abs(vx), abs(vy)))

    if angle_from_vertical < 5.0:
        vx = 0.0
        vy = 1.0
        M = cv2.moments(mask)
        if M["m00"] != 0:
            x0 = M["m10"] / M["m00"]

    return (vx, vy, x0, y0)

# --- MATH & VISUALIZATION ---

def calculate_parallax_rotation(masks, image_width, filename):
    # 1. Clean the masks
    raw_spine = merge_vertebrae(masks[THORACIC_SPINE_IDX])
    clean_spine = keep_largest_blob_only(raw_spine)
    clean_sternum = keep_largest_blob_only(masks[STERNUM_IDX])

    metrics = {
        "valid": False, "status": "ANATOMY ERROR", "error_msg": "",
        "spine_line": None, "sternum_line": None,
        "eval_y": 0, "spine_x": 0, "sternum_x": 0,
        "pixel_diff": 0, "offset_ratio": 0.0
    }

    # Sanity Check 1: Do both masks exist?
    if np.max(clean_spine) == 0:
        metrics["error_msg"] = "Missing Spine Mask"
        return metrics, clean_spine, clean_sternum
    if np.max(clean_sternum) == 0:
        metrics["error_msg"] = "Missing Sternum Mask"
        return metrics, clean_spine, clean_sternum

    # 2. Fit the lines
    spine_line = fit_vertical_line(clean_spine)
    sternum_line = fit_vertical_line(clean_sternum)

    if not spine_line or not sternum_line:
        metrics["error_msg"] = "Failed to fit structural lines"
        return metrics, clean_spine, clean_sternum

    metrics["spine_line"] = spine_line
    metrics["sternum_line"] = sternum_line

    # 3. Calculate Parallax Distance
    # We evaluate the horizontal distance between the lines at the vertical center of the sternum.
    _, _, _, y0_st = sternum_line
    eval_y = int(y0_st)
    metrics["eval_y"] = eval_y

    vx_sp, vy_sp, x0_sp, y0_sp = spine_line
    vx_st, vy_st, x0_st, y0_st = sternum_line

    # Helper function to find X at a given Y on a parametric line
    def get_x_at_y(target_y, vx, vy, x0, y0):
        if abs(vy) < 1e-6: return x0 # Prevent division by zero if perfectly horizontal (unlikely)
        t = (target_y - y0) / vy
        return x0 + t * vx

    spine_x = get_x_at_y(eval_y, vx_sp, vy_sp, x0_sp, y0_sp)
    sternum_x = get_x_at_y(eval_y, vx_st, vy_st, x0_st, y0_st)

    metrics["spine_x"] = spine_x
    metrics["sternum_x"] = sternum_x

    metrics["pixel_diff"] = abs(spine_x - sternum_x)
    metrics["offset_ratio"] = metrics["pixel_diff"] / image_width
    
    metrics["valid"] = True
    # The acceptable threshold for parallax offset will likely need empirical tuning!
    # Starting with 3% as a strict baseline.
    metrics["status"] = "PASS" if metrics["offset_ratio"] <= 0.03 else "REJECT"

    return metrics, clean_spine, clean_sternum

def visualize_parallax(img, masks_tuple, metrics, save_path):
    spine_mask, sternum_mask = masks_tuple
    
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_layer = overlay.copy()
    
    mask_layer[spine_mask > 0] = [255, 255, 0]   # Cyan for Spine
    mask_layer[sternum_mask > 0] = [0, 165, 255] # Orange for Sternum

    alpha = 0.3
    cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0, overlay)

    h, w = img.shape[:2]

    if metrics["valid"]:
        vx_sp, vy_sp, x0_sp, y0_sp = metrics["spine_line"]
        vx_st, vy_st, x0_st, y0_st = metrics["sternum_line"]
        eval_y = metrics["eval_y"]
        sp_x = int(metrics["spine_x"])
        st_x = int(metrics["sternum_x"])

        # Draw Spine Line (Red)
        if abs(vy_sp) > 1e-6: 
            top_x = int(x0_sp - y0_sp * (vx_sp / vy_sp))
            bottom_x = int(x0_sp + (h - y0_sp) * (vx_sp / vy_sp))
            cv2.line(overlay, (top_x, 0), (bottom_x, h), (0, 0, 255), 2, cv2.LINE_AA)

        # Draw Sternum Line (Green)
        if abs(vy_st) > 1e-6: 
            top_x = int(x0_st - y0_st * (vx_st / vy_st))
            bottom_x = int(x0_st + (h - y0_st) * (vx_st / vy_st))
            cv2.line(overlay, (top_x, 0), (bottom_x, h), (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the measurement bridge between the two lines
        cv2.line(overlay, (sp_x, eval_y), (st_x, eval_y), (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw evaluation points
        cv2.circle(overlay, (sp_x, eval_y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (st_x, eval_y), 4, (0, 255, 0), -1, cv2.LINE_AA)
        
        text = f"Parallax Offset: {metrics['offset_ratio']:.1%} [{metrics['status']}]"
        color = (0, 255, 0) if metrics["status"] == "PASS" else (0, 0, 255)
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    
    else:
        error_text = f"ERROR: {metrics.get('error_msg', 'Unknown Error')}"
        text_size = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 30
        cv2.putText(overlay, error_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

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
    
    print(f"\n🧪 Testing Parallax (Spine vs. Sternum) on {len(test_files)} Images...\n")
    print(f"{'FILENAME':<30} | {'OFFSET %':<10} | {'RESULT'}")
    print("-" * 60)

    counts = {"pass": 0, "reject": 0, "error": 0}

    for image_path in test_files:
        filename = image_path.name
        img_str = str(image_path)
        
        try:
            pil_img = Image.open(image_path).convert(mode="RGB")
            orig_file_size = (pil_img.size[1], pil_img.size[0]) 
            orig_width = pil_img.size[0]
        except Exception as e:
            continue

        cv_img_raw = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
        if cv_img_raw is None: continue
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

        metrics, sp_mask, st_mask = calculate_parallax_rotation(masks, orig_width, filename)

        if not metrics["valid"]:
            status_str = "ANATOMY ERROR"
            out_dir = DIR_ERROR
            counts["error"] += 1
            print(f"{filename:<30} | {'---':<10} | 🚨 {status_str} ({metrics.get('error_msg')})")
        else:
            if metrics["status"] == "PASS":
                out_dir = DIR_PASS
                counts["pass"] += 1
                icon = "✅"
            else:
                out_dir = DIR_REJECT
                counts["reject"] += 1
                icon = "⚠️"
            
            print(f"{filename:<30} | {metrics['offset_ratio']:<10.1%} | {icon} {metrics['status']}")

        save_path = out_dir / f"qc_{filename}"
        visualize_parallax(cv_img_enhanced, (sp_mask, st_mask), metrics, str(save_path))

    print("\n" + "="*40)
    print("🎯 BATCH PROCESSING COMPLETE")
    print(f"PASS: {counts['pass']} | REJECT: {counts['reject']} | ERROR: {counts['error']}")
    print("="*40)

if __name__ == "__main__":
    process_batch()