from pathlib import Path
import cv2
import math
import re
import torch
import numpy as np
import torchxrayvision as xrv

# --- CONFIGURATION ---
IMAGE_DIR = Path("data/TB_Chest_Radiography_Database/Normal") 
OUTPUT_DIR_BASE = Path("segment/xrv_output_masks")

# Pathlib makes joining directories incredibly clean
DIR_PASS = OUTPUT_DIR_BASE / "pass"
DIR_REJECT = OUTPUT_DIR_BASE / "reject"
DIR_ERROR = OUTPUT_DIR_BASE / "error"

IMAGE_WIDTH = 512
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
def load_segmentation_model():
    print(f"🚀 Loading TorchXRayVision PSPNet on {DEVICE}...")
    model = xrv.baseline_models.chestx_det.PSPNet().to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Loads image, applies CLAHE contrast, and prepares it for XRV."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
        
    # Resize to exactly 512x512 if it isn't already (XRV PSPNet requirement)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_WIDTH))
        
    # Apply CLAHE to boost local edge contrast of bones
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
        
    # XRV normalization maps pixel values to [-1024, 1024]
    normalized = xrv.datasets.normalize(enhanced_img, 255)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    return tensor, enhanced_img

def keep_largest_blob(binary_mask):
    """Finds all separate blobs in a mask and deletes everything except the largest one."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_mask # Return as-is if empty
        
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a fresh, pitch-black mask
    clean_mask = np.zeros_like(binary_mask)
    
    # Draw ONLY the largest contour back onto the clean mask in solid white (1)
    cv2.drawContours(clean_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    
    return clean_mask
def keep_significant_blobs(binary_mask, threshold_ratio=0.15):
    """
    Keeps the largest contour, PLUS any other contours that are 
    at least `threshold_ratio` (e.g., 15%) the size of the largest one.
    This stitches together clavicles broken by medical tubing while 
    still deleting tiny noise artifacts.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_mask # Return as-is if empty
        
    # Calculate areas for all contours
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    
    # If the largest area is basically zero, just return
    if max_area == 0: return binary_mask

    # Create a fresh, pitch-black mask
    clean_mask = np.zeros_like(binary_mask)
    
    # Draw any contour that is large enough
    valid_contours = []
    for i, contour in enumerate(contours):
        if areas[i] >= (max_area * threshold_ratio):
            valid_contours.append(contour)
            
    # Draw the valid contours back onto the clean mask in solid white (1)
    cv2.drawContours(clean_mask, valid_contours, -1, 1, thickness=cv2.FILLED)
    
    return clean_mask

def calculate_rotation(masks):
    left_clav_mask = keep_significant_blobs((masks[0] > 0.5).astype(np.uint8))
    right_clav_mask = keep_significant_blobs((masks[1] > 0.5).astype(np.uint8))
    full_spine_mask = keep_significant_blobs((masks[13] > 0.5).astype(np.uint8))

    # Note: "spine_x" is completely removed. We exclusively use "spine_line" now.
    metrics = {
        "valid": False, "status": "ERROR",
        "right_medial_pt": (0, 0), "left_medial_pt": (0, 0),
        "spine_line": (0.0, 1.0, 256.0, 256.0), # (vx, vy, x0, y0)
        "dist_right": 0, "dist_left": 0, "pixel_diff": 0, "offset_ratio": 0.0
    }

    # 1. Medial points
    r_y_coords, r_x_coords = np.where(right_clav_mask > 0)
    if len(r_x_coords) == 0: return metrics, left_clav_mask, right_clav_mask, full_spine_mask
    max_x_idx = np.argmax(r_x_coords)
    metrics["right_medial_pt"] = (int(r_x_coords[max_x_idx]), int(r_y_coords[max_x_idx]))

    l_y_coords, l_x_coords = np.where(left_clav_mask > 0)
    if len(l_x_coords) == 0: return metrics, left_clav_mask, right_clav_mask, full_spine_mask
    min_x_idx = np.argmin(l_x_coords)
    metrics["left_medial_pt"] = (int(l_x_coords[min_x_idx]), int(l_y_coords[min_x_idx]))

    # 2. Isolate upper spine
    y_min_clav = min(metrics["right_medial_pt"][1], metrics["left_medial_pt"][1])
    y_max_clav = max(metrics["right_medial_pt"][1], metrics["left_medial_pt"][1])
    
    top_buffer = 20
    bottom_buffer = 150 
    
    y_start = max(0, y_min_clav - top_buffer)
    y_end = min(full_spine_mask.shape[0], y_max_clav + bottom_buffer)
    
    upper_spine_mask = np.zeros_like(full_spine_mask)
    upper_spine_mask[y_start:y_end, :] = full_spine_mask[y_start:y_end, :]

    # 3. Line Fit (With NumPy Flattening Fix)
    spine_y, spine_x = np.where(upper_spine_mask > 0)
    if len(spine_x) == 0:
        return metrics, left_clav_mask, right_clav_mask, upper_spine_mask
        
    points = np.column_stack((spine_x, spine_y)).astype(np.float32)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    # Hybrid Snap-to-Vertical Catch
    angle_from_vertical = math.degrees(math.atan2(abs(vx), abs(vy)))

    if angle_from_vertical < 5.0:
        vx = 0.0
        vy = 1.0
        M_spine = cv2.moments(upper_spine_mask)
        if M_spine["m00"] != 0:
            x0 = M_spine["m10"] / M_spine["m00"]

    metrics["spine_line"] = (vx, vy, x0, y0)

    # 4. Math: Perpendicular Distance
    rx, ry = metrics["right_medial_pt"]
    lx, ly = metrics["left_medial_pt"]
    
    metrics["dist_right"] = abs((rx - x0) * vy - (ry - y0) * vx)
    metrics["dist_left"] = abs((lx - x0) * vy - (ly - y0) * vx)
    
    metrics["pixel_diff"] = abs(metrics["dist_left"] - metrics["dist_right"])
    metrics["offset_ratio"] = metrics["pixel_diff"] / IMAGE_WIDTH
    
    metrics["valid"] = True
    metrics["status"] = "PASS" if metrics["offset_ratio"] <= 0.05 else "REJECT"

    return metrics, left_clav_mask, right_clav_mask, upper_spine_mask

def visualize_results(img, masks_tuple, metrics, save_path):
    left_clav, right_clav, spine = masks_tuple
    
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_layer = overlay.copy()
    
    mask_layer[spine > 0] = [255, 255, 0]
    mask_layer[left_clav > 0] = [255, 0, 255]
    mask_layer[right_clav > 0] = [0, 255, 255]

    alpha = 0.3
    cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0, overlay)

    if metrics["valid"]:
        rx, ry = metrics["right_medial_pt"]
        lx, ly = metrics["left_medial_pt"]
        vx, vy, x0, y0 = metrics["spine_line"]
        
        h = img.shape[0]
        # Draw the main red spine line
        if abs(vy) > 1e-6: 
            top_x = int(x0 - y0 * (vx / vy))
            bottom_x = int(x0 + (h - y0) * (vx / vy))
            cv2.line(overlay, (top_x, 0), (bottom_x, h), (0, 0, 255), 1, cv2.LINE_AA)

        # Helper to find where the white lines intersect the red line at a 90 degree angle
        def get_perpendicular_pt(px, py, vx, vy, x0, y0):
            t = (px - x0) * vx + (py - y0) * vy
            return int(x0 + t * vx), int(y0 + t * vy)

        r_intersect = get_perpendicular_pt(rx, ry, vx, vy, x0, y0)
        l_intersect = get_perpendicular_pt(lx, ly, vx, vy, x0, y0)

        # Draw the white measurement arms
        cv2.line(overlay, (rx, ry), r_intersect, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (lx, ly), l_intersect, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw the green anchor points
        cv2.circle(overlay, (rx, ry), 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, (lx, ly), 3, (0, 255, 0), -1, cv2.LINE_AA)
        
        text = f"Offset: {metrics['offset_ratio']:.1%} [{metrics['status']}]"
        color = (0, 255, 0) if metrics["status"] == "PASS" else (0, 0, 255)
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, overlay)

def process_batch():
    # 1. Create the routing directories cleanly with pathlib
    DIR_PASS.mkdir(parents=True, exist_ok=True)
    DIR_REJECT.mkdir(parents=True, exist_ok=True)
    DIR_ERROR.mkdir(parents=True, exist_ok=True)
    
    model = load_segmentation_model()
    
    # 2. Grab and NATURALLY SORT all valid images
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    
    # Helper function to split filename strings into sortable text and integers
    def natural_sort_key(path_obj):
        # e.g., "Normal-10.png" splits into ["Normal-", 10, ".png"]
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', path_obj.name)]

    # Find files and sort them using the custom key
    all_files = sorted([
        p for p in IMAGE_DIR.iterdir() 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ], key=natural_sort_key)
    
    if not all_files:
        print(f"❌ No images found in {IMAGE_DIR}")
        return
    
    total_images = len(all_files)
    print(f"\n🧪 Processing ALL {total_images} Images from {IMAGE_DIR}...\n")
    print(f"{'FILENAME':<30} | {'LEFT DIST':<10} | {'RIGHT DIST':<10} | {'OFFSET %':<10} | {'RESULT'}")
    print("-" * 80)

    # Tracking counters
    counts = {"pass": 0, "reject": 0, "error": 0}

    for image_path in all_files[100:1000]:  # Process only the first 100 images for this demo
        filename = image_path.name # pathlib automatically extracts the filename

        # Preprocess (Cast Path object to string for OpenCV safety)
        img_tensor, raw_img = preprocess_image(str(image_path))
        if img_tensor is None:
            continue

        # Inference
        with torch.no_grad():
            output = model(img_tensor)
        masks = torch.sigmoid(output[0]).cpu().numpy()

        # Math
        metrics, left_m, right_m, spine_m = calculate_rotation(masks)

        # --- ROUTING LOGIC ---
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

        # Save to the specific routed folder using pathlib's / operator
        save_path = out_dir / f"qc_{filename}"
        visualize_results(raw_img, (left_m, right_m, spine_m), metrics, str(save_path))

    # Print Final Summary
    print("\n" + "="*40)
    print("🎯 BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f"Total Processed : {total_images}")
    print(f"✅ PASS          : {counts['pass']}")
    print(f"⚠️ REJECT        : {counts['reject']}")
    print(f"🚨 ERRORS        : {counts['error']}")
    print("="*40)
    print(f"Check the '{OUTPUT_DIR_BASE}' directory for routed results.\n")

if __name__ == "__main__":
    process_batch()