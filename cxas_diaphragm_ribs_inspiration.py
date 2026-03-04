import torch
import numpy as np
import cv2
import re
from pathlib import Path
import torch.nn.functional as F
from PIL import Image

from cxas.models.UNet.backbone_unet import BackboneUNet

# --- CONFIGURATION ---
WEIGHTS_PATH = Path.home() / ".cxas" / "weights" / "UNet_ResNet50_default.pth"
IMAGE_DIR = Path("data/TB_Chest_Radiography_Database/Normal")
OUTPUT_DIR_BASE = Path("segment/cxas_inspiration_masks")

DIR_PASS = OUTPUT_DIR_BASE / "pass"
DIR_REJECT_POOR = OUTPUT_DIR_BASE / "reject_poor_inspiration"
DIR_ERROR = OUTPUT_DIR_BASE / "error"
DIR_REJECT_OVER = OUTPUT_DIR_BASE / "reject_over_inspiration"

# --- ANALYSIS THRESHOLDS ---
RIB_VISIBILITY_THRESHOLD = 0.75 # 75% of a rib's pixels must be above the diaphragm to be counted

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- ANATOMY INDICES ---
RIGHT_HEMIDIAPHRAGM_IDX = 107

# Posterior Ribs (Right Side)
POSTERIOR_RIBS_RIGHT_INDICES = [
    79, # 1st
    75, # 2nd
    71, # 3rd
    67, # 4th
    63, # 5th
    59, # 6th
    55, # 7th
    51, # 8th
    47, # 9th
    43, # 10th
    39, # 11th
    36, # 12th
]

# --- HELPER FUNCTIONS ---

def normalize(array: torch.Tensor) -> torch.Tensor:
    array = (array - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / (
        torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    )
    return array

def load_segmentation_model():
    print(f"🚀 Loading CXAS BackboneUNet (Inspiration Mode) on {DEVICE}...")
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

def check_for_missing_ribs(masks, posterior_rib_indices):
    """
    Performs a two-stage check to detect missing ribs up to the 9th rib.
    1. Checks if any essential rib (1-9) has a completely empty segmentation mask.
    2. Checks for anatomically inconsistent spacing between the medial points of detected ribs.
    Returns: (error_message, error_location) if a check fails, otherwise (None, None).
    """
    rib_spine_points = []

    # --- Stage 1: Check for empty masks and gather spine points ---
    for i, rib_idx in enumerate(posterior_rib_indices):
        anatomical_rib_number = i + 1
        if anatomical_rib_number > 9:
            continue  # Only check essential ribs 1-9

        rib_mask = keep_largest_blob_only(masks[rib_idx])

        if np.sum(rib_mask) == 0:
            # This is the "cxas missing rib class" check. The segmentation is absent.
            return f"Missing rib segmentation (Rib {anatomical_rib_number})", None

        # If the rib is present, find its medial point (spine connection)
        y_coords_rib, x_coords_rib = np.where(rib_mask > 0)
        medial_idx = np.argmax(x_coords_rib)  # For right ribs, medial point (spine) is max x
        medial_x_at_spine = x_coords_rib[medial_idx]
        medial_y_at_spine = y_coords_rib[medial_idx]
        rib_spine_points.append((anatomical_rib_number, medial_x_at_spine, medial_y_at_spine))

    # --- Stage 2: Check for inconsistent spacing using the gathered spine points ---
    if len(rib_spine_points) > 2:  # Need at least 3 points to check spacing
        y_positions = [item[2] for item in rib_spine_points]
        distances = np.diff(y_positions)
        if len(distances) > 0:
            median_dist = np.median(distances)
            if median_dist > 5 and np.any(distances > median_dist * 2.0):
                gap_index = np.argmax(distances)
                rib_before = rib_spine_points[gap_index]
                rib_after = rib_spine_points[gap_index + 1]
                location = (int((rib_before[1] + rib_after[1]) / 2), int((rib_before[2] + rib_after[2]) / 2))
                return "Inconsistent rib spacing", location

    return None, None  # All checks passed

# --- CORE LOGIC ---

def calculate_inspiration_level(masks, image_height):
    metrics = {
        "valid": False, "status": "ERROR", "error_msg": "",
        "diaphragm_peak_y": 0, "visible_rib_count": 0, "rejection_reason": "N/A", "missing_rib_location": None,
        "rib_masks": {}
    }

    # 1. Validate Diaphragm
    diaphragm_mask = keep_largest_blob_only(masks[RIGHT_HEMIDIAPHRAGM_IDX])
    
    if np.sum(diaphragm_mask) < (image_height * 5): # Heuristic: must be at least 5px high on average
        metrics["error_msg"] = "Right hemidiaphragm not found or too small"
        return metrics, diaphragm_mask

    # Find the highest point (peak) of the diaphragm mask EARLY.
    # This ensures the visualization function has a line to draw even if sanity checks fail.
    y_coords, _ = np.where(diaphragm_mask > 0)
    if len(y_coords) == 0:
        metrics["error_msg"] = "Empty diaphragm mask after cleaning"
        return metrics, diaphragm_mask
    diaphragm_peak_y = int(np.min(y_coords)) # Highest point (lowest y-value)
    metrics["diaphragm_peak_y"] = diaphragm_peak_y

    # --- Populate rib masks for visualization FIRST ---
    # This ensures that even if a later sanity check fails, we have the masks to draw.
    rib_masks_viz = {}
    for i, rib_idx in enumerate(POSTERIOR_RIBS_RIGHT_INDICES):
        rib_mask = keep_largest_blob_only(masks[rib_idx])
        rib_masks_viz[i+1] = rib_mask
    metrics["rib_masks"] = rib_masks_viz

    # --- Run missing rib detection before counting ---
    error_msg, error_location = check_for_missing_ribs(masks, POSTERIOR_RIBS_RIGHT_INDICES)
    if error_msg:
        metrics["error_msg"] = error_msg
        if error_location:
            metrics["missing_rib_location"] = error_location
        return metrics, diaphragm_mask

    # 2. Count Visible Ribs
    visible_rib_count = 0
    # Use the rib_masks_viz that we already populated
    for i, rib_mask in rib_masks_viz.items():
        total_pixels = np.sum(rib_mask)
        if total_pixels == 0:
            continue

        # NEW: Use a percentage-based rule for rib visibility.
        # Get all y-coordinates of the rib's pixels.
        y_coords_rib, x_coords_rib = np.where(rib_mask > 0)
        
        # Count how many of those pixels are above the diaphragm peak.
        pixels_above_diaphragm = np.sum(y_coords_rib < diaphragm_peak_y)
        visibility_ratio = pixels_above_diaphragm / total_pixels if total_pixels > 0 else 0

        is_counted = visibility_ratio >= RIB_VISIBILITY_THRESHOLD

        if is_counted:
            visible_rib_count += 1

    metrics["visible_rib_count"] = visible_rib_count

    # 3. Determine Status
    # Simplified criteria: Use rib count only.
    is_high_count = visible_rib_count > 10

    if is_high_count:
        metrics["status"] = "REJECT_OVER" # Over-inspiration
        metrics["rejection_reason"] = "High Rib Count"
    elif visible_rib_count >= 9:
        metrics["status"] = "PASS" # Good inspiration
    else:
        metrics["status"] = "REJECT_POOR" # Poor inspiration
        metrics["rejection_reason"] = "Low Rib Count"

    metrics["valid"] = True
    return metrics, diaphragm_mask

def visualize_inspiration(img, diaphragm_mask, metrics, save_path):
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_layer = overlay.copy()

    # Diaphragm in Orange
    mask_layer[diaphragm_mask > 0] = [0, 165, 255]

    # This block now runs for both valid and error cases, drawing whatever info was gathered.
    diaphragm_peak_y = metrics.get("diaphragm_peak_y")
    rib_masks = metrics.get("rib_masks", {})

    if diaphragm_peak_y and rib_masks:
        counted_rib_counter = 1
        for i, rib_mask in rib_masks.items():
            if np.sum(rib_mask) > 0:
                # Replicate the percentage logic for accurate color-coding
                total_pixels = np.sum(rib_mask)
                y_coords_rib, x_coords_rib = np.where(rib_mask > 0)
                pixels_above_diaphragm = np.sum(y_coords_rib < diaphragm_peak_y)
                visibility_ratio = pixels_above_diaphragm / total_pixels if total_pixels > 0 else 0

                if visibility_ratio >= RIB_VISIBILITY_THRESHOLD:
                    mask_layer[rib_mask > 0] = [0, 255, 0]  # Green for counted ribs

                    M = cv2.moments(rib_mask.astype(np.uint8))
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        if rib_mask[cY, cX] == 0:
                            median_idx = len(y_coords_rib) // 2
                            cX = x_coords_rib[median_idx]
                            cY = y_coords_rib[median_idx]

                        cv2.putText(overlay, str(counted_rib_counter), (cX - 7, cY + 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    counted_rib_counter += 1
                else:
                    mask_layer[rib_mask > 0] = [0, 255, 255] # Yellow for detected but not counted


    alpha = 0.3
    cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0, overlay)

    # Always try to draw the diaphragm line if it was calculated
    if diaphragm_peak_y:
        cv2.line(overlay, (0, int(diaphragm_peak_y)), (img.shape[1], int(diaphragm_peak_y)), (0, 0, 255), 1, cv2.LINE_AA)

    if metrics["valid"]:
        h, _ = img.shape[:2]

        status = metrics['status']
        if status == "PASS":
            status_text = "Good Inspiration"
            color = (0, 255, 0)
        elif status == "REJECT_OVER":
            status_text = f"Over-Inspiration ({metrics['rejection_reason']})"
            color = (255, 100, 0) # Cyan/Blue for over-inspiration
        else: # REJECT_POOR
            status_text = "Poor Inspiration"
            color = (0, 0, 255)
        
        # Display text at the bottom of the image
        text = f"Inspiration: {metrics['visible_rib_count']} Ribs Visible"
        cv2.putText(overlay, text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"[{status_text}]", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        h, _ = img.shape[:2]
        error_text = f"ERROR: {metrics.get('error_msg', 'Unknown')}"
        cv2.putText(overlay, error_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Visualize the location of a detected missing rib, even on error
    missing_rib_location = metrics.get("missing_rib_location")
    if missing_rib_location:
        x, y = missing_rib_location
        # Draw a prominent red 'X' to mark the spot
        cv2.drawMarker(overlay, (x, y), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=30, thickness=3, line_type=cv2.LINE_AA)

    cv2.imwrite(save_path, overlay)

# --- BATCH PROCESSOR ---

def process_batch():
    DIR_PASS.mkdir(parents=True, exist_ok=True)
    DIR_REJECT_POOR.mkdir(parents=True, exist_ok=True)
    DIR_REJECT_OVER.mkdir(parents=True, exist_ok=True)
    DIR_ERROR.mkdir(parents=True, exist_ok=True)

    model = load_segmentation_model()

    valid_extensions = {'.png', '.jpg', '.jpeg'}
    def natural_sort_key(path_obj):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path_obj.name)]

    all_files = sorted([p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions], key=natural_sort_key)
    if not all_files:
        print(f"❌ No images found in {IMAGE_DIR}")
        return

    test_files = all_files[:100] # Limit for testing

    print(f"\n🧪 Testing Inspiration Level on {len(test_files)} Images...\n")
    print(f"{'FILENAME':<30} | {'VISIBLE RIBS':<15} | {'RESULT'}")
    print("-" * 65)

    counts = {"pass": 0, "reject_poor": 0, "reject_over": 0, "error": 0}

    for image_path in test_files:
        filename = image_path.name
        try:
            pil_img = Image.open(image_path).convert(mode="RGB")
            orig_file_size = (pil_img.size[1], pil_img.size[0])
            orig_height = pil_img.size[1]
        except Exception as e:
            print(f"❌ Skipping {filename} - Error loading: {e}")
            continue

        cv_img_raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if cv_img_raw is None: continue
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cv_img_enhanced = clahe.apply(cv_img_raw)

        array = np.array(pil_img)
        array = np.transpose(array, [2, 0, 1])
        array = torch.tensor(array).float() / 255.0
        array = normalize(array).to(DEVICE)

        img_tensor = F.interpolate(array.unsqueeze(0), size=(512, 512))
        with torch.no_grad():
            output_dict = model({"data": img_tensor})

        pred = output_dict["segmentation_preds"][0].float()
        pred = F.interpolate(pred.unsqueeze(0), size=orig_file_size, mode="nearest")
        masks = pred[0].bool().cpu().numpy()

        metrics, diaphragm_mask = calculate_inspiration_level(masks, orig_height)

        if not metrics["valid"]:
            out_dir = DIR_ERROR
            counts["error"] += 1
            print(f"{filename:<30} | {'---':<15} | 🚨 ERROR ({metrics.get('error_msg')})")
        else:
            status = metrics["status"]
            if status == "PASS":
                out_dir = DIR_PASS
                counts["pass"] += 1
                icon = "✅"
            elif status == "REJECT_POOR":
                out_dir = DIR_REJECT_POOR
                counts["reject_poor"] += 1
                icon = "⚠️"
            else: # REJECT_OVER
                out_dir = DIR_REJECT_OVER
                counts["reject_over"] += 1
                icon = "💨"
            print(f"{filename:<30} | {metrics['visible_rib_count']:<15} | {icon} {status}")

        save_path = out_dir / f"qc_{filename}"
        visualize_inspiration(cv_img_enhanced, diaphragm_mask, metrics, str(save_path))

    print("\n" + "="*40)
    print(f"🎯 COMPLETE | PASS: {counts['pass']} | POOR: {counts['reject_poor']} | OVER: {counts['reject_over']} | ERROR: {counts['error']}")
    print("="*40)

if __name__ == "__main__":
    process_batch()