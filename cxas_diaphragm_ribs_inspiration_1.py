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

# --- CORE LOGIC ---

def calculate_inspiration_level(masks, image_height):
    metrics = {
        "valid": False, "status": "ERROR", "error_msg": "",
        "diaphragm_peak_y": 0, "visible_rib_count": 0, "rejection_reason": "N/A",
        "rib_masks": {}
    }

    # 1. Validate Diaphragm
    diaphragm_mask = keep_largest_blob_only(masks[RIGHT_HEMIDIAPHRAGM_IDX])

    if np.sum(diaphragm_mask) < (image_height * 5): # Heuristic: must be at least 5px high on average
        metrics["error_msg"] = "Right hemidiaphragm not found or too small"
        return metrics, diaphragm_mask

    # Find the highest point (peak) of the diaphragm mask.
    y_coords, _ = np.where(diaphragm_mask > 0)
    if len(y_coords) == 0:
        metrics["error_msg"] = "Empty diaphragm mask after cleaning"
        return metrics, diaphragm_mask

    diaphragm_peak_y = int(np.min(y_coords)) # Highest point (lowest y-value)
    metrics["diaphragm_peak_y"] = diaphragm_peak_y

    # 2. Count Visible Ribs
    visible_rib_count = 0
    rib_masks_viz = {}
    for i, rib_idx in enumerate(POSTERIOR_RIBS_RIGHT_INDICES):
        rib_mask = keep_largest_blob_only(masks[rib_idx])
        rib_masks_viz[i+1] = rib_mask

        total_pixels = np.sum(rib_mask)
        if total_pixels == 0:
            continue

        # NEW: Use a percentage-based rule for rib visibility.
        # Get all y-coordinates of the rib's pixels.
        y_coords_rib, _ = np.where(rib_mask > 0)
        
        # Count how many of those pixels are above the diaphragm peak.
        pixels_above_diaphragm = np.sum(y_coords_rib < diaphragm_peak_y)
        visibility_ratio = pixels_above_diaphragm / total_pixels
        
        # A rib is "visible" if a sufficient percentage of it is above the diaphragm.
        if visibility_ratio >= RIB_VISIBILITY_THRESHOLD:
            visible_rib_count += 1

    metrics["visible_rib_count"] = visible_rib_count
    metrics["rib_masks"] = rib_masks_viz

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

    if metrics["valid"]:
        diaphragm_peak_y = metrics["diaphragm_peak_y"]
        
        counted_rib_counter = 1 # Start numbering the counted ribs from 1

        # Color ribs based on whether they were counted or not
        for i, rib_mask in metrics["rib_masks"].items(): # i is the anatomical rib number (1-12)
            if np.sum(rib_mask) > 0:
                # Replicate the percentage logic for accurate color-coding
                total_pixels = np.sum(rib_mask)
                y_coords_rib, x_coords_rib = np.where(rib_mask > 0)
                pixels_above_diaphragm = np.sum(y_coords_rib < diaphragm_peak_y)
                visibility_ratio = pixels_above_diaphragm / total_pixels if total_pixels > 0 else 0

                # If the rib's visibility ratio meets the threshold, it was counted.
                if visibility_ratio >= RIB_VISIBILITY_THRESHOLD:
                    mask_layer[rib_mask > 0] = [0, 255, 0]  # Green for counted ribs

                    # --- Add number to counted rib ---
                    # Find the centroid of the rib mask to place the number
                    M = cv2.moments(rib_mask.astype(np.uint8))
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # --- NEW: Check if centroid is inside the mask ---
                        # If the centroid is outside the actual mask (e.g., for a 'C' shape), find a guaranteed internal point.
                        if rib_mask[cY, cX] == 0:
                            # Fallback to the median point of all mask coordinates, which is guaranteed to be inside.
                            median_idx = len(y_coords_rib) // 2
                            cX = x_coords_rib[median_idx]
                            cY = y_coords_rib[median_idx]

                        # Draw the sequential count number on the image
                        cv2.putText(overlay, str(counted_rib_counter), (cX - 7, cY + 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    counted_rib_counter += 1
                else:
                    # This rib was detected but did not meet the threshold.
                    mask_layer[rib_mask > 0] = [0, 255, 255] # Yellow for detected but not counted


    alpha = 0.3
    cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0, overlay)

    if metrics["valid"]:
        h, _ = img.shape[:2]
        peak_y = metrics["diaphragm_peak_y"]
        cv2.line(overlay, (0, peak_y), (img.shape[1], peak_y), (0, 0, 255), 1, cv2.LINE_AA)

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



### NOTES - This script processes chest X-ray images to evaluate the level of inspiration based on the visibility of the right hemidiaphragm and posterior ribs.
### It uses a pre-trained UNet model to segment anatomical structures, then applies heuristics to classify the inspiration level as PASS, REJECT_POOR, or REJECT_OVER. 
# The results are visualized and saved in separate directories for each category. The script also includes error handling for cases where the diaphragm cannot be properly identified.

# GEMINI CHATS SIGNIFICANT: 
"""
Does this method work for both PA and AP?

That's an excellent and very important question. It gets to the heart of ensuring a QA tool is robust and clinically aware.

Yes, the fundamental method used in the script—counting posterior ribs above the diaphragm—is anatomically valid for both PA (Posteroanterior) and AP (Anteroposterior) views. The script will correctly identify the diaphragm and posterior ribs in either projection and perform the count.

However, there is a crucial clinical distinction that you must be aware of:

The Difference in Interpretation
PA View (The Standard): This is the gold standard, usually taken with the patient standing and taking a deep breath. The thresholds in the script (9-10 ribs = PASS) are based on what is expected for a good quality, erect PA film.

AP View (Often Portable): This view is typically used for patients who are too unwell to stand, such as those in an ICU. The image is taken with the X-ray source in front of the patient and the detector behind them, often while they are supine (lying down) or semi-erect.

Physiological Limitation: In a supine position, the abdominal contents press up on the diaphragm, making it physically harder for the patient to take a full, deep breath.
Expected Outcome: As a result, an AP film will almost always show a lower level of inspiration than a PA film from the same patient. Seeing only 7 or 8 posterior ribs on an AP film is very common and often considered acceptable given the clinical circumstances.
How This Affects the Script
The script, in its current form, applies the strict "PA standard" to every image. This means:

It will correctly measure the rib count on an AP film.
It will likely classify many clinically acceptable AP films as "REJECT_POOR" because they won't meet the 9-rib threshold.
This isn't a failure of the measurement, but a lack of context in the classification step. The script doesn't know why the inspiration is poor; it only knows that it is.

Potential Solutions for a More Advanced System
To make the QA agent truly world-class, you could implement a mechanism to differentiate between the views:
1.  **Viewpoint Classification (Pixel-Based)**: The most robust method is to measure the **Spine-Sternum Parallax**. 
In an AP view, the anterior sternum is closer to the X-ray source and is magnified, increasing its horizontal distance from the posterior spine. In a PA view, they appear much closer. 
By segmenting the sternum and spine and measuring the horizontal offset of their centerlines, one can reliably distinguish AP from PA. 
This method is superior to using Cardiothoracic Ratio (CTR), which fails in cases of cardiomegaly.
2.  **Context-Aware Thresholds**: Once the view is determined, the system can apply different rules:
    *   **If PA**: Use the current strict thresholds (e.g., < 9 ribs is REJECT_POOR).
    *   **If AP**: Use a more lenient threshold (e.g., < 7 ribs is REJECT_POOR), or simply report the rib count without a "PASS/REJECT" status, leaving the final interpretation to the clinician who understands the context.

In summary, the script's anatomical logic is sound for both views, but its classification thresholds are optimized for the PA standard. It provides a correct quantitative measurement for any view, but the qualitative judgment (PASS/REJECT) may be too harsh for AP images.
"""