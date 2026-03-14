import cv2
import numpy as np
from pathlib import Path
from PIL import ImageColor

# AI Pipeline Imports
from segment import load_cxas_model, load_xrv_model, predict_cxas, predict_xrv, resize_mask_back_to_orig, DEVICE
from preprocessing import preprocess_image, KEY_BASE_IMAGE, KEY_CXAS_TENSOR, KEY_XRV_TENSOR
from ensemble import blend_patient_masks
from tests.test_utils import TEST_IMAGES

# QA Module Imports
from inspiration import assess_inspiration

# --- CONFIGURATION ---
OUTPUT_DIR = Path("demo_output/inspiration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- VISUALIZATION CONSTANTS ---
def get_bgr(color_name: str) -> tuple[int, int, int]:
    """Converts a standard W3C color name to a strictly typed BGR 3-tuple for OpenCV."""
    r, g, b = ImageColor.getrgb(color_name)[:3]
    return b, g, r

COLOR_TEXT_MAIN     = get_bgr("white")
COLOR_DIAPHRAGM     = get_bgr("orange")
COLOR_LUNG_OUTLINE  = get_bgr("blue")
COLOR_BBOX          = get_bgr("cyan")
COLOR_RIB_VALID     = get_bgr("lime")
COLOR_RIB_INVALID   = get_bgr("yellow")
COLOR_DOME_LINE     = get_bgr("red")

# Text & Line Spacing Ratios (relative to scale_factor)
TEXT_PAD_X_RATIO = 20
TEXT_PAD_Y_RATIO = 30
LINE_STEP_MAIN_RATIO = 35
LINE_STEP_SUB_RATIO = 25
BOTTOM_REASONING_PAD_Y_RATIO = 10

# Blend opacity for the mask layer
BLEND_OPACITY = 0.3

def draw_inspiration_overlay(base_img: np.ndarray, blended_dict: dict, metrics: dict, reasoning: str) -> np.ndarray:
    """Refined visualization focusing on 2D anatomical expansion."""
    if len(base_img.shape) == 2:
        overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base_img.copy()
        
    h, w = overlay.shape[:2]
    scale_factor = max(0.5, h / 1000.0)
    
    # Scaling setup
    font_scale_main, font_scale_sub = 0.8 * scale_factor, 0.6 * scale_factor
    thick_line, thick_text = max(1, int(2 * scale_factor)), max(1, int(2 * scale_factor)) 
    pad_x, pad_y = int(TEXT_PAD_X_RATIO * scale_factor), int(TEXT_PAD_Y_RATIO * scale_factor)
    line_step_main, line_step_sub = int(LINE_STEP_MAIN_RATIO * scale_factor), int(LINE_STEP_SUB_RATIO * scale_factor)
    
    mask_layer = np.zeros_like(overlay)
    
    # 1. Draw Diaphragm (Orange)
    diaphragm = blended_dict.get("Diaphragm_Full")
    if diaphragm is not None:
        mask_layer[diaphragm > 0] = COLOR_DIAPHRAGM

    # 2. Draw Lungs (Blue) & Bounding Box (Cyan)
    lung_r, lung_l = blended_dict.get("Lung_Right"), blended_dict.get("Lung_Left")
    combined_lungs = np.zeros((h, w), dtype=np.uint8)
    for lung in [lung_r, lung_l]:
        if lung is not None and lung.max() > 0:
            combined_lungs = np.logical_or(combined_lungs, lung > 0).astype(np.uint8)
            contours, _ = cv2.findContours(lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, COLOR_LUNG_OUTLINE, thick_line)

    if combined_lungs.max() > 0:
        bx, by, bw, bh = cv2.boundingRect(combined_lungs)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), COLOR_BBOX, thick_line)
        # Centerlines
        cv2.line(overlay, (bx + bw//2, by), (bx + bw//2, by + bh), COLOR_BBOX, thick_line, cv2.LINE_AA)
        cv2.line(overlay, (bx, by + bh//2), (bx + bw, by + bh//2), COLOR_BBOX, thick_line, cv2.LINE_AA)

    # 3. Draw Ribs (Lime if Expansion Level, Yellow if masked by diaphragm)
    lowest_rib = metrics.get("expansion_level_rib", 0)
    for i in range(1, 13):
        rib = blended_dict.get(f"Posterior_Rib_{i}_Right")
        if rib is not None and rib.max() > 0:
            color = COLOR_RIB_VALID if i <= lowest_rib else COLOR_RIB_INVALID
            mask_layer[rib > 0] = color

    # 4. Final Blend
    cv2.addWeighted(mask_layer, BLEND_OPACITY, overlay, 1 - BLEND_OPACITY, 0, overlay)

    # 5. Clean Metrics Box
    text_lines = [
        f"Expansion Level (Rib): {metrics.get('expansion_level_rib', 0)}",
        f"Trusted Ribs: {metrics.get('trusted_rib_count', 0)}/12",
        f"Lung Aspect Ratio: {metrics.get('lung_aspect_ratio', 0):.2f}",
        f"Status: {metrics.get('status', 'ERROR').upper()}"
    ]
    
    y_off = pad_y
    for line in text_lines:
        cv2.putText(overlay, line, (pad_x, y_off), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, COLOR_TEXT_MAIN, thick_text, cv2.LINE_AA)
        y_off += line_step_main
        
    # Text Wrapping for reasoning
    max_w = w - (2 * pad_x)
    wrapped = []
    curr = ""
    for word in reasoning.split(" "):
        test = f"{curr} {word}".strip()
        if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, thick_text)[0][0] > max_w:
            wrapped.append(curr); curr = word
        else: curr = test
    if curr: wrapped.append(curr)
    
    y_off = h - (len(wrapped) * line_step_sub) - int(BOTTOM_REASONING_PAD_Y_RATIO * scale_factor)
    for line in wrapped:
        cv2.putText(overlay, line, (pad_x, y_off), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, COLOR_TEXT_MAIN, thick_text, cv2.LINE_AA)
        y_off += line_step_sub

    return overlay

def main():
    print(f"\n🚀 Loading Models into {DEVICE} memory...")
    cxas_model = load_cxas_model(DEVICE)
    xrv_model = load_xrv_model(DEVICE)

    if not TEST_IMAGES:
        print("❌ No images found in tests/test_data")
        return

    print(f"\n🧪 Running Inspiration Demo on {len(TEST_IMAGES)} images...")
    
    for img_path in TEST_IMAGES:
        print(f"Processing {img_path.name}...")
        
        try:
            # 1. Load & Preprocess
            data = preprocess_image(img_path)
            base_img = data[KEY_BASE_IMAGE]
            original_shape = base_img.shape # (H, W)
            
            # 2. Predict & Blend
            cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
            xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
            blended_raw = blend_patient_masks(cxas_raw, xrv_raw)
            
            # 3. Resize Masks to Original Image Aspect Ratio
            qa_ready_dict = {
                k: resize_mask_back_to_orig(v, original_shape) 
                for k, v in blended_raw.items() if v is not None
            }
            
            # 4. Run QA Logic
            result = assess_inspiration(qa_ready_dict, original_shape, projection="PA")
            
            # 5. Push status for drawing
            metrics_for_drawing = result.get("metrics", {})
            metrics_for_drawing["status"] = result.get("status", "UNKNOWN")
            
            # 6. Draw & Save Overlay
            overlay_img = draw_inspiration_overlay(
                base_img, 
                qa_ready_dict, 
                metrics_for_drawing, 
                result.get("reasoning", "Error")
            )
            
            # --- EXTENSION FIX ---
            orig_ext = img_path.suffix[1:] 
            out_path = OUTPUT_DIR / f"insp_{img_path.stem}_{orig_ext}.png"
            
            cv2.imwrite(str(out_path), overlay_img)
            
        except Exception as e:
            print(f"🚨 Failed to process {img_path.name}: {e}")
        
    print(f"\n✅ Demo complete! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()