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
from exposure import assess_exposure, BURNOUT_PIXEL_THRESHOLD

# --- CONFIGURATION ---
OUTPUT_DIR = Path("demo_output/exposure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- VISUALIZATION CONSTANTS ---
def get_bgr(color_name: str) -> tuple[int, int, int]:
    """Converts a standard W3C color name to a strictly typed BGR 3-tuple for OpenCV."""
    # Slicing [:3] drops any potential Alpha channel, guaranteeing exactly 3 values.
    r, g, b = ImageColor.getrgb(color_name)[:3]
    return b, g, r

COLOR_TEXT_MAIN             = get_bgr("white")
COLOR_HEART_OUTLINE         = get_bgr("red")
COLOR_SPINE_OUTLINE         = get_bgr("yellow")
COLOR_HEART_SPINE_OVERLAP   = get_bgr("magenta")
COLOR_LUNG_OUTLINE          = get_bgr("blue")
COLOR_LUNG_BURNOUT          = get_bgr("cyan")
COLOR_VERTEBRAE_OUTLINE     = get_bgr("lime") # 'lime' is the W3C standard for pure (0, 255, 0)

# Text & Line Spacing Ratios (relative to scale_factor)
TEXT_PAD_X_RATIO = 20
TEXT_PAD_Y_RATIO = 30
LINE_STEP_MAIN_RATIO = 35
LINE_STEP_SUB_RATIO = 25
BOTTOM_REASONING_PAD_Y_RATIO = 10

# Blend opacity for the mask layer
BLEND_OPACITY = 0.4

def draw_exposure_overlay(base_img: np.ndarray, blended_dict: dict, metrics: dict, reasoning: str) -> np.ndarray:
    """Draws a specialized clinical overlay showing exactly how the AI calculated Exposure."""
    # Convert grayscale to BGR for colored overlays
    if len(base_img.shape) == 2:
        overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base_img.copy()
        
    h, w = overlay.shape[:2]
    
    # --- DYNAMIC SCALING MATH ---
    scale_factor = max(0.5, h / 1000.0)
    
    font_scale_main = 0.8 * scale_factor
    font_scale_sub = 0.6 * scale_factor
    
    thick_line = max(1, int(2 * scale_factor))
    thick_text = max(1, int(2 * scale_factor)) 
    
    pad_x = int(TEXT_PAD_X_RATIO * scale_factor)
    pad_y = int(TEXT_PAD_Y_RATIO * scale_factor)
    line_step_main = int(LINE_STEP_MAIN_RATIO * scale_factor)
    line_step_sub = int(LINE_STEP_SUB_RATIO * scale_factor)
    
    mask_layer = np.zeros_like(overlay)
    
    # 1. Draw Lungs and Calculate Burnout Visualization
    lung_r = blended_dict.get("Lung_Right")
    lung_l = blended_dict.get("Lung_Left")
    combined_lungs = np.zeros((h, w), dtype=np.uint8)
    
    for lung in [lung_r, lung_l]:
        if lung is not None and lung.max() > 0:
            combined_lungs = np.logical_or(combined_lungs, lung > 0).astype(np.uint8)
            contours, _ = cv2.findContours(lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, COLOR_LUNG_OUTLINE, thick_line)

    # Highlight burned out pixels inside the lungs
    if combined_lungs.max() > 0:
        burned_out_mask = np.logical_and(combined_lungs > 0, base_img < BURNOUT_PIXEL_THRESHOLD)
        mask_layer[burned_out_mask] = COLOR_LUNG_BURNOUT 

    # 2. Draw Heart and Spine Overlap (Penetration Contrast Area)
    heart = blended_dict.get("Heart")
    spine = blended_dict.get("Spine")
    
    if heart is not None and heart.max() > 0:
        contours, _ = cv2.findContours(heart, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLOR_HEART_OUTLINE, thick_line)
        
    if spine is not None and spine.max() > 0:
        contours, _ = cv2.findContours(spine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLOR_SPINE_OUTLINE, thick_line)
        
    if heart is not None and spine is not None:
        overlap = np.logical_and(heart > 0, spine > 0)
        mask_layer[overlap] = COLOR_HEART_SPINE_OVERLAP 
        
    # 3. Draw Thoracic Vertebrae (T1-T12)
    for i in range(1, 13):
        t_vert = blended_dict.get(f"T{i}")
        if t_vert is not None and t_vert.max() > 0:
            contours, _ = cv2.findContours(t_vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, COLOR_VERTEBRAE_OUTLINE, thick_line)

    # 4. Blend the mask layer
    cv2.addWeighted(mask_layer, BLEND_OPACITY, overlay, 1 - BLEND_OPACITY, 0, overlay)

    # 5. Draw Metrics Text Box
    text_lines = [
        f"Vertebrae Count: {metrics.get('vertebrae_count', 0)}/12",
        f"Heart/Spine Contrast: {metrics.get('heart_spine_contrast', 0):.1%}",
        f"Lung Burnout: {metrics.get('lung_burnout_ratio', 0):.2%}",
        f"Status: {metrics.get('status', 'ERROR').upper()}"
    ]
    
    y_offset = pad_y
    for line in text_lines:
        cv2.putText(overlay, line, (pad_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, COLOR_TEXT_MAIN, thick_text, cv2.LINE_AA)
        y_offset += line_step_main
        
    # --- Robust Text Wrapping using cv2.getTextSize ---
    max_text_width = w - (2 * pad_x)
    
    wrapped_lines = []
    words = reasoning.split(" ")
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        text_width = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, thick_text)[0][0]
        
        if text_width > max_text_width and current_line:
            wrapped_lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
            
    if current_line:
        wrapped_lines.append(current_line)
    
    # Draw the dynamically wrapped lines at the bottom
    bottom_y_offset = h - (len(wrapped_lines) * line_step_sub) - int(BOTTOM_REASONING_PAD_Y_RATIO * scale_factor)
    for line in wrapped_lines:
        cv2.putText(overlay, line, (pad_x, bottom_y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, COLOR_TEXT_MAIN, thick_text, cv2.LINE_AA)
        bottom_y_offset += line_step_sub

    return overlay

def main():
    print(f"\n🚀 Loading Models into {DEVICE} memory...")
    cxas_model = load_cxas_model(DEVICE)
    xrv_model = load_xrv_model(DEVICE)

    if not TEST_IMAGES:
        print("❌ No images found in tests/test_data")
        return

    print(f"\n🧪 Running Exposure Demo on {len(TEST_IMAGES)} images...")
    
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
            result = assess_exposure(qa_ready_dict, base_img)
            
            # 5. Push status for drawing
            metrics_for_drawing = result.get("metrics", {})
            metrics_for_drawing["status"] = result.get("status", "UNKNOWN")
            
            # 6. Draw & Save Overlay
            overlay_img = draw_exposure_overlay(
                base_img, 
                qa_ready_dict, 
                metrics_for_drawing, 
                result.get("reasoning", "Error")
            )
            
            # File Extension safety fix
            orig_ext = img_path.suffix[1:] 
            out_path = OUTPUT_DIR / f"expo_{img_path.stem}_{orig_ext}.png"
            
            cv2.imwrite(str(out_path), overlay_img)
            
        except Exception as e:
            print(f"🚨 Failed to process {img_path.name}: {e}")
        
    print(f"\n✅ Demo complete! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()