import cv2
import numpy as np
from pathlib import Path

# --- Pipeline Imports ---
from segment import load_cxas_model, load_xrv_model, predict_cxas, predict_xrv, resize_mask_back_to_orig, DEVICE
from preprocessing import preprocess_image, KEY_BASE_IMAGE, KEY_CXAS_TENSOR, KEY_XRV_TENSOR
from ensemble import blend_patient_masks
from tests.test_utils import TEST_IMAGES

# --- QA Module Imports ---
from rotation import assess_rotation, RotationClass

# --- Config ---
OUTPUT_DIR = Path("demo_output/rotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# BGR Colors for OpenCV
COLOR_SPINE = (0, 0, 255)    # Red
COLOR_STERNUM = (0, 255, 0)  # Green
COLOR_TEXT = (255, 255, 255) # White
COLOR_BG = (0, 0, 0)         # Black

def draw_rotation_overlay(base_img: np.ndarray, masks: dict, result: dict) -> np.ndarray:
    """Draws masks, structural axes, and parallax metrics with dynamic scaling."""
    # Ensure image is BGR for colored drawing
    if len(base_img.shape) == 2:
        overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base_img.copy()

    h, w = overlay.shape[:2]
    
    # --- Dynamic Scaling Logic ---
    # Base scale off image width (1000px = scale 1.0)
    scale = max(0.5, w / 1000.0)
    thick = max(1, int(2 * scale))
    font_scale = 0.8 * scale
    pad = int(10 * scale) # Padding for text boxes
    point_radius = max(3, int(6 * scale))
    
    # 1. MASK OVERLAYS
    mask_layer = np.zeros_like(overlay)
    if "Spine" in masks and masks["Spine"].max() > 0:
        mask_layer[masks["Spine"] > 0] = COLOR_SPINE
    if "Sternum" in masks and masks["Sternum"].max() > 0:
        mask_layer[masks["Sternum"] > 0] = COLOR_STERNUM
        
    # Alpha blend masks with a 30% opacity
    cv2.addWeighted(mask_layer, 0.3, overlay, 0.7, 0, overlay)

    # 2. GEOMETRY LINES (If valid)
    if result["status"] != RotationClass.ERROR:
        vis = result["metrics"]["visuals"]
        sp_x, st_x, eval_y = vis["eval_pt"]
        
        # Draw Spine Line (Red)
        vx_sp, vy_sp, x0_sp, y0_sp = vis["spine_line"]
        if abs(vy_sp) > 1e-6:
            top_x_sp = int(x0_sp - y0_sp * (vx_sp / vy_sp))
            bot_x_sp = int(x0_sp + (h - y0_sp) * (vx_sp / vy_sp))
            cv2.line(overlay, (top_x_sp, 0), (bot_x_sp, h), COLOR_SPINE, thick, cv2.LINE_AA)
            
        # Draw Sternum Line (Green)
        vx_st, vy_st, x0_st, y0_st = vis["sternum_line"]
        if abs(vy_st) > 1e-6:
            top_x_st = int(x0_st - y0_st * (vx_st / vy_st))
            bot_x_st = int(x0_st + (h - y0_st) * (vx_st / vy_st))
            cv2.line(overlay, (top_x_st, 0), (bot_x_st, h), COLOR_STERNUM, thick, cv2.LINE_AA)

        # Draw the Parallax "Bridge"
        cv2.line(overlay, (sp_x, eval_y), (st_x, eval_y), COLOR_TEXT, thick, cv2.LINE_AA)
        cv2.circle(overlay, (sp_x, eval_y), point_radius, COLOR_SPINE, -1, cv2.LINE_AA)
        cv2.circle(overlay, (st_x, eval_y), point_radius, COLOR_STERNUM, -1, cv2.LINE_AA)

    # 3. TEXT INFO OVERLAY
    is_pass = result["status"] == RotationClass.PASS
    status_color = (0, 255, 0) if is_pass else (0, 0, 255)
    
    info_lines = [f"Verdict: {result['status'].upper()}"]
    if result["status"] != RotationClass.ERROR:
        info_lines.append(f"Parallax Ratio: {result['metrics']['parallax_ratio']:+.1%}")
    
    start_x = int(20 * scale)
    y_offset = int(40 * scale)
    
    for i, line in enumerate(info_lines):
        # Calculate exact text bounds
        (t_w, t_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        
        # Draw fitted background box
        cv2.rectangle(overlay, 
                      (start_x - pad, y_offset - t_h - pad), 
                      (start_x + t_w + pad, y_offset + baseline + pad), 
                      COLOR_BG, -1)
                      
        # Draw text
        color = status_color if i == 0 else COLOR_TEXT
        cv2.putText(overlay, line, (start_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, thick, cv2.LINE_AA)
        
        # Move down for next line
        y_offset += t_h + baseline + (pad * 2)
        
    # 4. REASONING STRING (Bottom Left)
    reasoning = result["reasoning"]
    r_font_scale = font_scale * 0.85  # Slightly smaller for the reasoning text
    (r_w, r_h), r_baseline = cv2.getTextSize(reasoning, cv2.FONT_HERSHEY_SIMPLEX, r_font_scale, thick)
    
    r_y = h - int(40 * scale)
    
    # Background box for reasoning
    cv2.rectangle(overlay, 
                  (start_x - pad, r_y - r_h - pad), 
                  (start_x + r_w + pad, r_y + r_baseline + pad), 
                  COLOR_BG, -1)
                  
    cv2.putText(overlay, reasoning, (start_x, r_y), cv2.FONT_HERSHEY_SIMPLEX, 
                r_font_scale, (0, 255, 255), thick, cv2.LINE_AA)

    return overlay

def main():
    print("🚀 Loading models for Rotation Demo on MPS...")
    cxas_model = load_cxas_model(DEVICE)
    xrv_model = load_xrv_model(DEVICE)

    for img_path in TEST_IMAGES:
        print(f"\nProcessing {img_path.name}...")
        
        # 1. Preprocess
        data = preprocess_image(img_path)
        base_img = data[KEY_BASE_IMAGE]
        
        # 2. Predict & Blend
        cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
        xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
        blended = blend_patient_masks(cxas_raw, xrv_raw)
        
        # 3. Resize Masks to Original Image Scale
        masks = {k: resize_mask_back_to_orig(v, base_img.shape) for k, v in blended.items() if v is not None}
        
        # 4. Assess Rotation (Defaults to PA projection)
        result = assess_rotation(masks, base_img.shape, projection="PA")
        
        # 5. Visualize & Save
        overlay = draw_rotation_overlay(base_img, masks, result)
        out_path = OUTPUT_DIR / f"rot_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"✅ Saved demo output to {out_path}")

if __name__ == "__main__":
    main()