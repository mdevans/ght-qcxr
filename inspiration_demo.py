import cv2
import numpy as np
from pathlib import Path

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

def draw_inspiration_overlay(base_img: np.ndarray, blended_dict: dict, metrics: dict, reasoning: str) -> np.ndarray:
    """Draws a specialized clinical overlay showing exactly how the AI calculated Inspiration."""
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
    pad_x = int(20 * scale_factor)
    pad_y = int(30 * scale_factor)
    line_step_main = int(35 * scale_factor)
    line_step_sub = int(25 * scale_factor)
    
    mask_layer = np.zeros_like(overlay)
    
    # 1. Draw Diaphragm (Orange)
    diaphragm = blended_dict.get("Diaphragm_Full")
    if diaphragm is not None:
        mask_layer[diaphragm > 0] = [0, 165, 255] # BGR Orange

    # 2. Draw Lungs & Bounding Box
    lung_r = blended_dict.get("Lung_Right")
    lung_l = blended_dict.get("Lung_Left")
    
    combined_lungs = np.zeros((h, w), dtype=np.uint8)
    
    for lung in [lung_r, lung_l]:
        if lung is not None and lung.max() > 0:
            combined_lungs = np.logical_or(combined_lungs, lung > 0).astype(np.uint8)
            contours, _ = cv2.findContours(lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), thick_line) # Blue outline

    # --- NEW: ASPECT RATIO BOUNDING BOX (Cyan) ---
    if combined_lungs.max() > 0:
        # Get the strict geometric box encompassing BOTH lungs
        x, y, box_w, box_h = cv2.boundingRect(combined_lungs)
        
        # Draw the bounding box
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (255, 255, 0), thick_line)
        
        # Draw CAD-style measurement axes (Crosshairs through the center of the box)
        center_x = x + (box_w // 2)
        center_y = y + (box_h // 2)
        
        # Vertical height line
        cv2.line(overlay, (center_x, y), (center_x, y + box_h), (255, 255, 0), thick_line, cv2.LINE_AA)
        # Horizontal width line
        cv2.line(overlay, (x, center_y), (x + box_w, center_y), (255, 255, 0), thick_line, cv2.LINE_AA)
        
        # Label the axes
        cv2.putText(overlay, f"W", (x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (255, 255, 0), thick_text, cv2.LINE_AA)
        cv2.putText(overlay, f"H", (center_x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (255, 255, 0), thick_text, cv2.LINE_AA)

    # 3. Draw Ribs (Green = Counted, Yellow = Below Dome/Ignored)
    lowest_rib = metrics.get("lowest_visible_rib", 0)
    for i in range(1, 13):
        rib = blended_dict.get(f"Posterior_Rib_{i}_Right")
        if rib is not None and rib.max() > 0:
            if i <= lowest_rib:
                mask_layer[rib > 0] = [0, 255, 0]     # Green
            else:
                mask_layer[rib > 0] = [0, 255, 255]   # Yellow

    # 4. Blend the mask layer (30% opacity)
    cv2.addWeighted(mask_layer, 0.3, overlay, 0.7, 0, overlay)

    # 5. Draw the Diaphragm Dome "Finish Line" (Red)
    dome_y = metrics.get("diaphragm_dome_y", 0)
    if dome_y > 0:
        cv2.line(overlay, (0, dome_y), (w, dome_y), (0, 0, 255), thick_line, cv2.LINE_AA)

    # 6. Draw Metrics Text Box (Pure White)
    text_lines = [
        f"Dome Y: {dome_y}",
        f"Lowest Visible Rib: {lowest_rib}",
        f"Aspect Ratio (H/W): {metrics.get('lung_aspect_ratio', 0):.2f}",
        f"Status: {metrics.get('status', 'ERROR').upper()}"
    ]
    
    y_offset = pad_y
    for line in text_lines:
        cv2.putText(overlay, line, (pad_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (255, 255, 255), thick_text, cv2.LINE_AA)
        y_offset += line_step_main
        
    # Wrap and draw reasoning at the bottom (Pure White)
    reasoning_words = reasoning.split(" ")
    lines, current_line = [], []
    for word in reasoning_words:
        current_line.append(word)
        char_limit = int(60 * (w / 1000.0) / max(0.5, scale_factor)) 
        if len(" ".join(current_line)) > char_limit:
            lines.append(" ".join(current_line[:-1]))
            current_line = [word]
    lines.append(" ".join(current_line))
    
    y_offset = h - (len(lines) * line_step_sub) - int(10 * scale_factor)
    for line in lines:
        cv2.putText(overlay, line, (pad_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255, 255, 255), thick_text, cv2.LINE_AA)
        y_offset += line_step_sub

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
            # Grab the suffix without the dot (e.g., 'dcm' or 'jpg')
            orig_ext = img_path.suffix[1:] 
            out_path = OUTPUT_DIR / f"insp_{img_path.stem}_{orig_ext}.png"
            
            cv2.imwrite(str(out_path), overlay_img)
            
        except Exception as e:
            print(f"🚨 Failed to process {img_path.name}: {e}")
        
    print(f"\n✅ Demo complete! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()