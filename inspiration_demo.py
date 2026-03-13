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
    # Convert grayscale to BGR for colored overlays
    if len(base_img.shape) == 2:
        overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base_img.copy()
        
    mask_layer = np.zeros_like(overlay)
    
    # 1. Draw Diaphragm (Orange)
    diaphragm = blended_dict.get("Diaphragm_Full")
    if diaphragm is not None:
        mask_layer[diaphragm > 0] = [0, 165, 255] # BGR Orange

    # 2. Draw Lungs (Blue Contours)
    for lung_key in ["Lung_Right", "Lung_Left"]:
        lung = blended_dict.get(lung_key)
        if lung is not None and lung.max() > 0:
            contours, _ = cv2.findContours(lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2) # Blue outline

    # 3. Draw Ribs (Green = Counted, Yellow = Below Dome/Ignored)
    lowest_rib = metrics.get("lowest_visible_rib", 0)
    for i in range(1, 13):
        rib = blended_dict.get(f"Posterior_Rib_{i}_Right")
        if rib is not None and rib.max() > 0:
            if i <= lowest_rib:
                mask_layer[rib > 0] = [0, 255, 0]     # Green (Passed threshold)
            else:
                mask_layer[rib > 0] = [0, 255, 255]   # Yellow (Below threshold)

    # 4. Blend the mask layer (30% opacity)
    cv2.addWeighted(mask_layer, 0.3, overlay, 0.7, 0, overlay)

    # 5. Draw the Diaphragm Dome "Finish Line" (Red)
    dome_y = metrics.get("diaphragm_dome_y", 0)
    h, w = overlay.shape[:2]
    if dome_y > 0:
        cv2.line(overlay, (0, dome_y), (w, dome_y), (0, 0, 255), 2, cv2.LINE_AA)

    # 6. Draw Metrics Text Box
    text_lines = [
        f"Dome Y: {dome_y}",
        f"Lowest Visible Rib: {lowest_rib}",
        f"Aspect Ratio: {metrics.get('lung_aspect_ratio', 0):.2f}",
        f"Status: {metrics.get('status', 'ERROR').upper()}"
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(overlay, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(overlay, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        y_offset += 30
        
    # Wrap and draw reasoning at the bottom
    reasoning_words = reasoning.split(" ")
    lines, current_line = [], []
    for word in reasoning_words:
        current_line.append(word)
        if len(" ".join(current_line)) > 60:
            lines.append(" ".join(current_line[:-1]))
            current_line = [word]
    lines.append(" ".join(current_line))
    
    y_offset = h - (len(lines) * 25) - 20
    for line in lines:
        cv2.putText(overlay, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset += 25

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
        
        # 1. Load & Preprocess
        base_img = preprocess_image(img_path)
        if base_img is None:
            continue
            
        data = preprocess_image(img_path)
        base_img = data[KEY_BASE_IMAGE]
        original_shape = base_img.shape
        
        # 2. Predict & Blend
        cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
        xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
        blended_raw = blend_patient_masks(cxas_raw, xrv_raw)
        
        # 3. Resize Masks to Original Image
        qa_ready_dict = {
            k: resize_mask_back_to_orig(v, original_shape) 
            for k, v in blended_raw.items() if v is not None
        }
        
        # 4. Run QA Logic
        result = assess_inspiration(qa_ready_dict, original_shape, projection="PA")
        
        # 5. Draw & Save Overlay
        overlay_img = draw_inspiration_overlay(
            base_img, 
            qa_ready_dict, 
            result.get("metrics", {}), 
            result.get("reasoning", "Error")
        )
        
        out_path = OUTPUT_DIR / f"insp_{img_path.stem}.png"
        cv2.imwrite(str(out_path), overlay_img)
        
    print(f"\n✅ Demo complete! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()