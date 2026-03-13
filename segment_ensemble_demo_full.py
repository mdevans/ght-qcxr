import cv2
import numpy as np
from pathlib import Path
from PIL import ImageColor

from preprocessing import preprocess_image, KEY_CXAS_TENSOR, KEY_XRV_TENSOR
from segment import (
    load_cxas_model, predict_cxas, 
    load_xrv_model, predict_xrv, 
    resize_mask_back_to_orig, DEVICE
)
from ensemble import blend_patient_masks, ANATOMY_RECIPES

# --- CONFIGURATION ---
INPUT_DIR = Path("tests/test_data")
OUTPUT_DIR = Path("demo_output")

def draw_single_contour(base_img: np.ndarray, mask: np.ndarray | None, color_name: str) -> np.ndarray:
    """Helper to draw a single colored contour on a copy of the base image."""
    overlay = base_img.copy()
    if mask is None or mask.max() == 0:
        return overlay # Return the plain image if there is no mask to draw
        
    # Get BGR tuple for OpenCV
    rgb_tuple = ImageColor.getrgb(color_name)
    bgr_tuple = rgb_tuple[::-1]
    
    # Extract contour
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw thick outline
    cv2.drawContours(overlay, contours, -1, bgr_tuple, thickness=3)
    return overlay

def run_visual_demo():
    print(f"🚀 Initializing AI Models on {DEVICE}...")
    cxas_model = load_cxas_model(DEVICE)
    xrv_model = load_xrv_model(DEVICE)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Build translation mappings to convert raw keys (e.g., class_135) to canonical names
    cxas_to_canonical = {r.cxas_key: name for name, r in ANATOMY_RECIPES.items() if r.cxas_key}
    xrv_to_canonical = {r.xrv_key: name for name, r in ANATOMY_RECIPES.items() if r.xrv_key}

    test_images = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    
    if not test_images:
        print("❌ No PNG or JPG files found in tests/test_data.")
        return

    for image_path in test_images:
        stem = image_path.stem
        print(f"\n📂 Processing Image: {stem}")
        
        base_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if base_img is None: continue
        orig_shape = base_img.shape[:2]
        
        # 1. Pipeline Execution
        data = preprocess_image(image_path)
        cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
        xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
        blended_raw = blend_patient_masks(cxas_raw, xrv_raw)
        
        # 2. Resize and map all outputs to canonical names
        cxas_mapped = {cxas_to_canonical[k]: resize_mask_back_to_orig(v, orig_shape) for k, v in cxas_raw.items() if k in cxas_to_canonical}
        xrv_mapped = {xrv_to_canonical[k]: resize_mask_back_to_orig(v, orig_shape) for k, v in xrv_raw.items() if k in xrv_to_canonical}
        blended_mapped = {k: resize_mask_back_to_orig(v, orig_shape) for k, v in blended_raw.items()}

        # 3. Generate Micro-Audits per Anatomy Target
        img_out_dir = OUTPUT_DIR / stem
        img_out_dir.mkdir(exist_ok=True)
        
        for anatomy_name, recipe in ANATOMY_RECIPES.items():
            c_mask = cxas_mapped.get(anatomy_name)
            x_mask = xrv_mapped.get(anatomy_name)
            b_mask = blended_mapped.get(anatomy_name)
            
            # Skip if none of the models found anything for this target (prevents empty folders)
            if c_mask is None and x_mask is None and b_mask is None:
                continue
                
            target_dir = img_out_dir / anatomy_name
            target_dir.mkdir(exist_ok=True)
            
            # A. Save isolated binary masks
            if c_mask is not None: cv2.imwrite(str(target_dir / "1_cxas_mask.png"), c_mask * 255)
            if x_mask is not None: cv2.imwrite(str(target_dir / "2_xrv_mask.png"), x_mask * 255)
            if b_mask is not None: cv2.imwrite(str(target_dir / "3_blended_mask.png"), b_mask * 255)
            
            # B. Save Contoured Overlays
            cv2.imwrite(str(target_dir / "4_cxas_overlay.png"), draw_single_contour(base_img, c_mask, recipe.color))
            cv2.imwrite(str(target_dir / "5_xrv_overlay.png"), draw_single_contour(base_img, x_mask, recipe.color))
            cv2.imwrite(str(target_dir / "6_blended_overlay.png"), draw_single_contour(base_img, b_mask, recipe.color))

    print(f"\n✅ Audit generation complete! Check the '{OUTPUT_DIR.name}' folder.")

if __name__ == "__main__":
    run_visual_demo()