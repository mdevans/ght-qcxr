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

def draw_single_contour(base_img: np.ndarray, mask: np.ndarray, color_name: str) -> np.ndarray:
    """Draws a single colored contour on a copy of the base image."""
    overlay = base_img.copy()
    rgb_tuple = ImageColor.getrgb(color_name)
    bgr_tuple = rgb_tuple[::-1]
    
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, bgr_tuple, thickness=3)
    return overlay

def draw_combined_contours(base_img: np.ndarray, mask_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Draws ALL available contours onto a single master image."""
    overlay = base_img.copy()
    for anatomy_name, mask in mask_dict.items():
        if mask is None or mask.max() == 0: continue
        if "Rib" in anatomy_name: continue # Skip ribs to reduce clutter
            
        color_name = ANATOMY_RECIPES[anatomy_name].color if anatomy_name in ANATOMY_RECIPES else "lime"
        rgb_tuple = ImageColor.getrgb(color_name)
        bgr_tuple = rgb_tuple[::-1]
        
        binary_mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, bgr_tuple, thickness=3)
        
    return overlay

def run_visual_demo():
    print(f"🚀 Initializing AI Models on {DEVICE}...")
    cxas_model = load_cxas_model(DEVICE)
    xrv_model = load_xrv_model(DEVICE)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
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
        
        # 2. Resize and map
        cxas_mapped = {cxas_to_canonical[k]: resize_mask_back_to_orig(v, orig_shape) for k, v in cxas_raw.items() if k in cxas_to_canonical}
        xrv_mapped = {xrv_to_canonical[k]: resize_mask_back_to_orig(v, orig_shape) for k, v in xrv_raw.items() if k in xrv_to_canonical}
        blended_mapped = {k: resize_mask_back_to_orig(v, orig_shape) for k, v in blended_raw.items()}

        # 3. Output Folder Setup
        img_out_dir = OUTPUT_DIR / stem
        img_out_dir.mkdir(exist_ok=True)
        
        # Save the Master Combined Blueprint
        cv2.imwrite(str(img_out_dir / "0_COMPLETE_BLENDED_OVERLAY.png"), draw_combined_contours(base_img, blended_mapped))
        
        # 4. Granular QA per Target
        for anatomy_name, recipe in ANATOMY_RECIPES.items():
            if "Rib" in anatomy_name: continue
                
            c_mask = cxas_mapped.get(anatomy_name)
            x_mask = xrv_mapped.get(anatomy_name)
            b_mask = blended_mapped.get(anatomy_name)
            
            # Helper to check if mask has actual data to determine if we create the folder
            def has_data(m) -> bool: return m is not None and m.max() > 0

            # Skip creating folder if absolutely nothing was found
            if not has_data(c_mask) and not has_data(x_mask) and not has_data(b_mask):
                continue
                
            target_dir = img_out_dir / anatomy_name
            target_dir.mkdir(exist_ok=True)
            
            # Save strictly what exists (Using explicit 'is not None' so Pylance type-narrows)
            if c_mask is not None and c_mask.max() > 0:
                cv2.imwrite(str(target_dir / "1_cxas_mask.png"), c_mask * 255)
                cv2.imwrite(str(target_dir / "4_cxas_overlay.png"), draw_single_contour(base_img, c_mask, recipe.color))
                
            if x_mask is not None and x_mask.max() > 0:
                cv2.imwrite(str(target_dir / "2_xrv_mask.png"), x_mask * 255)
                cv2.imwrite(str(target_dir / "5_xrv_overlay.png"), draw_single_contour(base_img, x_mask, recipe.color))
                
            if b_mask is not None and b_mask.max() > 0:
                cv2.imwrite(str(target_dir / "3_blended_mask.png"), b_mask * 255)
                cv2.imwrite(str(target_dir / "6_blended_overlay.png"), draw_single_contour(base_img, b_mask, recipe.color))

    print(f"\n✅ Audit generation complete! Check the '{OUTPUT_DIR.name}' folder.")

if __name__ == "__main__":
    run_visual_demo()