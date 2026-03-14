import cv2
import numpy as np
from pathlib import Path
from PIL import ImageColor

from preprocessing import preprocess_image, KEY_BASE_IMAGE, KEY_CXAS_TENSOR, KEY_XRV_TENSOR
from segment import (
    load_cxas_model, predict_cxas, 
    load_xrv_model, predict_xrv, 
    resize_mask_back_to_orig, DEVICE
)
from ensemble import blend_patient_masks, ANATOMY_RECIPES
from tests.test_utils import TEST_IMAGES

# --- CONFIGURATION ---
OUTPUT_DIR = Path("demo_output/segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_bgr(color_name: str) -> tuple[int, int, int]:
    """Converts a standard W3C color name to a strictly typed BGR 3-tuple for OpenCV."""
    r, g, b = ImageColor.getrgb(color_name)[:3]
    return b, g, r

def draw_single_contour(base_img_bgr: np.ndarray, mask: np.ndarray, color_name: str, thick_line: int) -> np.ndarray:
    """Draws a single colored contour on a copy of the base image."""
    overlay = base_img_bgr.copy()
    bgr_tuple = get_bgr(color_name)
    
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, bgr_tuple, thickness=thick_line)
    return overlay

def draw_combined_contours(base_img_bgr: np.ndarray, mask_dict: dict[str, np.ndarray], thick_line: int) -> np.ndarray:
    """Draws ALL available contours onto a single master image."""
    overlay = base_img_bgr.copy()
    for anatomy_name, mask in mask_dict.items():
        if mask is None or mask.max() == 0: continue
        if "Rib" in anatomy_name: continue # Skip ribs to reduce clutter
            
        color_name = ANATOMY_RECIPES[anatomy_name].color if anatomy_name in ANATOMY_RECIPES else "lime"
        bgr_tuple = get_bgr(color_name)
        
        binary_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, bgr_tuple, thickness=thick_line)
        
    return overlay

def run_visual_demo():
    print(f"🚀 Initializing AI Models on {DEVICE}...")
    cxas_model = load_cxas_model(DEVICE)
    xrv_model = load_xrv_model(DEVICE)
    
    # Reverse mapping dictionaries to map raw class/model outputs back to our canonical Anatomy keys
    cxas_to_canonical = {r.cxas_key: name for name, r in ANATOMY_RECIPES.items() if r.cxas_key}
    xrv_to_canonical = {r.xrv_key: name for name, r in ANATOMY_RECIPES.items() if r.xrv_key}

    if not TEST_IMAGES:
        print("❌ No images found in tests/test_data.")
        return

    for image_path in TEST_IMAGES:
        stem = image_path.stem
        print(f"\n📂 Processing Image: {stem}")
        
        try:
            # 1. Pipeline Execution (Natively handles DICOM, PNG, JPG)
            data = preprocess_image(image_path)
            gray_img = data[KEY_BASE_IMAGE]
            orig_shape = gray_img.shape[:2]
            
            # Convert the strictly 1-channel grayscale image to 3-channel BGR for colored overlays
            base_img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            
            # Calculate dynamic thickness based on clinical array size
            scale_factor = max(0.5, orig_shape[0] / 1000.0)
            thick_line = max(1, int(3 * scale_factor))
            
            # 2. AI Inference
            cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
            xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
            blended_raw = blend_patient_masks(cxas_raw, xrv_raw)
            
            # 3. Resize and map to canonical names
            cxas_mapped = {
                cxas_to_canonical[k]: resize_mask_back_to_orig(v, orig_shape) 
                for k, v in cxas_raw.items() if k in cxas_to_canonical and v is not None
            }
            xrv_mapped = {
                xrv_to_canonical[k]: resize_mask_back_to_orig(v, orig_shape) 
                for k, v in xrv_raw.items() if k in xrv_to_canonical and v is not None
            }
            blended_mapped = {
                k: resize_mask_back_to_orig(v, orig_shape) 
                for k, v in blended_raw.items() if v is not None
            }

            # 4. Output Folder Setup
            # Append extension safely to handle files with same name but different format (e.g. cxr.dcm vs cxr.png)
            orig_ext = image_path.suffix[1:]
            img_out_dir = OUTPUT_DIR / f"{stem}_{orig_ext}"
            img_out_dir.mkdir(exist_ok=True)
            
            # Save the Master Combined Blueprint
            cv2.imwrite(
                str(img_out_dir / "0_COMPLETE_BLENDED_OVERLAY.png"), 
                draw_combined_contours(base_img_bgr, blended_mapped, thick_line)
            )
            
            # 5. Granular QA per Target
            for anatomy_name, recipe in ANATOMY_RECIPES.items():
                if "Rib" in anatomy_name: continue
                    
                c_mask = cxas_mapped.get(anatomy_name)
                x_mask = xrv_mapped.get(anatomy_name)
                b_mask = blended_mapped.get(anatomy_name)
                
                def has_data(m) -> bool: return m is not None and m.max() > 0

                # Skip creating folder if absolutely nothing was found
                if not has_data(c_mask) and not has_data(x_mask) and not has_data(b_mask):
                    continue
                    
                target_dir = img_out_dir / anatomy_name
                target_dir.mkdir(exist_ok=True)
                
                # Save strictly what exists
                if c_mask is not None and c_mask.max() > 0:
                    cv2.imwrite(str(target_dir / "1_cxas_mask.png"), c_mask * 255)
                    cv2.imwrite(str(target_dir / "4_cxas_overlay.png"), draw_single_contour(base_img_bgr, c_mask, recipe.color, thick_line))
                    
                if x_mask is not None and x_mask.max() > 0:
                    cv2.imwrite(str(target_dir / "2_xrv_mask.png"), x_mask * 255)
                    cv2.imwrite(str(target_dir / "5_xrv_overlay.png"), draw_single_contour(base_img_bgr, x_mask, recipe.color, thick_line))
                    
                if b_mask is not None and b_mask.max() > 0:
                    cv2.imwrite(str(target_dir / "3_blended_mask.png"), b_mask * 255)
                    cv2.imwrite(str(target_dir / "6_blended_overlay.png"), draw_single_contour(base_img_bgr, b_mask, recipe.color, thick_line))

        except Exception as e:
            print(f"🚨 Failed to process {image_path.name}: {e}")

    print(f"\n✅ Audit generation complete! Check the '{OUTPUT_DIR.name}' folder.")

if __name__ == "__main__":
    run_visual_demo()