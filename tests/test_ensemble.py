import pytest
import numpy as np
import cv2
from pathlib import Path

# 1. Centralized Test Utils
from tests.test_utils import TEST_IMAGES

# 2. AI Pipeline Imports
from segment import predict_cxas, predict_xrv, resize_mask_back_to_orig, DEVICE
from preprocessing import preprocess_image, KEY_CXAS_TENSOR, KEY_XRV_TENSOR, KEY_BASE_IMAGE

from ensemble import (
    Strategy,
    apply_merge_strategy,
    blend_patient_masks,
    generate_preview_overlay,
    ANATOMY_RECIPES
)

# ==========================================
# 1. ATOMIC MATH TESTS (Pure boolean logic & Exceptions)
# ==========================================

def test_strategy_math_logic():
    """Proves the mathematical merging strategies work correctly at a pixel level."""
    # CXAS has the left pixel, XRV has the right pixel. They overlap in the bottom right.
    cxas = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    xrv =  np.array([[0, 1], [0, 1]], dtype=np.uint8)
    
    # UNION should combine pixels (3 pixels total)
    union_res = apply_merge_strategy(cxas, xrv, Strategy.UNION)
    assert union_res.max() == 1
    assert np.sum(union_res) == 3
    
    # INTERSECTION should keep ONLY the overlap (1 pixel total)
    intersect_res = apply_merge_strategy(cxas, xrv, Strategy.INTERSECTION)
    assert intersect_res.max() == 1
    assert np.sum(intersect_res) == 1
    
    # Fallbacks returning pure arrays safely
    assert np.array_equal(apply_merge_strategy(cxas, None, Strategy.XRV_PRIORITY), cxas)
    assert np.array_equal(apply_merge_strategy(None, xrv, Strategy.CXAS_PRIORITY), xrv)

def test_strategy_exceptions():
    """Proves that missing required masks raises explicit ValueErrors instead of silently returning None."""
    dummy_mask = np.ones((10, 10), dtype=np.uint8)
    
    # Both missing
    with pytest.raises(ValueError, match="Both CXAS and XRV masks are missing"):
        apply_merge_strategy(None, None, Strategy.UNION)
        
    # Missing CXAS when strictly required
    with pytest.raises(ValueError, match="strictly requires a CXAS mask"):
        apply_merge_strategy(None, dummy_mask, Strategy.CXAS_ONLY)
        
    # Missing XRV when strictly required (Intersection requires both)
    with pytest.raises(ValueError, match="strictly requires an XRV mask"):
        apply_merge_strategy(dummy_mask, None, Strategy.INTERSECTION)

def test_derived_diaphragm_full():
    """Proves the ensemble successfully synthesizes the Diaphragm_Full mask from available parts."""
    # Create fake 512x512 masks for the sub-components
    cxas_dict = {
        "class_106": np.zeros((512, 512), dtype=np.uint8), # Left Diaphragm
        "class_107": np.zeros((512, 512), dtype=np.uint8)  # Right Diaphragm
    }
    xrv_dict = {
        "Facies_Diaphragmatica": np.zeros((512, 512), dtype=np.uint8) # XRV Diaphragm
    }
    
    # Draw non-overlapping squares to represent the three parts
    cxas_dict["class_106"][10:20, 10:20] = 1
    cxas_dict["class_107"][30:40, 30:40] = 1
    xrv_dict["Facies_Diaphragmatica"][50:60, 50:60] = 1
    
    blended = blend_patient_masks(cxas_dict, xrv_dict)
    
    assert "Diaphragm_Full" in blended
    df = blended["Diaphragm_Full"]
    
    # The Full Diaphragm should contain pixels from ALL THREE locations
    assert df[15, 15] == 1 # Left is present
    assert df[35, 35] == 1 # Right is present
    assert df[55, 55] == 1 # XRV is present

# ==========================================
# 2. REAL CLINICAL DATA: IN-MEMORY BLENDING
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_blending_logic(image_path: Path, loaded_models):
    """
    Generates REAL dicts from the models in RAM and blends them.
    Strictly verifies that everything produced matches the ANATOMY_RECIPES contract.
    """
    cxas_model, xrv_model = loaded_models
    
    data = preprocess_image(image_path)
    cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
    xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
    
    blended_dict = blend_patient_masks(cxas_raw, xrv_raw)
        
    for target_key, mask_array in blended_dict.items():
        assert target_key in ANATOMY_RECIPES, f"Pipeline output an undocumented anatomy key: {target_key}"
        assert mask_array is not None, f"Pylance guard: {target_key} is None."
        assert mask_array.shape == (512, 512), f"{target_key} dimensions corrupted."
        assert mask_array.max() > 0, f"{target_key} was blended into an empty array."
        assert mask_array.dtype == np.uint8, f"{target_key} must be uint8, got {mask_array.dtype}"

# ==========================================
# 3. VISUAL OVERLAY TEST
# ==========================================

@pytest.mark.integration
def test_generate_preview_overlay_with_real_data(tmp_path: Path, loaded_models):
    """
    Takes ONE real image (even DICOM), stretches the masks to match the patient's 
    clinical resolution, and ensures the OpenCV overlay drawing logic writes a valid RGB image.
    """
    # We can use the very first test image, regardless of whether it is a DICOM or PNG!
    image_path = TEST_IMAGES[0] 
    cxas_model, xrv_model = loaded_models
    
    # 1. Pipeline Execution
    data = preprocess_image(image_path)
    base_img_array = data[KEY_BASE_IMAGE] 
    orig_shape = base_img_array.shape
    
    cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
    xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
    
    # 2. Blend the 512x512 masks
    blended_dict_512 = blend_patient_masks(cxas_raw, xrv_raw)
    
    # 3. GEOMETRY RESTORATION (Crucial! Masks must match the base_img before drawing)
    final_resized_masks = {
        name: resize_mask_back_to_orig(mask_512, orig_shape)
        for name, mask_512 in blended_dict_512.items()
    }
    
    # 4. Generate the Visual Output using the raw NumPy array
    out_img_path = tmp_path / f"preview_{image_path.stem}.png"
    generate_preview_overlay(base_img_array, final_resized_masks, out_img_path)
    
    # 5. Assertions
    assert out_img_path.exists(), "generate_preview_overlay failed to write file to disk."
    
    result_img = cv2.imread(str(out_img_path), cv2.IMREAD_COLOR)
    assert result_img is not None, "Saved preview image is corrupted."
    
    # The output RGB image should exactly match the dimensions of the original clinical array
    assert result_img.shape[:2] == orig_shape, f"Preview dimensions {result_img.shape[:2]} do not match base image {orig_shape}."