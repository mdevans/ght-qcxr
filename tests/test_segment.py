import pytest
import numpy as np
import cv2
from pathlib import Path

# 1. Centralized Test Utils
from cxas.models.UNet.backbone_unet import BackboneUNet
from torchxrayvision.baseline_models.chestx_det import PSPNet
from tests.test_utils import TEST_IMAGES

# 2. Import the core utility and inference functions
from segment import (
    resize_mask_back_to_orig,
    extract_polygons_from_mask,
    predict_cxas,
    predict_xrv,
    DEVICE
)
from preprocessing import preprocess_image, KEY_CXAS_TENSOR, KEY_XRV_TENSOR, KEY_BASE_IMAGE

# 3. Import the anatomy contract
from ensemble import ANATOMY_RECIPES

# ==========================================
# ATOMIC GEOMETRY TESTS
# ==========================================

def test_resize_mask_back_to_orig_stretch():
    """Tests geometry restoration for a strictly squashed image."""
    original_shape = (200, 800) # (H, W)
    mask_512 = np.zeros((512, 512), dtype=np.uint8)
    
    # Draw a 100x100 square in the middle of the 512x512 mask
    mask_512[206:306, 206:306] = 1
    
    # Stretch it back to the extreme 200x800 aspect ratio
    restored_mask = resize_mask_back_to_orig(mask_512, original_shape)
    
    # Assert it matches the target clinical shape exactly
    assert restored_mask.shape == original_shape
    # Ensure no weird interpolation artifacts altered the binary nature of the mask
    assert restored_mask.dtype == np.uint8
    assert np.isin(restored_mask, [0, 1]).all()

def test_extract_polygons_from_mask():
    """Tests that the JSON-ready polygon extraction correctly shapes the coordinates."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:50, 40:50] = 1 
    
    polygons = extract_polygons_from_mask(mask, min_area=10)
    assert len(polygons) == 1
    # A 10x10 square usually yields 4 corners in cv2.CHAIN_APPROX_SIMPLE
    assert len(polygons[0]) >= 4 
    assert len(polygons[0][0]) == 2 # [x, y] coordinates

# ==========================================
# RAW MODEL SEGMENTATION TEST
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_model_segmentation_robustness(image_path: Path, loaded_models: tuple[BackboneUNet, PSPNet]):
    """
    Feeds REAL images into the models and exhaustively tests the resulting masks 
    for shape, type, and mathematical bounds.
    """
    cxas_model, xrv_model = loaded_models
    
    # 1. PREPROCESS
    data = preprocess_image(image_path)
    cxas_tensor = data[KEY_CXAS_TENSOR].to(DEVICE)
    xrv_tensor = data[KEY_XRV_TENSOR].to(DEVICE)
    
    # 2. RAW INFERENCE
    cxas_raw_data = predict_cxas(cxas_model, cxas_tensor)
    xrv_raw_data = predict_xrv(xrv_model, xrv_tensor)

    # 3. ROBUSTNESS VALIDATION AGAINST ANATOMY_RECIPES
    for target_name, recipe in ANATOMY_RECIPES.items():
        # Validate CXAS Output
        if recipe.cxas_key and recipe.cxas_key in cxas_raw_data:
            mask = cxas_raw_data[recipe.cxas_key]
            
            # Type & Bounds Checks
            assert mask.dtype == np.uint8, f"CXAS {recipe.cxas_key} is {mask.dtype}, expected uint8."
            assert mask.shape == (512, 512), f"CXAS {recipe.cxas_key} has corrupted dimensions {mask.shape}."
            
            # Mathematical Sanity Checks
            unique_vals = np.unique(mask)
            assert len(unique_vals) <= 2, f"CXAS {recipe.cxas_key} is not binary. Found values: {unique_vals}"
            assert mask.max() == 1, f"CXAS {recipe.cxas_key} mask values out of bounds [0, 1]."
                
        # Validate XRV Output
        if recipe.xrv_key and recipe.xrv_key in xrv_raw_data:
            mask = xrv_raw_data[recipe.xrv_key]
            
            # Type & Bounds Checks
            assert mask.dtype == np.uint8, f"XRV {recipe.xrv_key} is {mask.dtype}, expected uint8."
            assert mask.shape == (512, 512), f"XRV {recipe.xrv_key} has corrupted dimensions {mask.shape}."
            
            # Mathematical Sanity Checks
            unique_vals = np.unique(mask)
            assert len(unique_vals) <= 2, f"XRV {recipe.xrv_key} is not binary. Found values: {unique_vals}"
            assert mask.max() == 1, f"XRV {recipe.xrv_key} mask values out of bounds [0, 1]."

    # 4. PRIMARY ANCHOR SANITY CHECK
    # Even if peripheral ribs are missing, a valid CXR must ALWAYS contain lungs. 
    assert "class_085" in cxas_raw_data or "class_086" in cxas_raw_data, "CXAS failed to segment ANY lungs."
    assert "Right_Lung" in xrv_raw_data or "Left_Lung" in xrv_raw_data, "XRV failed to segment ANY lungs."