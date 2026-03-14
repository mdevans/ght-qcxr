import pytest
import numpy as np
from pathlib import Path

# 1. Centralized Test Utils
from tests.test_utils import TEST_IMAGES

# 2. AI Pipeline Imports
from segment import predict_cxas, predict_xrv, resize_mask_back_to_orig, DEVICE
from preprocessing import preprocess_image, KEY_CXAS_TENSOR, KEY_XRV_TENSOR, KEY_BASE_IMAGE
from ensemble import blend_patient_masks

# 3. QA Module Imports
from rotation import (
    calculate_parallax_offset, 
    assess_rotation, 
    RotationClass, 
    THRESHOLD_PARALLAX_PA_PASS, 
    THRESHOLD_PARALLAX_AP_PASS
)

# ==========================================
# LEVEL 1: ORIGINAL ATOMIC MATH TESTS
# ==========================================

def test_parallax_zero_rotation():
    """Proves perfectly aligned spine and sternum return 0% offset."""
    h, w = 500, 500
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:400, 248:252] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[150:350, 248:252] = 1
    
    mock_dict = {
        "Spine": spine,
        "Sternum": sternum,
        "Lung_Right": np.ones((10, 10), dtype=np.uint8),
        "Lung_Left": np.ones((10, 10), dtype=np.uint8)
    }
    
    result = calculate_parallax_offset(mock_dict, (h, w))
    assert result["valid"]
    assert pytest.approx(result["offset_ratio"], abs=1e-4) == 0.0

def test_parallax_shifted_rotation():
    """Proves 50px shift on 500px image results in a correctly signed -10% offset."""
    h, w = 500, 500
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:400, 249:251] = 1
    # Sternum shifted RIGHT (patient rotated left)
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[150:350, 299:301] = 1
    
    mock_dict = {
        "Spine": spine,
        "Sternum": sternum,
        "Lung_Right": np.ones((10, 10), dtype=np.uint8),
        "Lung_Left": np.ones((10, 10), dtype=np.uint8)
    }
    
    result = calculate_parallax_offset(mock_dict, (h, w))
    assert result["valid"]
    # Spine (250) - Sternum (300) = -50. Ratio = -50 / 500 = -0.10
    assert pytest.approx(result["offset_ratio"], abs=1e-2) == -0.10

def test_rotation_projection_thresholds():
    """Ensures PA is stricter than AP for rotation rejection."""
    def assess_logic(ratio, proj):
        limit = THRESHOLD_PARALLAX_PA_PASS if proj == "PA" else THRESHOLD_PARALLAX_AP_PASS
        # Using absolute ratio for limit checks, exactly like the module does
        return RotationClass.PASS if abs(ratio) <= limit else RotationClass.REJECT

    assert assess_logic(0.04, "PA") == RotationClass.REJECT
    assert assess_logic(0.04, "AP") == RotationClass.PASS
    assert assess_logic(-0.04, "PA") == RotationClass.REJECT # Negative values trigger reject too

# ==========================================
# LEVEL 2: GEOMETRIC PERMUTATIONS & TILT
# ==========================================

def test_rotation_direction_detection():
    """Proves math correctly assigns positive (Right) and negative (Left) signs."""
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 495:505] = 1
    
    # Left shift (Sternum=405, Spine=500) -> Positive ratio
    sternum_left = np.zeros((h, w), dtype=np.uint8); sternum_left[300:600, 400:410] = 1
    mock_left = {"Spine": spine, "Sternum": sternum_left, "Lung_Right": np.ones((10, 10), dtype=np.uint8), "Lung_Left": np.ones((10, 10), dtype=np.uint8)}
    assert 0.09 < calculate_parallax_offset(mock_left, (h, w))["offset_ratio"] < 0.10

    # Right shift (Sternum=595, Spine=500) -> Negative ratio
    sternum_right = np.zeros((h, w), dtype=np.uint8); sternum_right[300:600, 590:600] = 1
    mock_right = {"Spine": spine, "Sternum": sternum_right, "Lung_Right": np.ones((10, 10), dtype=np.uint8), "Lung_Left": np.ones((10, 10), dtype=np.uint8)}
    assert -0.10 < calculate_parallax_offset(mock_right, (h, w))["offset_ratio"] < -0.09

def test_rotation_extreme_tilt_handling():
    """Ensures axis fitting works with significant postural lean."""
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8)
    for i in range(200, 800): spine[i, i//2] = 1 
        
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[400:600, 250:260] = 1
    
    mock_dict = {"Spine": spine, "Sternum": sternum, "Lung_Right": np.ones((10, 10), dtype=np.uint8), "Lung_Left": np.ones((10, 10), dtype=np.uint8)}
    result = calculate_parallax_offset(mock_dict, (h, w))
    assert result["valid"]
    assert "spine_line" in result


# ==========================================
# LEVEL 3: REAL DATA INTEGRATION
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_rotation_calibration_printout(image_path: Path, loaded_models):
    """Runs real images through ensemble to calibrate thresholds."""
    cxas_model, xrv_model = loaded_models
    data = preprocess_image(image_path)
    original_shape = data[KEY_BASE_IMAGE].shape
    
    cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
    xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
    blended_raw = blend_patient_masks(cxas_raw, xrv_raw)
    
    qa_ready_dict = {
        k: resize_mask_back_to_orig(v, original_shape) 
        for k, v in blended_raw.items() if v is not None
    }
    
    result = assess_rotation(qa_ready_dict, original_shape, projection="PA")
    
    print(f"\n--- Rotation Metrics for {image_path.name} ---")
    if result["status"] == RotationClass.ERROR:
        print(f"  🚨 ERROR: {result['reasoning']}")
    else:
        metrics = result["metrics"]
        print(f"  Parallax Ratio: {metrics['parallax_ratio']:.1%}")
        print(f"  AI Verdict: {result['status'].upper()} | {result['reasoning']}")