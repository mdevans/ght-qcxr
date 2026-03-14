import pytest
import numpy as np
from pathlib import Path
import cv2

# Centralized Utils & Pipeline
from tests.test_utils import TEST_IMAGES
from segment import predict_cxas, predict_xrv, resize_mask_back_to_orig, DEVICE
from preprocessing import preprocess_image, KEY_CXAS_TENSOR, KEY_XRV_TENSOR, KEY_BASE_IMAGE
from ensemble import blend_patient_masks

# QA Module
from rotation import (
    calculate_parallax_offset, 
    calculate_lung_symmetry,
    assess_rotation, 
    RotationClass
)

# ==========================================
# LEVEL 1: ATOMIC MATH TESTS
# ==========================================

def test_parallax_zero_rotation():
    h, w = 500, 500
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:400, 248:252] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[150:350, 248:252] = 1
    mock_dict = {"Spine": spine, "Sternum": sternum}
    result = calculate_parallax_offset(mock_dict, (h, w))
    assert result["valid"]
    assert pytest.approx(result["offset_ratio"], abs=1e-4) == 0.0

def test_parallax_shifted_rotation():
    h, w = 500, 500
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:400, 249:251] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[150:350, 299:301] = 1
    mock_dict = {"Spine": spine, "Sternum": sternum}
    result = calculate_parallax_offset(mock_dict, (h, w))
    assert pytest.approx(result["offset_ratio"], abs=1e-2) == -0.10

def test_lung_symmetry_perfect():
    h, w = 500, 500
    lung_r = np.zeros((h, w), dtype=np.uint8); lung_r[100:300, 100:200] = 1 
    lung_l = np.zeros((h, w), dtype=np.uint8); lung_l[100:300, 300:400] = 1
    mock_dict = {"Lung_Right": lung_r, "Lung_Left": lung_l}
    result = calculate_lung_symmetry(mock_dict)
    assert result["valid"]
    assert pytest.approx(result["area_ratio"], abs=1e-4) == 0.0

def test_lung_symmetry_skewed():
    h, w = 500, 500
    lung_r = np.zeros((h, w), dtype=np.uint8); lung_r[100:300, 100:200] = 1 # 20k
    lung_l = np.zeros((h, w), dtype=np.uint8); lung_l[100:200, 300:400] = 1 # 10k
    mock_dict = {"Lung_Right": lung_r, "Lung_Left": lung_l}
    result = calculate_lung_symmetry(mock_dict)
    # Diff(10k) / Total(30k) = 0.333
    assert pytest.approx(result["area_ratio"], abs=1e-2) == 0.333

# ==========================================
# LEVEL 2: GEOMETRIC & SANITY TESTS
# ==========================================

def test_rotation_extreme_tilt_handling():
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8)
    for i in range(200, 800): spine[i, i//2] = 1 
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[400:600, 250:260] = 1
    mock_dict = {"Spine": spine, "Sternum": sternum}
    result = calculate_parallax_offset(mock_dict, (h, w))
    assert result["valid"]
    assert "spine_line" in result["visuals"]

def test_error_handling_anatomic_failure():
    h, w = 512, 512
    # Sternum area is way too big (same size as spine)
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:400, 250:260] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[100:400, 260:270] = 1
    mock_dict = {
        "Spine": spine, "Sternum": sternum,
        "Lung_Right": np.ones((100, 100), dtype=np.uint8),
        "Lung_Left": np.ones((100, 100), dtype=np.uint8)
    }
    result = assess_rotation(mock_dict, (h, w))
    assert result["status"] == RotationClass.ERROR
    assert "Anatomic Failure" in result["reasoning"]

# ==========================================
# LEVEL 3: CONSENSUS LOGIC TESTS
# ==========================================

def test_consensus_either_engine_pass_rules():
    """If one engine passes, verdict is PASS."""
    h, w = 1000, 1000
    # Perfect Lungs (PASS)
    lung_r = np.zeros((h, w), dtype=np.uint8); lung_r[200:600, 100:300] = 1
    lung_l = np.zeros((h, w), dtype=np.uint8); lung_l[200:600, 700:900] = 1
    # Rotated Skeleton (REJECT: ~10% parallax)
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 450:455] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[400:600, 550:555] = 1
    
    mock_dict = {"Spine": spine, "Sternum": sternum, "Lung_Right": lung_r, "Lung_Left": lung_l}
    result = assess_rotation(mock_dict, (h, w))
    assert result["status"] == RotationClass.PASS
    assert "Lung Skew: PASS" in result["reasoning"]

def test_consensus_unanimous_reject():
    """Both engines must reject for a REJECT status."""
    h, w = 1000, 1000
    # Skewed Lungs (REJECT)
    lung_r = np.zeros((h, w), dtype=np.uint8); lung_r[100:800, 100:400] = 1
    lung_l = np.zeros((h, w), dtype=np.uint8); lung_l[400:500, 600:700] = 1
    # Rotated Skeleton (REJECT)
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 400:405] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[400:600, 550:555] = 1
    
    mock_dict = {"Spine": spine, "Sternum": sternum, "Lung_Right": lung_r, "Lung_Left": lung_l}
    result = assess_rotation(mock_dict, (h, w))
    assert result["status"] == RotationClass.REJECT

# ==========================================
# LEVEL 4: INTEGRATION
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_rotation_calibration_printout(image_path: Path, loaded_models):
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
        m = result["metrics"]
        print(f"  Parallax: {m.get('parallax_ratio', 0.0):+.1%}")
        print(f"  Lung Skew: {m.get('lung_area_skew', 0.0):.1%}")
        print(f"  AI Verdict: {result['status'].upper()} | {result['reasoning']}")