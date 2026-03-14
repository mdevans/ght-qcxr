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
    run_parallax_engine,
    run_symmetry_engine,
    assess_rotation, 
    RotationClass
)

# ==========================================
# LEVEL 1: ATOMIC MATH TESTS
# ==========================================

def test_parallax_zero_rotation():
    """Verifies that run_parallax_engine returns 0% for aligned spine/sternum."""
    h, w = 500, 500
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:400, 248:252] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[150:350, 248:252] = 1
    mock_dict = {"Spine": spine, "Sternum": sternum}
    
    result = run_parallax_engine(mock_dict, (h, w), projection="PA")
    assert result["status"] == RotationClass.PASS
    assert pytest.approx(result["value"], abs=1e-4) == 0.0

def test_parallax_shifted_rotation():
    """Verifies shifted skeleton returns correct -10% offset."""
    h, w = 500, 500
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:400, 249:251] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[150:350, 299:301] = 1
    mock_dict = {"Spine": spine, "Sternum": sternum}
    
    result = run_parallax_engine(mock_dict, (h, w), projection="PA")
    # (250 - 300) / 500 = -0.10
    assert pytest.approx(result["value"], abs=1e-2) == -0.10

def test_lung_symmetry_perfect():
    """Verifies 0% contour skew for identical lung outlines."""
    h, w = 500, 500
    lung_r = np.zeros((h, w), dtype=np.uint8); lung_r[100:300, 100:200] = 1 
    lung_l = np.zeros((h, w), dtype=np.uint8); lung_l[100:300, 300:400] = 1
    mock_dict = {"Lung_Right": lung_r, "Lung_Left": lung_l}
    
    result = run_symmetry_engine(mock_dict)
    assert result["status"] == RotationClass.PASS
    assert pytest.approx(result["value"], abs=1e-4) == 0.0

# ==========================================
# LEVEL 2: GEOMETRIC & SANITY TESTS
# ==========================================

def test_rotation_extreme_tilt_handling():
    """Ensures fitLine handles significant postural tilt."""
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8)
    for i in range(200, 800): spine[i, i//2] = 1 
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[400:600, 250:260] = 1
    mock_dict = {"Spine": spine, "Sternum": sternum}
    
    result = run_parallax_engine(mock_dict, (h, w), projection="PA")
    assert "spine_line" in result["visuals"]

def test_error_handling_anatomic_failure():
    """Triggers total ERROR by failing sanity for both Skeleton and Lungs."""
    h, w = 512, 512
    # Skeleton fail: Sternum same size as Spine
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:400, 250:260] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[100:400, 260:270] = 1
    # Lung fail: Tiny dots
    lung_r = np.zeros((h, w), dtype=np.uint8); lung_r[10:12, 10:12] = 1
    lung_l = np.zeros((h, w), dtype=np.uint8); lung_l[10:12, 12:14] = 1
    
    mock_dict = {"Spine": spine, "Sternum": sternum, "Lung_Right": lung_r, "Lung_Left": lung_l}
    result = assess_rotation(mock_dict, (h, w))
    assert result["status"] == RotationClass.ERROR

# ==========================================
# LEVEL 3: CONSENSUS LOGIC TESTS
# ==========================================

def test_consensus_resilience_rule():
    """
    PROVES RESILIENCE: 
    If Parallax is MISSING (None), we need BOTH backups to pass.
    """
    h, w = 1000, 1000
    # Parallax is explicitly missing
    mock_dict = {
        "Spine": np.zeros((h, w), dtype=np.uint8), # Need spine for clavicle midline
        "Sternum": None, # This forces a skip of Parallax
        "Lung_Right": np.zeros((h, w), dtype=np.uint8),
        "Lung_Left": np.zeros((h, w), dtype=np.uint8),
        "Clavicle_Right": np.zeros((h, w), dtype=np.uint8),
        "Clavicle_Left": np.zeros((h, w), dtype=np.uint8)
    }
    
    # Set up perfect spine for midline
    mock_dict["Spine"][100:900, 498:502] = 1
    
    # Backups both PASS (Symmetrical)
    mock_dict["Lung_Right"][200:600, 100:300] = 1
    mock_dict["Lung_Left"][200:600, 700:900] = 1
    mock_dict["Clavicle_Right"][100:200, 200:450] = 1 # medial tip 450
    mock_dict["Clavicle_Left"][100:200, 550:800] = 1  # medial tip 550
    
    result = assess_rotation(mock_dict, (h, w))
    
    assert result["status"] == RotationClass.PASS
    assert "Dual Backup PASS" in result["reasoning"]

def test_consensus_unanimous_reject():
    """Ensures REJECT when Parallax is missing and both backups reject."""
    h, w = 1000, 1000
    
    # Skeleton: Spine exists, Sternum is None (Skip Parallax)
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 498:502] = 1
    
    # Lungs: Skewed perimeters (REJECT)
    lung_r = np.zeros((h, w), dtype=np.uint8); lung_r[100:800, 100:400] = 1
    lung_l = np.zeros((h, w), dtype=np.uint8); lung_l[400:450, 600:650] = 1 # Tiny
    
    # Clavicles: Skewed Medial Tips (REJECT)
    cr = np.zeros((h, w), dtype=np.uint8); cr[100:200, 100:490] = 1 # Medial at 490 (10px from spine)
    cl = np.zeros((h, w), dtype=np.uint8); cl[100:200, 600:900] = 1 # Medial at 600 (100px from spine)
    
    mock_dict = {
        "Spine": spine, "Sternum": None, 
        "Lung_Right": lung_r, "Lung_Left": lung_l,
        "Clavicle_Right": cr, "Clavicle_Left": cl
    }
    
    result = assess_rotation(mock_dict, (h, w))
    
    assert result["status"] == RotationClass.REJECT
    assert "Dual Backup REJECT" in result["reasoning"]

# ==========================================
# LEVEL 4: DISEASE RESILIENCE (CONTOUR TEST)
# ==========================================

def test_lung_contour_resilience_to_pathology():
    """Perimeter math must ignore internal 'holes' in segments."""
    h, w = 500, 500
    lung_solid = np.zeros((h, w), dtype=np.uint8); lung_solid[100:300, 100:200] = 1
    
    # Internal holes change area but NOT perimeter length
    lung_dirty = np.zeros((h, w), dtype=np.uint8); lung_dirty[100:300, 100:200] = 1
    lung_dirty[150:250, 120:180] = 0 
    
    mock_dict = {"Lung_Right": lung_solid, "Lung_Left": lung_dirty}
    result = run_symmetry_engine(mock_dict)
    
    assert pytest.approx(result["value"], abs=1e-4) == 0.0

# ==========================================
# LEVEL 5: PRODUCTION EDGE CASES (CLAVICLES)
# ==========================================

def test_clavicle_crossover_failure():
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 498:502] = 1
    # Large enough lungs to pass the ratio check
    lungs = np.zeros((h, w), dtype=np.uint8); lungs[200:800, 100:400] = 1
    
    # Create Crossover: Tips are large enough to pass ratio, but overlap/swap
    cr = np.zeros((h, w), dtype=np.uint8); cr[100:200, 300:600] = 1 # Tip at 600
    cl = np.zeros((h, w), dtype=np.uint8); cl[100:200, 500:800] = 1 # Tip at 500
    
    mock_dict = {
        "Clavicle_Right": cr, "Clavicle_Left": cl, 
        "Spine": spine, "Lung_Right": lungs, "Lung_Left": lungs
    }
    
    from rotation import is_clavicle_sane
    sane, msg = is_clavicle_sane(mock_dict, (h, w))
    assert not sane
    assert "Crossover" in msg

def test_parallax_missing_landmark_graceful_backup():
    """Verifies that missing Sternum pivots to Dual Backup (Lungs/Clavicles)."""
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 498:502] = 1
    
    # Perfectly symmetrical backups
    cr = np.zeros((h, w), dtype=np.uint8); cr[100:150, 200:450] = 1 
    cl = np.zeros((h, w), dtype=np.uint8); cl[100:150, 550:800] = 1 
    lung = np.zeros((h, w), dtype=np.uint8); lung[200:800, 100:400] = 1
    
    mock_dict = {
        "Spine": spine, "Sternum": None, 
        "Clavicle_Right": cr, "Clavicle_Left": cl,
        "Lung_Right": lung, "Lung_Left": lung
    }
    
    result = assess_rotation(mock_dict, (h, w))
    assert result["status"] == RotationClass.PASS
    assert "Dual Backup PASS" in result["reasoning"]

# ==========================================
# LEVEL 6: INTEGRATION
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
        print(f"  Lung Contour Skew: {m.get('lung_contour_skew', 0.0):.1%}")
        print(f"  Clavicle Skew: {m.get('clavicle_skew', 0.0):.1%}")
        print(f"  AI Verdict: {result['status'].upper()} | {result['reasoning']}")