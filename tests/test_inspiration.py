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
from inspiration import (
    passes_anatomic_sanity,
    get_diaphragm_dome_y,
    calculate_lowest_visible_rib,
    calculate_lung_aspect_ratio,
    assess_inspiration,
    InspirationClass
)

# ==========================================
# LEVEL 1: ATOMIC MATH & SANITY CHECKS
# ==========================================

def test_anatomic_sanity_gates():
    """Proves the pipeline correctly rejects physical impossibilities."""
    h, w = 512, 512
    
    # 1. Missing Diaphragm
    mock_dict = {"Lung_Right": np.ones((h, w))}
    is_sane, msg = passes_anatomic_sanity(mock_dict, (h, w))
    assert not is_sane
    assert "Diaphragm" in msg
    
    # 2. Impossibly High Diaphragm (e.g., in the neck)
    diaphragm = np.zeros((h, w))
    diaphragm[10, 256] = 1 # Y=10 is way too high
    mock_dict["Diaphragm_Full"] = diaphragm
    is_sane, msg = passes_anatomic_sanity(mock_dict, (h, w))
    assert not is_sane
    assert "impossibly high" in msg
    
    # 3. Lungs too small
    diaphragm = np.zeros((h, w))
    diaphragm[400, 256] = 1 # Normal low position
    tiny_lung = np.zeros((h, w))
    tiny_lung[200:210, 200:210] = 1 # Only 100 pixels
    mock_dict = {"Diaphragm_Full": diaphragm, "Lung_Right": tiny_lung}
    is_sane, msg = passes_anatomic_sanity(mock_dict, (h, w))
    assert not is_sane
    assert "impossibly small" in msg

def test_get_diaphragm_dome_y():
    """Proves the helper extracts the highest y-coordinate."""
    diaphragm = np.zeros((100, 100), dtype=np.uint8)
    diaphragm[80:100, 0:100] = 1
    diaphragm[75, 50] = 1 # The peak/dome
    
    assert get_diaphragm_dome_y(diaphragm) == 75

def test_calculate_lung_aspect_ratio():
    """Proves the geometric ratio accurately measures Height / Width."""
    r_lung = np.zeros((100, 100))
    l_lung = np.zeros((100, 100))
    
    # Right lung: Y from 10 to 90 (Height 80), X from 10 to 40
    r_lung[10:90, 10:40] = 1 
    # Left lung: Y from 20 to 80, X from 60 to 90
    l_lung[20:80, 60:90] = 1
    
    # Total combined Box: Y:(10-90) -> H=80. X:(10-90) -> W=80. Ratio = 1.0
    mock_dict = {"Lung_Right": r_lung, "Lung_Left": l_lung}
    assert calculate_lung_aspect_ratio(mock_dict) == 1.0

def test_lowest_visible_rib_logic():
    """Proves the 75% intersection threshold logic."""
    rib_8 = np.zeros((100, 100))
    rib_9 = np.zeros((100, 100))
    
    # Dome is at Y=50
    dome_y = 50
    
    # Rib 8 is completely above the dome (Y=30-40)
    rib_8[30:40, 10:20] = 1 
    
    # Rib 9 straddles the dome (Y=45-55) -> 5 pixels above, 5 below -> 50% visibility
    rib_9[45:55, 10:20] = 1 
    
    mock_dict = {
        "Posterior_Rib_8_Right": rib_8,
        "Posterior_Rib_9_Right": rib_9
    }
    
    highest_idx, is_reliable = calculate_lowest_visible_rib(mock_dict, dome_y, threshold=0.75)
    
    assert highest_idx == 8 # Rib 9 failed the 75% threshold!
    assert not is_reliable # Only 2 ribs found, fails the 5-rib reliability check

# ==========================================
# LEVEL 2: REAL DATA CALIBRATION TEST
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_inspiration_calibration_printout(image_path: Path, loaded_models):
    """
    Runs real test images through the pipeline to print Inspiration metrics.
    Run with `pytest -s` to view output for threshold tuning.
    """
    cxas_model, xrv_model = loaded_models
    
    # 1. Pipeline Prep
    data = preprocess_image(image_path)
    original_shape = data[KEY_BASE_IMAGE].shape # (H, W)
    
    # 2. AI Inference
    cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
    xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
    blended_raw = blend_patient_masks(cxas_raw, xrv_raw)
    
    # 3. Geometry Alignment
    qa_ready_dict = {
        k: resize_mask_back_to_orig(v, original_shape) 
        for k, v in blended_raw.items() if v is not None
    }
    
    # 4. Execute Orchestrator
    result = assess_inspiration(qa_ready_dict, original_shape, projection="PA")
    
    # 5. Output
    print(f"\n--- Inspiration Metrics for {image_path.name} ---")
    if result["status"] == InspirationClass.ERROR:
        print(f"  🚨 ERROR: {result['reasoning']}")
    else:
        metrics = result["metrics"]
        print(f"  Diaphragm Dome Y: {metrics['diaphragm_dome_y']}")
        print(f"  Lowest Visible Rib: {metrics['lowest_visible_rib']}")
        print(f"  Lung Aspect Ratio: {metrics['lung_aspect_ratio']:.2f}")
        print(f"  AI Verdict: {result['status']} | {result['reasoning']}")