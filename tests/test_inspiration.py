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
    mock_dict = {"Lung_Right": np.ones((h, w), dtype=np.uint8)}
    is_sane, msg = passes_anatomic_sanity(mock_dict, (h, w))
    assert not is_sane
    assert "Diaphragm" in msg
    
    # 2. Impossibly High Diaphragm
    diaphragm = np.zeros((h, w), dtype=np.uint8)
    diaphragm[10, 256] = 1 
    mock_dict["Diaphragm_Full"] = diaphragm
    is_sane, msg = passes_anatomic_sanity(mock_dict, (h, w))
    assert not is_sane
    assert "is too high" in msg 
    
    # 3. Lungs too small
    diaphragm = np.zeros((h, w), dtype=np.uint8)
    diaphragm[400, 256] = 1 
    tiny_lung = np.zeros((h, w), dtype=np.uint8)
    tiny_lung[200:210, 200:210] = 1 
    mock_dict = {"Diaphragm_Full": diaphragm, "Lung_Right": tiny_lung}
    is_sane, msg = passes_anatomic_sanity(mock_dict, (h, w))
    assert not is_sane
    assert "impossibly small" in msg

def test_get_diaphragm_dome_y():
    """Proves the helper extracts the highest y-coordinate."""
    diaphragm = np.zeros((100, 100), dtype=np.uint8)
    diaphragm[80:100, 0:100] = 1
    diaphragm[75, 50] = 1 
    
    assert get_diaphragm_dome_y(diaphragm) == 75

def test_calculate_lung_aspect_ratio():
    """Proves the geometric ratio accurately measures Height / Width."""
    r_lung = np.zeros((100, 100), dtype=np.uint8)
    l_lung = np.zeros((100, 100), dtype=np.uint8)
    r_lung[10:90, 10:40] = 1 
    l_lung[20:80, 60:90] = 1
    
    mock_dict = {"Lung_Right": r_lung, "Lung_Left": l_lung}
    assert calculate_lung_aspect_ratio(mock_dict) == 1.0

def test_lowest_visible_rib_logic_2d_anatomical():
    """Proves the 2D mask-subtraction logic handles curved diaphragm edges."""
    h, w = 100, 100
    img_shape = (h, w)
    
    # Lung for structural trust
    lung_mask = np.zeros((h, w), dtype=np.uint8)
    lung_mask[10:90, 20:60] = 1 
    
    # Diaphragm mask (occupying bottom 30 pixels)
    diaphragm_mask = np.zeros((h, w), dtype=np.uint8)
    diaphragm_mask[70:100, 0:100] = 1

    # Rib 8: Completely above diaphragm mask (Y: 50-60)
    rib_8 = np.zeros((h, w), dtype=np.uint8)
    rib_8[50:60, 25:55] = 1 
    
    # Rib 11: Completely buried inside diaphragm mask (Y: 75-85)
    rib_11 = np.zeros((h, w), dtype=np.uint8)
    rib_11[75:85, 25:55] = 1 
    
    mock_dict = {
        "Lung_Right": lung_mask,
        "Posterior_Rib_8_Right": rib_8,
        "Posterior_Rib_11_Right": rib_11
    }
    
    # Execute with 2D mask instead of dome_y
    level, reliable, count = calculate_lowest_visible_rib(mock_dict, diaphragm_mask, img_shape)
    
    assert level == 8 
    assert count == 2 # Both are structurally sound, but only 8 is "visible"

def test_rib_leniency_proximal_2d():
    """Ensures Ribs 1-3 pass width checks with 2D return signature."""
    h, w = 100, 100
    lung_mask = np.zeros((h, w), dtype=np.uint8)
    lung_mask[10:90, 20:60] = 1
    diaphragm_mask = np.zeros((h, w), dtype=np.uint8) # Empty diaphragm for full visibility
    
    rib_1 = np.zeros((h, w), dtype=np.uint8)
    rib_1[15:20, 30:40] = 1 # Small proximal rib
    
    mock_dict = {
        "Lung_Right": lung_mask,
        "Posterior_Rib_1_Right": rib_1
    }
    
    level, _, count = calculate_lowest_visible_rib(mock_dict, diaphragm_mask, (h, w))
    assert level == 1
    assert count == 1

def test_rib_contiguity_gap_2d():
    """Proves contiguity protection using 2D mask logic without broadcasting errors."""
    h, w = 100, 100
    lung_mask = np.ones((h, w), dtype=np.uint8)
    diaphragm_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Ensure Rib masks are the same shape as the diaphragm_mask
    rib4 = np.zeros((h, w), dtype=np.uint8); rib4[10:15, 10:60] = 1
    rib5 = np.zeros((h, w), dtype=np.uint8); rib5[20:25, 10:60] = 1
    rib9 = np.zeros((h, w), dtype=np.uint8); rib9[60:65, 10:60] = 1
    
    mock_dict = {
        "Lung_Right": lung_mask,
        "Posterior_Rib_4_Right": rib4,
        "Posterior_Rib_5_Right": rib5,
        "Posterior_Rib_9_Right": rib9 # Gap of 4
    }
    
    # Now signatures and shapes align
    level, _, _ = calculate_lowest_visible_rib(mock_dict, diaphragm_mask, (h, w))
    assert level == 5
    
# ==========================================
# LEVEL 2: REAL DATA CALIBRATION TEST
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_inspiration_calibration_printout(image_path: Path, loaded_models):
    """Runs real test images through the pipeline to print Inspiration metrics."""
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
    
    result = assess_inspiration(qa_ready_dict, original_shape, projection="PA")
    
    print(f"\n--- Inspiration Metrics for {image_path.name} ---")
    if result["status"] == InspirationClass.ERROR:
        print(f"  🚨 ERROR: {result['reasoning']}")
    else:
        metrics = result["metrics"]
        print(f"  Expansion Level (Rib): {metrics['expansion_level_rib']}")
        print(f"  Trusted Ribs Found: {metrics['trusted_rib_count']}")
        print(f"  Lung Aspect Ratio: {metrics['lung_aspect_ratio']:.2f}")
        print(f"  AI Verdict: {result['status']} | {result['reasoning']}")