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
from projection import (
    is_frontal_view, 
    get_fov_completeness, 
    determine_ap_pa_projection, 
    assess_projection,
    ProjectionClass
)

# ==========================================
# LEVEL 1: FRONTAL VS LATERAL TESTS
# ==========================================

def test_frontal_confirmed_alignment():
    """Spine and Sternum are perfectly stacked in the midline."""
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 490:510] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[400:600, 490:510] = 1
    
    mock_dict = {"Spine": spine, "Sternum": sternum}
    is_frontal, msg = is_frontal_view(mock_dict, (h, w))
    
    assert is_frontal is True
    assert "Frontal View Confirmed" in msg

def test_lateral_detected_by_offset():
    """Spine is at x=200, Sternum is at x=800. Clearly lateral."""
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 190:210] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[400:600, 790:810] = 1
    
    mock_dict = {"Spine": spine, "Sternum": sternum}
    is_frontal, msg = is_frontal_view(mock_dict, (h, w))
    
    assert is_frontal is False
    assert "Lateral" in msg

# ==========================================
# LEVEL 2: FOV COMPLETENESS (CLIPPING)
# ==========================================

def test_fov_perfectly_complete():
    """Lungs are centered with plenty of margin."""
    h, w = 1000, 1000
    lung = np.zeros((h, w), dtype=np.uint8); lung[100:800, 200:400] = 1
    
    mock_dict = {"Lung_Right": lung, "Lung_Left": lung}
    result = get_fov_completeness(mock_dict, (h, w))
    
    assert result["complete"] is True
    assert "Complete" in result["reason"]

def test_fov_clipping_single_edge_apices():
    """Only the top of the lungs is clipped."""
    h, w = 1000, 1000
    lung = np.zeros((h, w), dtype=np.uint8); lung[0:800, 200:400] = 1 # Hits y=0
    
    mock_dict = {"Lung_Right": lung, "Lung_Left": lung}
    result = get_fov_completeness(mock_dict, (h, w))
    
    assert result["complete"] is False
    assert "Apices" in result["clipping"][0]
    assert len(result["clipping"]) == 1

def test_fov_clipping_lateral():
    """Lung mask hits the left edge."""
    h, w = 1000, 1000
    lung = np.zeros((h, w), dtype=np.uint8); lung[200:800, 0:400] = 1 # Hits x=0
    
    mock_dict = {"Lung_Right": lung, "Lung_Left": lung}
    result = get_fov_completeness(mock_dict, (h, w))
    
    assert result["complete"] is False
    assert "Lateral Ribs" in result["clipping"]

# ==========================================
# LEVEL 3: PROJECTION HEURISTICS (AP VS PA)
# ==========================================

def test_projection_pa_low_clavicles():
    """PA Simulation: Clavicles are deep into the lung field."""
    h, w = 1000, 1000
    lung = np.zeros((h, w), dtype=np.uint8); lung[100:900, 200:400] = 1 # Apex y=100, Height=800
    clav = np.zeros((h, w), dtype=np.uint8); clav[290:310, 250:350] = 1 # Mean y=300
    # rel_height = (300 - 100) / 800 = 0.25 (> 0.02)
    
    mock_dict = {"Lung_Right": lung, "Lung_Left": lung, "Clavicle_Right": clav, "Clavicle_Left": clav}
    proj, rel_h = determine_ap_pa_projection(mock_dict)
    
    assert proj == ProjectionClass.PA
    assert pytest.approx(rel_h, abs=1e-3) == 0.25

def test_projection_ap_high_clavicles():
    """AP Simulation: Clavicles are above the lung apex."""
    h, w = 1000, 1000
    lung = np.zeros((h, w), dtype=np.uint8); lung[200:900, 200:400] = 1 # Apex y=200
    clav = np.zeros((h, w), dtype=np.uint8); clav[170:190, 250:350] = 1 # Mean y=180
    # rel_height = (180 - 200) / 700 = -0.028 (< 0.02)
    
    mock_dict = {"Lung_Right": lung, "Lung_Left": lung, "Clavicle_Right": clav, "Clavicle_Left": clav}
    proj, rel_h = determine_ap_pa_projection(mock_dict)
    
    assert proj == ProjectionClass.AP
    assert rel_h < 0.0

def test_projection_missing_clavicles():
    """Fails gracefully if clavicles aren't segmented."""
    h, w = 1000, 1000
    lung = np.zeros((h, w), dtype=np.uint8); lung[100:900, 200:400] = 1
    
    mock_dict = {"Lung_Right": lung, "Lung_Left": lung, "Clavicle_Right": None, "Clavicle_Left": None}
    proj, rel_h = determine_ap_pa_projection(mock_dict)
    
    assert proj == "Unknown"
    assert rel_h == 0.0

# ==========================================
# LEVEL 4: ORCHESTRATOR & EXTREME EDGE CASES
# ==========================================

def test_assess_projection_the_cat_scenario():
    """
    The 'Cat' image scenario: Anatomy is missing or total garbage.
    Should trigger the Anatomic Panic Button.
    """
    h, w = 1000, 1000
    mock_dict = {"Spine": None, "Sternum": None, "Lung_Right": None, "Lung_Left": None}
    
    result = assess_projection(mock_dict, (h, w))
    
    assert result["is_frontal"] is False
    assert result["status"] == ProjectionClass.ERROR
    assert "missing" in result["reasoning"].lower()

def test_assess_projection_frontal_incomplete():
    """Tests orchestrator correctly tagging a clipped frontal image."""
    h, w = 1000, 1000
    spine = np.zeros((h, w), dtype=np.uint8); spine[100:900, 490:510] = 1
    sternum = np.zeros((h, w), dtype=np.uint8); sternum[400:600, 490:510] = 1
    lung = np.zeros((h, w), dtype=np.uint8); lung[0:800, 200:400] = 1 # Clipped top
    clav = np.zeros((h, w), dtype=np.uint8); clav[200:220, 250:350] = 1
    
    mock_dict = {
        "Spine": spine, "Sternum": sternum, 
        "Lung_Right": lung, "Lung_Left": lung,
        "Clavicle_Right": clav, "Clavicle_Left": clav
    }
    
    result = assess_projection(mock_dict, (h, w))
    
    assert result["is_frontal"] is True
    assert result["status"] == "frontal_incomplete"
    assert "Apices" in result["metrics"]["clipping_list"][0]

# ==========================================
# LEVEL 5: REAL IMAGE INTEGRATION
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_projection_calibration_printout(image_path: Path, loaded_models):
    """Runs the projection module on real images and prints the calibration values."""
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
    
    result = assess_projection(qa_ready_dict, original_shape)
    
    print(f"\n--- Projection Metrics for {image_path.name} ---")
    if result["status"] == ProjectionClass.ERROR:
        print(f"  🚨 ERROR: {result['reasoning']}")
    else:
        m = result["metrics"]
        print(f"  Is Frontal: {m['is_frontal']}")
        print(f"  Predicted View: {m['projection']} (Clavicle-Apex Ratio: {m['clavicle_apex_ratio']:.2f})")
        print(f"  FOV Complete: {m['is_complete']}")
        if not m['is_complete']:
            print(f"  Clipping: {', '.join(m['clipping_list'])}")
        print(f"  AI Verdict: {result['status'].upper()} | {result['reasoning']}")