import pytest
import numpy as np
from pathlib import Path

# 1. Centralized Test Utils
from tests.test_utils import TEST_IMAGES

# 2. AI Pipeline Imports
from segment import predict_cxas, predict_xrv, resize_mask_back_to_orig, DEVICE
from preprocessing import KEY_BASE_IMAGE, preprocess_image, KEY_CXAS_TENSOR, KEY_XRV_TENSOR
from ensemble import blend_patient_masks

# 3. QA Module Imports
from exposure import (
    cxas_thoracic_vertebrae_segmented,
    heart_spine_penetration_variance,
    lung_field_burnout,
    is_over_exposed,
    is_under_exposed,
    assess_exposure,
    ExposureClass
)

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(scope="module")
def sample_clinical_data(loaded_models):
    """
    Runs the strict pipeline on ONE real image to generate a realistic payload 
    for our intermediate atomic tests, saving massive amounts of GPU time.
    """
    if not TEST_IMAGES:
        pytest.skip("No test images found to generate clinical data.")
        
    image_path = TEST_IMAGES[0]
    cxas_model, xrv_model = loaded_models
    
    # 1. Preprocess using the official single-source pipeline
    data = preprocess_image(image_path)
    base_img = data[KEY_BASE_IMAGE]
    original_shape = base_img.shape # Safely pulls (H, W) from the strict GrayscaleImage
    
    # 2. AI Inference
    cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
    xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
    
    # 3. Assemble (No external anatomical filters, exposure cleans its own data!)
    blended_raw = blend_patient_masks(cxas_raw, xrv_raw)
    
    # 4. Geometry Alignment
    qa_ready_masks = {
        k: resize_mask_back_to_orig(v, original_shape) 
        for k, v in blended_raw.items() if v is not None
    }
    
    return qa_ready_masks, base_img

# ==========================================
# LEVEL 1: SYNTHETIC ATOMIC TESTS (Edge Cases)
# ==========================================

def test_cxas_thoracic_vertebrae_segmented_logic():
    """Proves the counter strictly counts T1-T12 and ignores empty arrays."""
    # Added dtype=np.uint8 because cv2.findContours strictly requires 8-bit arrays
    segmented_masks = {
        "T1": np.ones((5, 5), dtype=np.uint8), 
        "T12": np.ones((5, 5), dtype=np.uint8), 
        "T13": np.ones((5, 5), dtype=np.uint8), 
        "Spine": np.ones((5, 5), dtype=np.uint8) 
    }
    assert cxas_thoracic_vertebrae_segmented(segmented_masks) == 2

def test_heart_spine_penetration_variance_math():
    """Proves the variance is calculated STRICTLY where the heart and spine overlap."""
    heart = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    spine = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    segmented_masks = {"Heart": heart, "Spine": spine}
    
    gray_img = np.array([[100, 50], [200, 250]], dtype=np.uint8)
    assert heart_spine_penetration_variance(segmented_masks, gray_img) == 0.0
    
    spine_adjusted = np.array([[1, 1], [0, 0]], dtype=np.uint8) 
    segmented_masks["Spine"] = spine_adjusted
    assert heart_spine_penetration_variance(segmented_masks, gray_img) == 625.0

def test_lung_field_burnout_math():
    """Proves burnout ratio accurately counts pixels below the blackness threshold."""
    segmented_masks = {"Lung_Right": np.array([[1, 1], [1, 1]], dtype=np.uint8)}
    gray_img = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    assert lung_field_burnout(segmented_masks, gray_img, black_threshold=15) == 0.25

# ==========================================
# LEVEL 2: REAL DATA ATOMIC TESTS
# ==========================================

def test_real_vertebrae_extraction(sample_clinical_data):
    """Tests that the vertebrae helper functions correctly on massive real arrays."""
    segmented_masks, _ = sample_clinical_data
    count = cxas_thoracic_vertebrae_segmented(segmented_masks)
    
    assert isinstance(count, int)
    assert 0 <= count <= 12, f"Vertebrae count {count} is completely out of biological bounds."

def test_real_penetration_variance(sample_clinical_data):
    """Tests the Heart-Spine intersection math on real pixel geometries."""
    segmented_masks, gray_img = sample_clinical_data
    variance = heart_spine_penetration_variance(segmented_masks, gray_img)
    
    assert isinstance(variance, float)
    assert variance >= 0.0, "Variance cannot be mathematically negative."

def test_real_lung_burnout(sample_clinical_data):
    """Tests the Lung blackout ratio on real pixel geometries."""
    segmented_masks, gray_img = sample_clinical_data
    ratio = lung_field_burnout(segmented_masks, gray_img)
    
    assert isinstance(ratio, float)
    assert 0.0 <= ratio <= 1.0, "Burnout ratio must be a clean percentage between 0 and 1."

def test_decision_logic_boundaries():
    """Proves the boolean decision functions trigger exactly when expected."""
    assert is_over_exposed(burnout_ratio=0.35, burnout_threshold=0.30) is True
    assert is_over_exposed(burnout_ratio=0.10, burnout_threshold=0.30) is False
    
    assert is_under_exposed(variance=30.0, vertebrae_count=10, min_variance=50.0) is True
    assert is_under_exposed(variance=100.0, vertebrae_count=5, min_vertebrae=8) is True
    assert is_under_exposed(variance=100.0, vertebrae_count=10) is False

# ==========================================
# LEVEL 3: FULL ORCHESTRATOR INTEGRATION
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_exposure_calibration_printout(image_path: Path, loaded_models):
    """
    Runs real test images through the strict linear pipeline to print metrics.
    Run with `pytest -s` to view output for threshold tuning.
    """
    cxas_model, xrv_model = loaded_models
    
    # 1. Preprocess
    data = preprocess_image(image_path)
    base_img = data[KEY_BASE_IMAGE]
    original_shape = base_img.shape
    
    # 2. AI Inference
    cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
    xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
    
    # 3. Assemble
    blended_raw = blend_patient_masks(cxas_raw, xrv_raw)
    
    # 4. Geometry Alignment
    qa_ready_masks = {
        k: resize_mask_back_to_orig(v, original_shape) 
        for k, v in blended_raw.items() if v is not None
    }
    
    # 5. Assessment
    result = assess_exposure(qa_ready_masks, base_img)
    metrics = result["metrics"]
    
    # 6. Calibration Output
    print(f"\n--- Exposure Metrics for {image_path.name} ---")
    print(f"  Vertebrae Count: {metrics['vertebrae_count']}/12")
    print(f"  Heart/Spine Variance: {metrics['heart_spine_variance']:.2f}")
    print(f"  Lung Burnout Ratio: {metrics['lung_burnout_ratio']:.2%}")
    print(f"  AI Verdict: {result['status']} | {result['reasoning']}")