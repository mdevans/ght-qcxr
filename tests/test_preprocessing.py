import pytest
import torch
import numpy as np
import cv2
from PIL import Image
import pydicom
from pathlib import Path
from pydicom.data import get_testdata_file

# 1. Centralized Test Utils
from tests.test_utils import TEST_IMAGES

# 2. Pipeline Imports
from preprocessing import (
    preprocess_image,
    KEY_CXAS_TENSOR,
    KEY_XRV_TENSOR,
    KEY_BASE_IMAGE,
    KEY_METADATA,
    EXTRACTED_DICOM_TAGS, 
)

# ==========================================
# LEVEL 1: SYNTHETIC ATOMIC TESTS (Edge Cases)
# ==========================================

def test_jpg_wide_squash(tmp_path: Path):
    """Tests that a wide image is strictly squashed to 512x512, with no padding."""
    img_8bit = np.random.randint(100, 256, (256, 800), dtype=np.uint8)
    image_path = tmp_path / "test_wide.jpg"
    Image.fromarray(img_8bit).save(image_path)
    
    result = preprocess_image(image_path)
    
    # 1. Base Image should retain exact original dimensions for drawing overlays
    assert result[KEY_BASE_IMAGE].shape == (256, 800)
    
    # 2. Tensors should be aggressively squashed to exact 512x512
    assert result[KEY_XRV_TENSOR].shape == (1, 1, 512, 512)
    assert result[KEY_CXAS_TENSOR].shape == (1, 3, 512, 512)

def test_jpg_tall_squash(tmp_path: Path):
    """Tests that a tall image is strictly squashed to 512x512, with no padding."""
    img_8bit = np.random.randint(100, 256, (800, 256), dtype=np.uint8)
    image_path = tmp_path / "test_tall.jpg"
    Image.fromarray(img_8bit).save(image_path)
    
    result = preprocess_image(image_path)
    
    assert result[KEY_BASE_IMAGE].shape == (800, 256)
    assert result[KEY_XRV_TENSOR].shape == (1, 1, 512, 512)

def test_exact_target_size(tmp_path: Path):
    """Tests that an image exactly matching TARGET_SIZE is not altered geometrically."""
    img_8bit = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    image_path = tmp_path / "test_exact.png"
    Image.fromarray(img_8bit).save(image_path)
    
    result = preprocess_image(image_path)
    
    assert result[KEY_BASE_IMAGE].shape == (512, 512)
    assert result[KEY_CXAS_TENSOR].shape == (1, 3, 512, 512)
    assert result[KEY_XRV_TENSOR].shape == (1, 1, 512, 512)

def test_rgb_png_to_grayscale(tmp_path: Path):
    """Tests that a 3-channel RGB PNG is safely converted to a 2D single channel base image."""
    img_rgb = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    image_path = tmp_path / "test_rgb.png"
    Image.fromarray(img_rgb, mode="RGB").save(image_path)

    result = preprocess_image(image_path)
    
    # The base image must be strictly 2D (H, W), shedding the 3 color channels
    assert result[KEY_BASE_IMAGE].shape == (500, 500)
    assert len(result[KEY_BASE_IMAGE].shape) == 2 
    assert result[KEY_CXAS_TENSOR].shape == (1, 3, 512, 512)

def test_png_16bit_processing(tmp_path: Path):
    """Tests preprocessing of a high bit-depth 16-bit PNG file."""
    img_16bit = np.random.randint(0, 4096, (700, 900), dtype=np.uint16)
    image_path = tmp_path / "test_16bit.png"
    
    # Save natively as 16-bit using OpenCV
    cv2.imwrite(str(image_path), img_16bit)

    result = preprocess_image(image_path)

    # Base image must be safely normalized down to 8-bit unsigned integer
    assert result[KEY_BASE_IMAGE].shape == (700, 900)
    assert result[KEY_BASE_IMAGE].dtype == np.uint8
    
    assert result[KEY_CXAS_TENSOR].dtype == torch.float32
    assert result[KEY_XRV_TENSOR].dtype == torch.float32

def test_dicom_metadata_completeness_and_multivalue(tmp_path: Path):
    """Tests that all required tags exist in the output dict, and MultiValues are joined."""
    source_dicom_path = get_testdata_file("CT_small.dcm") 
    image_path = tmp_path / "test_meta.dcm"
    
    ds = pydicom.dcmread(str(source_dicom_path))
    ds.PixelSpacing = [0.25, 0.25] 
    ds.save_as(str(image_path))

    result = preprocess_image(image_path)
    meta = result[KEY_METADATA]

    for tag in EXTRACTED_DICOM_TAGS:
        assert tag in meta, f"Missing required metadata key: {tag}"

    assert meta["PixelSpacing"] == "0.25, 0.25"
    assert meta["BodyPartExamined"] == "Not Found"

def test_dicom_monochrome1_inversion(tmp_path: Path):
    """Tests that MONOCHROME1 files are mathematically inverted."""
    source_dicom_path = get_testdata_file("CT_small.dcm") 
    image_path = tmp_path / "test_mono1.dcm"
    
    ds = pydicom.dcmread(str(source_dicom_path))
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.save_as(str(image_path))

    result_mono1 = preprocess_image(image_path)
    
    original_path = tmp_path / "test_mono2.dcm"
    pydicom.dcmread(str(source_dicom_path)).save_as(str(original_path))
    result_mono2 = preprocess_image(original_path)

    tensor_mono1 = result_mono1[KEY_XRV_TENSOR]
    tensor_mono2 = result_mono2[KEY_XRV_TENSOR]

    assert not torch.allclose(tensor_mono1, tensor_mono2)

def test_invalid_image_path(tmp_path: Path):
    """Tests that passing a non-existent file raises the proper ValueError."""
    fake_path = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="Failed to load image"):
        preprocess_image(fake_path)

def test_corrupt_dicom_file(tmp_path: Path):
    """Tests fallback behavior if a file claims to be a DICOM but is garbage data."""
    bad_dicom_path = tmp_path / "corrupt.dcm"
    with open(bad_dicom_path, "wb") as f:
        f.write(b"NOT A REAL DICOM FILE 1234567890")
    
    with pytest.raises(ValueError, match="Failed to load image"):
        preprocess_image(bad_dicom_path)

# ==========================================
# LEVEL 2: REAL DATA INTEGRATION
# ==========================================

@pytest.mark.integration
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=[p.name for p in TEST_IMAGES])
def test_real_images_preprocessing_success(image_path: Path):
    """
    Ensures the preprocessing pipeline successfully handles all real test clinical 
    files (DICOM, PNG, JPG), returning mathematically normalized and verified arrays.
    """
    result = preprocess_image(image_path)
    
    # 1. Verify strict data contract keys exist
    assert KEY_BASE_IMAGE in result
    assert KEY_CXAS_TENSOR in result
    assert KEY_XRV_TENSOR in result
    assert KEY_METADATA in result
    
    # 2. Verify Base Image Integrity & Content
    base_img = result[KEY_BASE_IMAGE]
    assert isinstance(base_img, np.ndarray), "Base image is not a numpy array"
    assert base_img.dtype == np.uint8, f"Base image is {base_img.dtype}, expected uint8"
    assert len(base_img.shape) == 2, f"Base image has shape {base_img.shape}, expected 2D"
    
    # Deep Numerical Checks: Image must be a realistic size and not solid black
    h, w = base_img.shape
    assert h > 100 and w > 100, f"Image dimensions ({w}x{h}) are impossibly small for a CXR."
    assert base_img.max() > 0, "Base image is completely black (all zeros). Loading failed silently."

    # 3. Verify AI-ready shapes
    cxas_tensor = result[KEY_CXAS_TENSOR]
    xrv_tensor = result[KEY_XRV_TENSOR]
    assert cxas_tensor.shape == (1, 3, 512, 512), "CXAS tensor shape mismatch."
    assert xrv_tensor.shape == (1, 1, 512, 512), "XRV tensor shape mismatch."
    
    # 4. Verify PyTorch types and strict Mathematical Normalization
    assert cxas_tensor.dtype == torch.float32
    assert xrv_tensor.dtype == torch.float32
    
    # XRV uses [-1024, 1024] scaling. It must therefore contain negative and positive values.
    assert xrv_tensor.min().item() < 0.0, "XRV tensor not properly normalized (min >= 0)"
    assert xrv_tensor.max().item() > 0.0, "XRV tensor not properly normalized (max <= 0)"
    
    # CXAS uses ImageNet standardization. It must cross zero.
    assert cxas_tensor.min().item() < 0.0, "CXAS tensor not ImageNet normalized (min >= 0)"
    assert cxas_tensor.max().item() > 0.0, "CXAS tensor not ImageNet normalized (max <= 0)"

    # 5. Verify Metadata Logic Check
    metadata = result[KEY_METADATA]
    assert isinstance(metadata, dict), "Metadata must be a dictionary"
    
    if image_path.suffix.lower() == ".dcm":
        assert len(metadata) > 0, "DICOM file failed to extract any metadata tags."
        assert "PatientID" in metadata, "DICOM is missing foundational PatientID tag."
    else:
        assert len(metadata) == 0, f"Non-DICOM file ({image_path.suffix}) should return an empty metadata dictionary."