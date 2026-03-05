import pytest
import torch
import numpy as np
import cv2
from PIL import Image
import pydicom
from pathlib import Path
from pydicom.data import get_testdata_file

# Assume the main script is named preprocessing.py
from preprocessing import (
    preprocess_image,
    KEY_CXAS_TENSOR,
    KEY_XRV_TENSOR,
    KEY_METADATA,
    KEY_ORIGINAL_SHAPE,
    EXTRACTED_DICOM_TAGS, # Make sure to import this!
)

def test_jpg_wide_padding(tmp_path: Path):
    """Tests preprocessing of a wide image (top/bottom padding)."""
    # Use random bright noise to prevent OpenCV NORM_MINMAX from flattening a zero-variance image
    img_8bit = np.random.randint(100, 256, (256, 800), dtype=np.uint8)
    
    # Anchor the min/max values so normalization and CLAHE behave consistently
    img_8bit[0, 0] = 0 
    img_8bit[0, 1] = 255 
    
    image_path = tmp_path / "test_wide.jpg"
    Image.fromarray(img_8bit).save(image_path)
    
    result = preprocess_image(image_path)
    xrv_tensor = result[KEY_XRV_TENSOR]

    # Top row should be padding (black/minimum value), center should be image data
    assert xrv_tensor[0, 0, 0, 256].item() < xrv_tensor[0, 0, 256, 256].item()
    assert result[KEY_ORIGINAL_SHAPE] == (256, 800)

def test_jpg_tall_padding(tmp_path: Path):
    """Tests preprocessing of a tall image (left/right padding)."""
    # Use random bright noise to prevent zero-variance flattening
    img_8bit = np.random.randint(100, 256, (800, 256), dtype=np.uint8)
    
    # Anchor the min/max values
    img_8bit[0, 0] = 0
    img_8bit[0, 1] = 255
    
    image_path = tmp_path / "test_tall.jpg"
    Image.fromarray(img_8bit).save(image_path)
    
    result = preprocess_image(image_path)
    xrv_tensor = result[KEY_XRV_TENSOR]

    # Leftmost column should be padding, center should be image data
    assert xrv_tensor[0, 0, 256, 0].item() < xrv_tensor[0, 0, 256, 256].item()
    assert result[KEY_ORIGINAL_SHAPE] == (800, 256)

def test_exact_target_size(tmp_path: Path):
    """Tests that an image exactly matching TARGET_SIZE is not altered geometrically."""
    img_8bit = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    image_path = tmp_path / "test_exact.png"
    Image.fromarray(img_8bit).save(image_path)
    
    result = preprocess_image(image_path)
    
    assert result[KEY_ORIGINAL_SHAPE] == (512, 512)
    assert result[KEY_CXAS_TENSOR].shape == (1, 3, 512, 512)
    assert result[KEY_XRV_TENSOR].shape == (1, 1, 512, 512)

def test_rgb_png_to_grayscale(tmp_path: Path):
    """Tests that a 3-channel RGB PNG is safely squashed to single channel."""
    img_rgb = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    image_path = tmp_path / "test_rgb.png"
    Image.fromarray(img_rgb, mode="RGB").save(image_path)

    result = preprocess_image(image_path)
    
    assert result[KEY_ORIGINAL_SHAPE][:2] == (500, 500)
    assert result[KEY_CXAS_TENSOR].shape == (1, 3, 512, 512)

def test_png_16bit_processing(tmp_path: Path):
    """Tests preprocessing of a high bit-depth 16-bit PNG file."""
    img_16bit = np.random.randint(0, 4096, (700, 900), dtype=np.uint16)
    image_path = tmp_path / "test_16bit.png"
    Image.fromarray(img_16bit).save(image_path)

    result = preprocess_image(image_path)

    assert result[KEY_ORIGINAL_SHAPE] == (700, 900)
    assert result[KEY_CXAS_TENSOR].dtype == torch.float32
    assert result[KEY_XRV_TENSOR].dtype == torch.float32

def test_dicom_metadata_completeness_and_multivalue(tmp_path: Path):
    """Tests that all required tags exist in the output dict, and MultiValues are joined."""
    source_dicom_path = get_testdata_file("CT_small.dcm") 
    image_path = tmp_path / "test_meta.dcm"
    
    ds = pydicom.dcmread(str(source_dicom_path))
    # Inject a MultiValue list to test the join logic
    ds.PixelSpacing = [0.25, 0.25] 
    ds.save_as(str(image_path))

    result = preprocess_image(image_path)
    meta = result[KEY_METADATA]

    # 1. Prove every tag defined in the constants exists in the final dict
    for tag in EXTRACTED_DICOM_TAGS:
        assert tag in meta, f"Missing required metadata key: {tag}"

    # 2. Prove MultiValue parsing joined the list into a string
    assert meta["PixelSpacing"] == "0.25, 0.25"
    
    # 3. Prove missing tags gracefully fallback
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

    # Verify they are drastically different (inversion)
    assert not torch.allclose(tensor_mono1, tensor_mono2)

def test_invalid_image_path(tmp_path: Path):
    """Tests that passing a non-existent file raises the proper ValueError."""
    fake_path = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="Failed to load image"):
        preprocess_image(fake_path)

def test_corrupt_dicom_file(tmp_path: Path):
    """Tests fallback behavior if a file claims to be a DICOM but is garbage data."""
    bad_dicom_path = tmp_path / "corrupt.dcm"
    # Write garbage bytes to simulate a corrupt file
    with open(bad_dicom_path, "wb") as f:
        f.write(b"NOT A REAL DICOM FILE 1234567890")
    
    # pydicom's is_dicom will reject it, so it falls back to PIL/OpenCV, 
    # which will then fail to read it as an image and raise ValueError.
    with pytest.raises(ValueError, match="Failed to load image"):
        preprocess_image(bad_dicom_path)