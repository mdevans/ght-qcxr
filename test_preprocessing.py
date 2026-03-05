import pytest
import torch
import numpy as np
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
)

def test_jpg_processing(tmp_path: Path):
    """Tests preprocessing of a standard 8-bit JPG file."""
    # Create a dummy 8-bit image
    img_8bit = np.random.randint(0, 256, (600, 800), dtype=np.uint8)
    image_path = tmp_path / "test_8bit.jpg"
    Image.fromarray(img_8bit).save(image_path)
    
    result = preprocess_image(image_path)

    assert KEY_CXAS_TENSOR in result
    assert KEY_XRV_TENSOR in result
    assert KEY_METADATA in result
    assert KEY_ORIGINAL_SHAPE in result

    # Check Original Shape retention
    assert result[KEY_ORIGINAL_SHAPE] == (600, 800)

    # Check CXAS tensor properties
    cxas_tensor = result[KEY_CXAS_TENSOR]
    assert isinstance(cxas_tensor, torch.Tensor)
    assert cxas_tensor.shape == (1, 3, 512, 512)
    assert cxas_tensor.dtype == torch.float32

    # Check XRV tensor properties
    xrv_tensor = result[KEY_XRV_TENSOR]
    assert isinstance(xrv_tensor, torch.Tensor)
    assert xrv_tensor.shape == (1, 1, 512, 512)
    assert xrv_tensor.dtype == torch.float32
    
    # Verify XRV centering occurred (should contain negative values)
    assert xrv_tensor.min() < 0

def test_png_16bit_processing(tmp_path: Path):
    """Tests preprocessing of a high bit-depth 16-bit PNG file."""
    # Create a dummy 16-bit image
    img_16bit = np.random.randint(0, 4096, (700, 900), dtype=np.uint16)
    image_path = tmp_path / "test_16bit.png"
    Image.fromarray(img_16bit).save(image_path)

    result = preprocess_image(image_path)

    assert result[KEY_CXAS_TENSOR].shape == (1, 3, 512, 512)
    assert result[KEY_XRV_TENSOR].shape == (1, 1, 512, 512)
    
    # Verify ImageNet normalization happened on CXAS (should have negative values)
    assert result[KEY_CXAS_TENSOR].min() < 0

def test_dicom_standard_processing(tmp_path: Path):
    """Tests preprocessing of a standard DICOM file."""
    source_dicom_path = get_testdata_file("CT_small.dcm") 
    
    if not source_dicom_path:
        pytest.fail("pydicom test data file 'CT_small.dcm' not found.")
    
    image_path = tmp_path / "test_standard.dcm"
    
    # Convert path to string for dcmread
    ds = pydicom.dcmread(str(source_dicom_path))
    ds.save_as(str(image_path))

    result = preprocess_image(image_path)

    assert KEY_CXAS_TENSOR in result
    assert KEY_XRV_TENSOR in result
    assert KEY_METADATA in result

    meta = result[KEY_METADATA]
    assert meta["PatientID"] != "Not Found"

    assert result[KEY_CXAS_TENSOR].shape == (1, 3, 512, 512)
    assert result[KEY_XRV_TENSOR].shape == (1, 1, 512, 512)

def test_dicom_monochrome1_inversion(tmp_path: Path):
    """Tests that MONOCHROME1 files are correctly inverted."""
    source_dicom_path = get_testdata_file("CT_small.dcm") 
    
    if not source_dicom_path:
        pytest.fail("pydicom test data file 'CT_small.dcm' not found.")

    image_path = tmp_path / "test_mono1.dcm"
    
    # Convert path to string for dcmread
    ds = pydicom.dcmread(str(source_dicom_path))
    
    # Force it to be MONOCHROME1 to trigger our inversion logic
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.save_as(str(image_path))

    result = preprocess_image(image_path)
    
    # Process the original file for comparison
    original_path = tmp_path / "test_mono2.dcm"
    pydicom.dcmread(str(source_dicom_path)).save_as(str(original_path))
    original_result = preprocess_image(original_path)

    mono1_tensor = result[KEY_XRV_TENSOR]
    mono2_tensor = original_result[KEY_XRV_TENSOR]

    # Verify they are NOT identical, proving the bitwise_not inversion logic ran
    assert not torch.allclose(mono1_tensor, mono2_tensor)