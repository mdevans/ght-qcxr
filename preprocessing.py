import torch
import numpy as np
import cv2
import pydicom
from pydicom.misc import is_dicom
from pydicom.pixels.processing import apply_voi_lut, apply_modality_lut
from pydicom.multival import MultiValue
from pathlib import Path
import torchxrayvision as xrv

# --- Preprocessing Constants ---
TARGET_SIZE = 512
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- Output Dictionary Keys ---
KEY_XRV_TENSOR = "xrv_Tensor"
KEY_CXAS_TENSOR = "cxas_Tensor"
KEY_METADATA = "metadata"
KEY_ORIGINAL_SHAPE = "original_shape"

# --- DICOM Metadata Extraction ---
EXTRACTED_DICOM_TAGS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "ViewPosition",             
    "BodyPartExamined",
    "SeriesDescription",
    "Rows",
    "Columns",
    "PixelSpacing",
    "BitsStored",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
]

def _load_dicom_image(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Loads a DICOM file, applies LUTs, and returns an 8-bit, Single Channel NumPy array and metadata."""
    ds = pydicom.dcmread(path)
    
    # Apply Modality LUT first, then VOI LUT for correct windowing. This is equivalent to CLAHE on jpg/png.
    pixel_array = apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds)

    # Normalize to 8-bit using OpenCV's normalization, which handles outliers gracefully.
    # Pre-allocate an empty 8-bit array
    pixels = np.empty(pixel_array.shape, dtype=np.uint8)
    cv2.normalize(src=pixel_array, dst=pixels, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Handle MONOCHROME1 (inverted) images
    if ds.PhotometricInterpretation == "MONOCHROME1":
        pixels = cv2.bitwise_not(pixels)

    # Extract only the DICOM fields essential for QA processing into a flat dictionary.
    metadata = {}
    for keyword in EXTRACTED_DICOM_TAGS:
        # getattr fetches the underlying value directly (string, int, float, or MultiValue)
        val = getattr(ds, keyword, None)
        
        if val is not None:
            # If the value is a list-like MultiValue (e.g., PixelSpacing), join it into a string
            if isinstance(val, (MultiValue, list)):
                metadata[keyword] = ", ".join(map(str, val))
            else:
                metadata[keyword] = str(val)
        else:
            metadata[keyword] = "Not Found"

    return pixels, metadata

def _load_pil_image(path: Path) -> tuple[np.ndarray, dict]:
    """Loads a standard image file (PNG/JPG), applies CLAHE and returns an 8-bit, Single Channel NumPy array."""
    # 1. Load natively (Preserves 16-bit as uint16, forces single channel)
    pixels = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    
    if pixels is None:
        raise ValueError(f"Failed to load image at {path}")
    
    # 2. Apply CLAHE on the HIGH BIT-DEPTH data first
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    pixel_array = clahe.apply(pixels)

    # 3. Normalize the enhanced image down to 8-bit safely
    pixels = np.empty(pixel_array.shape, dtype=np.uint8)
    cv2.normalize(src=pixel_array, dst=pixels, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return pixels, {}

def _resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resizes an image while preserving its aspect ratio by adding black padding."""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas and place the resized image in the center
    padded = np.zeros((target_size, target_size), dtype=image.dtype)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized
    
    return padded

def _create_xrv_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Creates a preprocessed tensor for the TorchXRayVision model from a base image.
    The input `image` is expected to be an 8-bit, Single Channel NumPy array.

    NOTE: This pipeline is specific to XRV models. It requires a single-channel
    image normalized to the range [-1024, 1024].
    """
    padded_image = _resize_with_padding(image, target_size=TARGET_SIZE)
    normalized = xrv.datasets.normalize(padded_image, 255)
    
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    return tensor.unsqueeze(0) # Add batch dimension

def _create_cxas_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Creates a preprocessed tensor for the CXAS model from a base image.
    The input `image` is expected to be an 8-bit, Single Channel NumPy array.

    NOTE: This pipeline is specific to models with an ImageNet-trained backbone (like ResNet).
    It requires a 3-channel image and normalization with ImageNet's mean and std.
    """
    padded_image = _resize_with_padding(image, target_size=TARGET_SIZE)
    
    # Create a 3-channel image by stacking the grayscale channel, without losing float precision
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_GRAY2RGB)
    
    # Transpose from (H, W, C) to (C, H, W) and convert to tensor
    tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1)).float() / 255.0
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized_tensor = (tensor - mean) / std
    
    return normalized_tensor.unsqueeze(0) # Add batch dimension


# --- Main Orchestrator Function ---
def preprocess_image(image_path: Path) -> dict:
    """
    Main preprocessing function that orchestrates the loading and transformation pipelines.

    Args:
        image_path: Path to the input image (DICOM, PNG, or JPG).

    Returns:
        A dictionary containing the preprocessed tensors for each segmentation model and metadata.
    """
    # Load the image:
    pixels, metadata = _load_dicom_image(image_path) if is_dicom(image_path) else _load_pil_image(image_path)

    return {
        KEY_XRV_TENSOR: _create_xrv_tensor(pixels),
        KEY_CXAS_TENSOR: _create_cxas_tensor(pixels),
        KEY_METADATA: metadata,
        KEY_ORIGINAL_SHAPE: pixels.shape
    }