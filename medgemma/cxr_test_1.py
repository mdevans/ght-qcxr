import os, sys
import torch
import numpy as np
import pydicom
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- CONFIGURATION ---
MODEL_PATH = "medgemma/medgemma_vision"
TEST_FILE = "data/cxr01.dcm"
DEVICE = torch.device("mps")

# SigLIP expects inputs normalized to mean 0.5, std 0.5
# We will manually handle this to ensure precision
TARGET_SIZE = 896

def load_local_medgemma(path: str):
    print(f"Loading MedGemma 1.5 from local path: {path}...")
    
    try:
        # 1. Load the Processor
        # Explicitly set use_fast=True to silence the "breaking change" warning
        processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True)
        
        # 2. Load Model to CPU first (The Fix)
        # REMOVED: device_map={"": DEVICE} <-- This was causing the 8GB crash
        # UPDATED: torch_dtype -> dtype
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=torch.bfloat16,   
            local_files_only=True
        )
        
        # 3. Manually move to MPS
        # This streams the model to GPU memory safely without one giant allocation
        print(f"🔄 Moving model to {DEVICE} (Apple Silicon)...")
        
        model = model.to(DEVICE) # type: ignore
        
        print("✅ Model loaded successfully on MPS.")
        return model, processor
        
    except Exception as e:
        print(f"[bold red]❌ Error loading local model:[/bold red] {e}")
        sys.exit(1)

def get_high_fidelity_input(dicom_path: str) -> np.ndarray:
    """
    Reads DICOM, applies Windowing, and returns a Float32 array (0.0 - 1.0).
    Skips the 8-bit JPEG/PNG bottleneck entirely.
    """
    ds = pydicom.dcmread(dicom_path)
    
    # 1. Apply Clinical Windowing (High Bit-Depth Preservation)
    # We use pydicom's apply_voi_lut which handles the 12/16-bit math correctly
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    if 'WindowWidth' in ds and 'WindowCenter' in ds:
        print("   Applying VOI LUT for clinical windowing...")
        arr = apply_voi_lut(ds.pixel_array, ds)
    else:
        arr = ds.pixel_array.astype(float)

    # 2. Normalize to 0.0 - 1.0 (Float32)
    # This preserves the gradients between "3200" and "3201" as distinct float values
    arr = arr.astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr)

    # 3. Handle Inversion (Monochrome1 means White is 0)
    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        arr = 1.0 - arr

    # 4. Convert to RGB (Duplicate channels) - Result is (H, W, 3)
    arr = np.stack([arr] * 3, axis=-1)
    
    return arr

def pad_and_resize_float(img_array: np.ndarray, target_size=896) -> np.ndarray:
    """
    Resizes a Float32 array without squashing (Distortion check).
    Fills background with 0.5 (Grey) to match SigLIP distribution.
    """
    # img_array is (Height, Width, 3)
    h, w, c = img_array.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # We use PIL for the resizing interpolation, but we convert back to float immediately
    # Mode "F" (Float) in PIL is tricky with 3 channels, so we do per-channel
    resized_channels = []
    for i in range(3):
        # Convert single channel to PIL Image (Float mode)
        pil_ch = Image.fromarray(img_array[:, :, i])
        # Resize using Bicubic (High quality)
        pil_resized = pil_ch.resize((new_w, new_h), Image.Resampling.BICUBIC)
        resized_channels.append(np.array(pil_resized))
    
    # Stack back to (New_H, New_W, 3)
    resized_img = np.stack(resized_channels, axis=-1)
    
    # Create Square Canvas (Filled with 0.5 grey)
    canvas = np.full((target_size, target_size, 3), 0.5, dtype=np.float32)
    
    # Paste centered
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return canvas

def debug_view(img_array: np.ndarray, filename="debug_input.png"):
    """Saves the float array as a visible PNG to verify geometry."""
    # Convert 0.0-1.0 float back to 0-255 uint8 just for human viewing
    vis = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(vis).save(filename)
    print(f"🛑 Debug: Saved model input view to {filename}")

def analyze(model, processor, image_path, query):
    # 1. High Fidelity Load
    if image_path.endswith(".dcm"):
        float_img = get_high_fidelity_input(image_path)
    else:
        # Fallback for JPG
        pil_img = Image.open(image_path).convert("RGB")
        float_img = np.array(pil_img).astype(np.float32) / 255.0

    # 2. Distortion-Free Padding
    compliant_input = pad_and_resize_float(float_img, TARGET_SIZE)
    
    # 3. Debug Check
    debug_view(compliant_input)

    # 4. Processor Ingestion
    # We pass the numpy array directly. 
    # Since it is already 0.0-1.0 and we disabled do_rescale, processor just normalizes (x-0.5)/0.5
    inputs = processor(
        text=query, 
        images=compliant_input, 
        return_tensors="pt"
    ).to(DEVICE)
    
    # Cast to bfloat16 to match model weights
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # 5. Generate
    print("🩺 Running Inference...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    
    print(processor.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    model, processor = load_local_medgemma(MODEL_PATH)
    analyze(model, processor, TEST_FILE, "Analyze this chest X-ray.")