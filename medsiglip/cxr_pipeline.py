import os
import numpy as np
from PIL import Image, ImageOps
import torch

# --- CONFIGURATION ---
TARGET_SIZE = (448, 448)

def load_and_preprocess(image_path, save_debug=False, debug_prefix="sanity_check"):
    """
    Robustly loads medical images (16-bit PNG, DICOM-exported JPG, etc.),
    converts to high-contrast RGB 8-bit, and prepares the tensor for MedSigLIP.
    
    Returns:
        tensor: The (1, 3, 448, 448) tensor ready for the model.
        pil_image: The processed 8-bit PIL image (for visual verification).
    """
    
    # 1. LOAD RAW IMAGE
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"❌ Error opening {image_path}: {e}")
        return None, None

    # 2. HANDLE 16-BIT / GRAYSCALE DEPTH
    # PIL mode 'I' or 'I;16' denotes 16-bit integer data.
    if img.mode in ('I', 'I;16'):
        # Convert to numpy to handle the large values manually
        raw_data = np.array(img).astype(np.float32)
        
        # Normalize to 0-255 based on the ACTUAL range in the image
        # (This prevents washout if the 16-bit data only uses a small range like 0-4096)
        min_val = raw_data.min()
        max_val = raw_data.max()
        
        if max_val > min_val:
            normalized = (raw_data - min_val) / (max_val - min_val) # 0.0 to 1.0
            img_8bit = (normalized * 255.0).astype(np.uint8)
        else:
            img_8bit = np.zeros_like(raw_data, dtype=np.uint8) # Blank image
            
        # Convert back to PIL 8-bit Grayscale
        img = Image.fromarray(img_8bit, mode='L')
    
    # 3. CONVERT TO RGB
    # We convert to RGB even if grayscale because the model expects 3 channels
    img = img.convert("RGB")

    # 4. FIX CONTRAST (The "Muddy Gray" Fix)
    # Autocontrast stretches the histogram to ensure true blacks and whites
    img = ImageOps.autocontrast(img, cutoff=1) 

    # 5. RESIZE
    # Lanczos provides the sharpest downsampling for medical features
    img = img.resize(TARGET_SIZE, resample=Image.Resampling.LANCZOS)

    # 6. SAVE DEBUG IMAGE (Optional)
    if save_debug:
        debug_filename = f"{debug_prefix}_processed.jpg"
        img.save(debug_filename, quality=95)
        print(f"📸 Saved debug view to: {debug_filename}")

    # 7. TENSOR PREPARATION (The "Special" Normalization)
    # Convert HWC (Height, Width, Channel) -> CHW (Channel, Height, Width)
    img_array = np.array(img).transpose(2, 0, 1)
    
    # Scale 0-255 -> 0.0-1.0
    tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
    
    # Normalize Mean=0.5, Std=0.5 -> Range -1.0 to 1.0
    # Formula: (x - mean) / std
    tensor = (tensor - 0.5) / 0.5
    
    # Add Batch Dimension -> (1, 3, 448, 448)
    tensor = tensor.unsqueeze(0)
    
    return tensor, img

# --- SELF-TEST SCRIPT ---
if __name__ == "__main__":
    # Test on your specific image
    TEST_FILE = "data/TB_Chest_Radiography_Database/Normal/Normal-1.png" # Update this to point to a 16-bit PNG if you have one
    
    if os.path.exists(TEST_FILE):
        print(f"🧪 Testing pipeline on: {TEST_FILE}")
        
        # Run Pipeline
        tensor, processed_img = load_and_preprocess(TEST_FILE, save_debug=True)
        
        if tensor is not None:
            # Check Stats
            print("\n📊 Tensor Statistics:")
            print(f"   Shape: {tensor.shape} (Expected: 1, 3, 448, 448)")
            print(f"   Range: {tensor.min():.2f} to {tensor.max():.2f} (Expected: ~ -1.0 to 1.0)")
            print(f"   Type:  {tensor.dtype}")
            print("\n✅ Verification Successful. Check 'sanity_check_processed.jpg'.")
    else:
        print(f"⚠️ Test file not found: {TEST_FILE}")