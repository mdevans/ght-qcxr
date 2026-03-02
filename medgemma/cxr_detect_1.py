import os
import sys
import argparse
import re
import numpy as np
import pydicom
from PIL import Image, ImageOps
import time
from datetime import datetime

# MLX Imports
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

import warnings
import logging
warnings.simplefilter("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_PATH = "./medgemma_vision"

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] 🛠️  {message}", file=sys.stderr)

def normalize_pixels(arr):
    """Normalizes any pixel array to 0-255 uint8 range."""
    arr = arr.astype(float)
    if arr.max() != arr.min():
        arr = (np.maximum(arr, 0) / arr.max()) * 255.0
    else:
        arr = arr * 0 
    return arr.astype(np.uint8)

def load_pixels_robust(image_path):
    log(f"Loading: {image_path}")
    try:
        # --- DICOM HANDLING ---
        if image_path.lower().endswith(('.dcm', '.dicom')):
            ds = pydicom.dcmread(image_path)
            arr = ds.pixel_array
            
            # 1. Photometric Inversion Check
            # MONOCHROME1 means 0=White, Max=Black (Inverted appearance).
            # We want 0=Black, Max=White (Standard X-ray).
            photometric = ds.get("PhotometricInterpretation", "UNKNOWN")
            log(f"DICOM Photometric: {photometric}")
            
            if photometric == "MONOCHROME1":
                log("Applying Inversion (MONOCHROME1 detected)...")
                # Invert values based on bit depth usually, but simple max inversion works safest
                arr = np.max(arr) - arr
            
            # 2. Normalize to 0-255
            arr = normalize_pixels(arr)
            return Image.fromarray(arr).convert("RGB")

        # --- PNG / STANDARD IMAGE HANDLING ---
        else:
            img = Image.open(image_path).convert("RGB")
            # User specified PNGs will always have black background (standard).
            # No inversion needed.
            return img

    except Exception as e:
        log(f"❌ Error loading image: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    # 1. Model Check
    if not os.path.exists(MODEL_PATH):
        print("False") # Fail safe for pipeline
        log(f"CRITICAL: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    log("Initializing Model...")
    # Load model
    model, processor = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True, "use_fast": True})
    config = load_config(MODEL_PATH)

    # 2. Image Load
    image = load_pixels_robust(args.image_path)
    if image is None:
        print("False") 
        sys.exit(0)

    # 3. ROBUST PROMPT (Anatomy Bucket Strategy)
    # We ask the model to pick from a list. This prevents "Yes/No" hallucination loops.
    query = """
    Identify the primary body part in this medical image.
    
    Choose strictly ONE from this list:
    - CHEST
    - KNEE
    - WRIST
    - HAND
    - SKULL
    - PELVIS
    - ABDOMEN
    - LUMBAR SPINE
    - THORACIC SPINE
    
    Answer with just the word.
    """
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    prompt = apply_chat_template(processor, config, messages)
    
    log("Generating...")
    start = time.time()
    
    response = generate(
        model, processor, prompt, image, 
        verbose=False, max_tokens=10, temp=0.0
    )
    
    # 4. Parsing
    answer = response.text.strip().upper().replace(".", "")
    log(f"Time: {time.time() - start:.2f}s")
    log(f"AI Answer: '{answer}'")

    # 5. Exact Match Logic
    # We look for the standalone word "CHEST"
    if re.search(r"\bCHEST\b", answer):
        print("True")
    else:
        log(f"Rejected: Model saw '{answer}'")
        print("False")

if __name__ == "__main__":
    main()