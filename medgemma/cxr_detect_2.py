import os
import sys
import argparse
import numpy as np
import pydicom
from PIL import Image
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
    arr = arr.astype(float)
    if arr.max() != arr.min():
        arr = (np.maximum(arr, 0) / arr.max()) * 255.0
    else:
        arr = arr * 0 
    return arr.astype(np.uint8)



def resize_with_padding(image, target_size=(384, 384)):
    """
    Resizes image to target_size while maintaining aspect ratio via padding.
    Prevents 'squishing' distortion.
    """
    # 1. Resize the image so the longest side matches the target
    image.thumbnail(target_size, Image.Resampling.BICUBIC)
    
    # 2. Create a black background (standard for X-rays)
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    
    # 3. Paste the resized image into the center
    left = (target_size[0] - image.width) // 2
    top = (target_size[1] - image.height) // 2
    new_image.paste(image, (left, top))
    
    return new_image

def load_pixels_robust(image_path):
    try:
        # --- DICOM ---
        if image_path.lower().endswith(('.dcm', '.dicom')):
            ds = pydicom.dcmread(image_path)
            arr = ds.pixel_array
            
            # Photometric Inversion check
            photometric = ds.get("PhotometricInterpretation", "UNKNOWN")
            if photometric == "MONOCHROME1":
                arr = np.max(arr) - arr
            
            arr = normalize_pixels(arr)
            return Image.fromarray(arr).convert("RGB")

        # --- STANDARD IMAGES ---
        elif image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return Image.open(image_path).convert("RGB")
            
        else:
            return None

    except Exception as e:
        log(f"Skipping {os.path.basename(image_path)}: {e}")
        return None

def process_image(model, processor, config, image_path) -> str:
    """
    Core processing logic. 
    """
    # 1. Load Image
    image = load_pixels_robust(image_path)
    if image is None:
        raise ValueError("Image loading failed.")
    
    image = resize_with_padding(image)

    # 2. Prompt 
    # (Anatomy Bucket Strategy)
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
    - SPINE
    
    Answer with just the word.
    """

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    prompt = apply_chat_template(processor, config, messages)
    
    # 3. Generate
    response = generate(
        model, processor, prompt, image, 
        verbose=False, max_tokens=10, temp=0.0
    )
    
    answer = response.text.strip().upper().replace(".", "")
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing images to scan")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        log(f"CRITICAL: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    log("Initializing Model...")
    model, processor = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True, "use_fast": True})
    config = load_config(MODEL_PATH)

    if not os.path.isdir(args.input_dir):
        log(f"Error: {args.input_dir} is not a directory.")
        sys.exit(1)

    # Gather supported files
    files = [
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if f.lower().endswith(('.dcm', '.dicom', '.png', '.jpg', '.jpeg'))
    ]
    files.sort()
    
    log(f"Found {len(files)} images. Starting batch processing...") 

    for filepath in files:
        filename = os.path.basename(filepath)
        answer = process_image(model, processor, config, filepath)
        log(f"{filename},{answer}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()