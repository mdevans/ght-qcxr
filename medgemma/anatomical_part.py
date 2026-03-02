# --- SILENCE LOGS ---
import warnings
import logging
import os
warnings.simplefilter("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import argparse
import numpy as np
import pydicom
from PIL import Image, ImageOps
from typing import Optional, Tuple, Any, Dict

# MLX Imports
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from transformers import PreTrainedTokenizer

MODEL_PATH = "./medgemma_vision"

def load_model(model_path: str):
    print(f"🧠 Loading Visual Cortex...", end="\r")
    model, processor = load(model_path, tokenizer_config={"trust_remote_code": True, "use_fast": True})
    config = load_config(model_path)
    print(f"🧠 Visual Cortex Ready.      ")
    return model, processor, config

def load_dicom_data(image_path: str) -> Tuple[Optional[Image.Image], Dict[str, str]]:
    """
    Loads DICOM using strict Header rules for Photometric Interpretation.
    Also extracts relevant metadata tags.
    """
    try:
        ds = pydicom.dcmread(image_path)
        
        # 1. Extract Metadata
        meta = {
            "BodyPartExamined": str(ds.get("BodyPartExamined", "Not Found in Header")),
            "SeriesDescription": str(ds.get("SeriesDescription", "N/A")),
            "Photometric": str(ds.get("PhotometricInterpretation", "Unknown"))
        }

        # 2. Extract Raw Pixels & Normalize
        pixel_array = ds.pixel_array.astype(float)
        
        # Auto-Windowing (0-255)
        if pixel_array.max() != pixel_array.min():
            scaled = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        else:
            scaled = pixel_array * 0
        
        image = Image.fromarray(np.uint8(scaled))
        
        # 3. STRICT HEADER-BASED INVERSION
        # We trust the tag. We do NOT look at the pixels to guess.
        if meta["Photometric"] == "MONOCHROME1":
            # Monochrome1 means 0=White, so we invert to make it standard (0=Black)
            image = ImageOps.invert(image)
            meta["Inverted"] = "True (Monochrome1 detected)"
        else:
            meta["Inverted"] = "False"

        return image.convert("RGB"), meta

    except Exception as e:
        print(f"❌ DICOM Error: {e}")
        return None, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to dicom file")
    args = parser.parse_args()

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    model, processor, config = load_model(MODEL_PATH)

    # 2. Load Image & Metadata
    if not args.image_path.lower().endswith(('.dcm', '.dicom')):
        print("Error: This script is optimized for DICOM files to check headers.")
        sys.exit(1)

    image, meta = load_dicom_data(args.image_path)
    if image is None:
        sys.exit(1)

    # 3. Open-Ended Prompt
    # We allow the model to say whatever it sees, or "Unknown"
    query = """
    Identify the primary anatomical body part shown in this medical image.
    Return strictly the name of the body part used in General Radiograpy.
    
    If the image is not recognizable as a specific body part or is not medical, return "Unknown".
    Do not add extra text or punctuation.
    """

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    prompt = apply_chat_template(processor, config, messages)

    # 4. Generate
    response = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        max_tokens=20,
        verbose=False,
        temp=0.0
    )

    ai_finding = response.text.strip().replace("Anatomical part:", "").replace(".", "")

    # 5. Print Comparison
    print("\n" + "="*40)
    print("💀 ANATOMICAL IDENTIFICATION")
    print("-" * 40)
    print(f"🤖 AI Vision Says:       \033[1;32m{ai_finding}\033[0m")
    print(f"📄 DICOM Header Says:    \033[1;36m{meta['BodyPartExamined']}\033[0m")
    print("-" * 40)
    print(f"ℹ️  Photometric Tag:      {meta['Photometric']}")
    print(f"ℹ️  Inversion Applied:    {meta['Inverted']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()