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

def normalize_laterality(lat_code: str) -> str:
    """Normalizes DICOM 'L'/'R' codes to full words."""
    lat_code = lat_code.upper().strip()
    if lat_code == "L": return "Left"
    if lat_code == "R": return "Right"
    if lat_code == "B": return "Bilateral"
    return "Unknown" if not lat_code else lat_code

def normalize_projection(proj_code: str) -> str:
    """Groups specific views into main categories for easier comparison."""
    proj_code = proj_code.upper().strip()
    if proj_code in ["PA", "AP"]: return "Frontal"
    if proj_code in ["LL", "RL", "LATERAL", "LAT"]: return "Lateral"
    return "Unknown" if not proj_code else proj_code

def load_dicom_full(image_path: str) -> Tuple[Optional[Image.Image], Dict[str, str]]:
    try:
        ds = pydicom.dcmread(image_path)
        
        # 1. Extract & Normalize Headers
        raw_lat = str(ds.get("ImageLaterality", ds.get("Laterality", "")))
        raw_view = str(ds.get("ViewPosition", ""))
        
        meta = {
            "Anatomy": str(ds.get("BodyPartExamined", "Unknown")).strip(),
            "Laterality": normalize_laterality(raw_lat),
            "Projection": normalize_projection(raw_view),
            "RawView": raw_view, 
            "Photometric": str(ds.get("PhotometricInterpretation", "Unknown"))
        }

        # 2. Extract Pixels
        pixel_array = ds.pixel_array.astype(float)
        if pixel_array.max() != pixel_array.min():
            scaled = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        else:
            scaled = pixel_array * 0
        
        image = Image.fromarray(np.uint8(scaled))
        
        # 3. Handle Inversion
        if meta["Photometric"] == "MONOCHROME1":
            image = ImageOps.invert(image)
        
        return image.convert("RGB"), meta

    except Exception as e:
        print(f"❌ DICOM Error: {e}")
        return None, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to dicom file")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    model, processor, config = load_model(MODEL_PATH)

    if not args.image_path.lower().endswith(('.dcm', '.dicom')):
        print("Error: Input must be a DICOM file.")
        sys.exit(1)

    image, meta = load_dicom_full(args.image_path)
    if image is None: sys.exit(1)

    # --- UPDATED PROMPT: Handling Bilateral Context ---
    query = """
    Analyze this medical image strictly.
    
    1. ANATOMY: What is the primary body part? (e.g. Chest, Knee, Shoulder)
    2. PROJECTION: Is it "Frontal" (PA/AP) or "Lateral"?
    3. LATERALITY:
       - If the image is a standard PA/AP Chest or Pelvis showing BOTH sides, return "Bilateral".
       - If it shows only ONE limb/side, return "Left" or "Right".
       
    Return format:
    Anatomy: [Result]
    Projection: [Result]
    Laterality: [Result]
    """

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    prompt = apply_chat_template(processor, config, messages)

    # Generate
    response = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        max_tokens=60,
        verbose=False,
        temp=0.0
    )

    # Parse AI Output
    lines = response.text.strip().split('\n')
    ai_results = {"Anatomy": "Unknown", "Laterality": "Unknown", "Projection": "Unknown"}
    
    for line in lines:
        if "Anatomy:" in line: ai_results["Anatomy"] = line.split(":", 1)[1].strip()
        if "Laterality:" in line: ai_results["Laterality"] = line.split(":", 1)[1].strip()
        if "Projection:" in line: ai_results["Projection"] = line.split(":", 1)[1].strip()

    # --- LOGIC CORRECTION: FORCE BILATERAL FOR FRONTAL CHEST ---
    # Even if AI says "Left" (because heart is on left), we correct it for reporting
    # if it's a Frontal Chest X-Ray.
    
    is_chest = "CHEST" in ai_results["Anatomy"].upper()
    is_frontal = "FRONTAL" in ai_results["Projection"].upper()
    
    if is_chest and is_frontal and ai_results["Laterality"] != "Bilateral":
        # Override visual laterality for standard CXR
        ai_results["Laterality"] = "Bilateral (Implicit)"

    # --- COMPARISON LOGIC ---
    print("\n" + "="*65)
    print(f"🔎 FULL IMAGE CLASSIFICATION: {os.path.basename(args.image_path)}")
    print("="*65)
    print(f"{'AXIS':<15} | {'AI VISION':<20} | {'DICOM HEADER':<20} | {'MATCH'}")
    print("-" * 65)

    # 1. Anatomy Check
    match_anat = "✅" if (meta["Anatomy"].upper() in ai_results["Anatomy"].upper() or 
                           ai_results["Anatomy"].upper() in meta["Anatomy"].upper()) else "⚠️"
    if meta["Anatomy"] == "Unknown": match_anat = "ℹ️"
    
    print(f"{'Anatomy':<15} | \033[1;32m{ai_results['Anatomy']:<20}\033[0m | \033[1;36m{meta['Anatomy']:<20}\033[0m | {match_anat}")

    # 2. Projection Check
    ai_proj = ai_results["Projection"].upper()
    head_proj = meta["Projection"].upper()

    match_proj = "✅"
    if head_proj == "UNKNOWN": match_proj = "ℹ️"
    elif "FRONTAL" in ai_proj and head_proj == "FRONTAL": match_proj = "✅"
    elif "LATERAL" in ai_proj and head_proj == "LATERAL": match_proj = "✅"
    else: match_proj = "⚠️"

    print(f"{'Projection':<15} | \033[1;32m{ai_results['Projection']:<20}\033[0m | \033[1;36m{meta['RawView']} ({head_proj}){'':<5}\033[0m | {match_proj}")

    # 3. Laterality Check (Context Aware)
    ai_lat = ai_results["Laterality"].upper()
    head_lat = meta["Laterality"].upper()
    
    match_lat = "✅"
    
    # If inherent bilateral (Chest PA), we accept "Unknown" in header as valid match
    if "BILATERAL" in ai_lat and head_lat == "UNKNOWN":
        match_lat = "✅" 
    elif head_lat == "UNKNOWN": 
        match_lat = "ℹ️"
    elif head_lat not in ai_lat: 
        match_lat = "⚠️"

    print(f"{'Laterality':<15} | \033[1;32m{ai_results['Laterality']:<20}\033[0m | \033[1;36m{meta['Laterality']:<20}\033[0m | {match_lat}")
    
    print("-" * 65)
    print("Match Key: ✅=Agree  ⚠️=Mismatch  ℹ️=Missing Header Data")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()