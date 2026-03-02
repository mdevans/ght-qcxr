# --- 1. SILENCE LOGS (Must be at the very top) ---
import warnings
import logging
import os

# Filter internal DeprecationWarnings from libraries
warnings.simplefilter("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import json
import argparse
import re
# FIX: Added Dict to imports
from typing import cast, Any, Tuple, Optional, Dict

import numpy as np
import pydicom
from PIL import Image, ImageOps

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from transformers import PreTrainedTokenizer

# --- CONFIGURATION ---
MODEL_PATH = "./medgemma_vision" 

def load_unified_model(model_path: str) -> Tuple[Any, PreTrainedTokenizer, Any]:
    print(f"🧠 Loading MedGemma 1.5 from {model_path}...")
    try:
        model, raw_processor = load(
            model_path, 
            tokenizer_config={"trust_remote_code": True, "use_fast": True}
        )
        if hasattr(raw_processor, "use_fast"):
            raw_processor.use_fast = True
        config = load_config(model_path)
        processor = cast(PreTrainedTokenizer, raw_processor)
        return model, processor, config
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def process_dicom(dicom_path: str) -> Optional[Image.Image]:
    try:
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(float)
        if pixel_array.max() != pixel_array.min():
            scaled = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        else:
            scaled = pixel_array * 0 
        image = Image.fromarray(np.uint8(scaled))
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            image = ImageOps.invert(image)
        return image.convert("RGB")
    except Exception as e:
        print(f"❌ DICOM Processing Error: {e}")
        return None

def load_image(image_path: str) -> Optional[Image.Image]:
    try:
        ext = os.path.splitext(image_path)[1].lower()
        image = None
        if ext in ['.dcm', '.dicom']:
            image = process_dicom(image_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
            image = Image.open(image_path).convert("RGB")
        
        if image is None: return None

        # Robust Inversion Check
        img_gray = np.array(image.convert("L"))
        corners = [
            img_gray[:50, :50], img_gray[:50, -50:],
            img_gray[-50:, :50], img_gray[-50:, -50:]
        ]
        avg_bg = np.median([np.mean(c) for c in corners])
        if avg_bg > 128:
            print(f"🔄 Inverted X-Ray Detected (Bg: {avg_bg:.0f}). Flipping...")
            image = ImageOps.invert(image)
        return image
    except Exception as e:
        print(f"❌ Image Load Error: {e}")
        return None

def extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extracts JSON. Prioritizes markdown code blocks.
    Ignores valid-looking but garbage schema definitions echoed by the model.
    """
    try:
        # Strategy 1: Look for ```json ... ``` blocks (Most reliable)
        matches = re.findall(r"```json(.*?)```", text, re.DOTALL)
        if matches:
            # If multiple blocks, usually the last one is the result (after the thought process)
            for match in reversed(matches):
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # Strategy 2: Look for generic ``` ... ``` blocks
        matches = re.findall(r"```(.*?)```", text, re.DOTALL)
        if matches:
            for match in reversed(matches):
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Strategy 3: Naive brace finding (Fallback)
        # Find the LAST pair of braces in the text, as the answer is usually at the end
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            potential_json = text[start:end+1]
            return json.loads(potential_json)
            
        raise ValueError("No JSON content found")
        
    except Exception as e:
        raise ValueError(f"JSON Extraction Failed: {e}")

def analyze_cxr(model: Any, processor: PreTrainedTokenizer, config: Any, image_path: str, user_query: str) -> None:
    image = load_image(image_path)
    if image is None: return

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_query}]}]
    raw_prompt = apply_chat_template(processor, config, messages)
    formatted_prompt = cast(str, raw_prompt)

    console = Console()
    console.print(f"\n[bold blue]👀 Analyzing Image:[/bold blue] {image_path}")
    start_time = time.time()

    response_result = generate(
        model=model,
        processor=processor,
        prompt=formatted_prompt,
        image=cast(Any, image), 
        max_tokens=500, # Increased slightly to allow for thought process
        verbose=False,
        temp=0.0
    )
    end_time = time.time()

    try:
        # FIX: Using the new robust extractor
        data = extract_json(response_result.text)

        # Logic Check: Valid CXR?
        is_valid = data.get("valid_cxr", False)
        projection = data.get("projection", "Unknown")

        # Logic Check: Lateral or Invalid?
        if not is_valid or projection == "Lateral":
            console.print("\n[bold red]⛔ REJECTED: Image Validation Failed[/bold red]")
            
            reason = data.get("rejection_reason", "")
            if projection == "Lateral":
                reason = "Lateral view detected (Protocol requires PA/AP only)"
            elif not reason:
                reason = "Did not meet inclusion criteria"
                
            console.print(Panel(f"[white]{reason}[/white]", title="Rejection Reason", border_style="red"))
            console.print(f"[dim]⏱️  Inference Time: {end_time - start_time:.2f}s[/dim]\n")
            return

        # Render Table
        table = Table(title="CXR Technical Quality", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Assessment", style="green")

        rot_val = data.get("rotation_detected")
        rot_str = "[red]DETECTED[/red]" if rot_val is True else "None"

        table.add_row("Projection", str(projection))
        table.add_row("Rotation", rot_str)
        table.add_row("Inspiration", str(data.get("inspiration", "N/A")))
        table.add_row("Exposure", str(data.get("exposure", "N/A")))
        table.add_row("Quality", str(data.get("diagnostic_quality", "N/A")))

        console.print("\n")
        console.print(table)
        console.print(Panel(Text(data.get("findings", "No findings"), justify="left"), title="[bold yellow]Clinical Findings[/bold yellow]", border_style="yellow"))
        
    except Exception as e:
        console.print(f"[bold red]❌ Processing Error: {e}[/bold red]")
        console.print("[dim]Raw Model Output:[/dim]")
        # Safe fallback printing
        clean_text = response_result.text.replace("[", "\\[") 
        console.print(clean_text)

    console.print(f"\n[dim]⏱️  Inference Time: {end_time - start_time:.2f}s[/dim]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file")
    args = parser.parse_args()

    print("🚀 GHT-qCXR MVP: Initializing on M4...")
    
    if os.path.exists(args.image_path):
        model, processor, config = load_unified_model(MODEL_PATH)
        
        query = """
        Analyze this image.
        
        STEP 1: IDENTIFICATION
        Is this a Chest X-Ray? 
        Identify the projection: PA, AP, or Lateral.

        STEP 2: JSON OUTPUT
        Return a strictly valid JSON object.
        
        Schema:
        {
          "valid_cxr": (boolean, set to false if not a chest X-ray),
          "projection": (string, choose exactly one: "PA", "AP", "Lateral"),
          "rotation_detected": (boolean, true if clavicles are not equidistant from spine),
          "inspiration": (string, choose exactly one: "Under-inflated", "Adequate", "Hyper-inflated"),
          "exposure": (string, choose exactly one: "Under-exposed", "Adequate", "Over-exposed"),
          "diagnostic_quality": (string, choose exactly one: "Non-diagnostic", "Suboptimal", "Optimal"),
          "findings": "Brief summary of clinical findings"
        }
        """
        analyze_cxr(model, processor, config, args.image_path, query)
    else:
        print(f"⚠️  File not found: {args.image_path}")