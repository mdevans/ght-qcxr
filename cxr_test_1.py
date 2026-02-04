# --- 1. SILENCE LOGS (Must be at the very top) ---
import warnings
import logging
import os

# Filter internal DeprecationWarnings from libraries
warnings.simplefilter("ignore")

# Force Transformers library to only show ERRORS (hides "Using a slow processor" warning)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import json
from typing import cast, Any, Tuple, Optional, List, Dict

import numpy as np
import pydicom
from PIL import Image, ImageOps

# Rich Imports (For Pretty Printing)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# MLX & Transformers Imports
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from transformers import PreTrainedTokenizer

# --- CONFIGURATION ---
MODEL_PATH = "./medgemma_vision" 
TEST_FILE = "cxr/cxr1.dcm"

def load_unified_model(model_path: str) -> Tuple[Any, PreTrainedTokenizer, Any]:
    print(f"🧠 Loading MedGemma 1.5 from {model_path}...")
    try:
        # We still set use_fast=True for performance, even if logs are silenced
        model, raw_processor = load(
            model_path, 
            tokenizer_config={"trust_remote_code": True, "use_fast": True}
        )
        
        # Manually force the property if the config didn't catch it
        if hasattr(raw_processor, "use_fast"):
            raw_processor.use_fast = True

        config = load_config(model_path)
        processor = cast(PreTrainedTokenizer, raw_processor)
        
        # --- VERIFICATION BLOCK ---
        print("\n🔍 DIAGNOSTICS:")
        print(f"   • Tokenizer Class: {type(processor.tokenizer).__name__}")
        print(f"   • Image Processor: {type(processor.image_processor).__name__}")
        print("-" * 20)
        
        return model, processor, config
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def process_dicom(dicom_path: str) -> Optional[Image.Image]:
    """Pure function: DICOM -> RGB Image with Auto-Windowing."""
    try:
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(float)
        
        # Auto-Windowing
        if pixel_array.max() != pixel_array.min():
            scaled = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        else:
            scaled = pixel_array * 0 

        image = Image.fromarray(np.uint8(scaled))
        
        # Fix Inverted X-Rays
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            image = ImageOps.invert(image)
            
        return image.convert("RGB")
    except Exception as e:
        print(f"❌ DICOM Processing Error: {e}")
        return None

def analyze_cxr(model: Any, processor: PreTrainedTokenizer, config: Any, image_path: str, user_query: str) -> None:
    # 1. Load Image
    image: Optional[Image.Image] = None
    if image_path.lower().endswith('.dcm'):
        image = process_dicom(image_path)
    
    if image is None:
        print(f"⚠️  Could not process image: {image_path}")
        return

    # 2. PROMPT CONSTRUCTION
    # Format as list to trigger the <image> token insertion from the template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_query}
            ]
        }
    ]
    
    raw_prompt = apply_chat_template(processor, config, messages)
    formatted_prompt = cast(str, raw_prompt)

    # Initialize Rich Console
    console = Console()
    console.print(f"\n[bold blue]👀 Analyzing Image:[/bold blue] {image_path}")
    start_time = time.time()

    # 3. Generate
    response_text = generate(
        model=model,
        processor=processor,
        prompt=formatted_prompt,
        image=cast(Any, image), 
        max_tokens=300,
        verbose=False,
        temp=0.0
    )
    
    end_time = time.time()

    # --- 4. PARSING & PRETTY PRINTING ---
    try:
        # Clean the response: remove markdown code blocks if present
        clean_text = response_text.text.strip()
        # Handle ```json and ``` blocks robustly
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        # Parse JSON
        data = json.loads(clean_text)

        # A. Create a Technical Quality Table
        table = Table(title="CXR Technical Quality", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Assessment", style="green")

        # Add rows dynamically
        # We use .get() to avoid crashing if a key is missing
        rotation_val = data.get("rotation_detected")
        if isinstance(rotation_val, bool):
            rotation_str = "[red]DETECTED[/red]" if rotation_val else "None"
        else:
            rotation_str = str(rotation_val)

        table.add_row("Projection", str(data.get("projection", "N/A")))
        table.add_row("Rotation", rotation_str)
        table.add_row("Inspiration", str(data.get("inspiration", "N/A")))
        table.add_row("Exposure", str(data.get("exposure", "N/A")))
        table.add_row("Diagnostic Quality", str(data.get("diagnostic_quality", "N/A")))

        # B. Print the outputs
        console.print("\n")
        console.print(table)
        
        # C. Findings Panel
        findings_text = data.get("findings", "No findings provided")
        console.print(Panel(
            Text(findings_text, justify="left"),
            title="[bold yellow]Clinical Findings[/bold yellow]",
            border_style="yellow"
        ))
        
    except json.JSONDecodeError:
        # Fallback if model fails to generate valid JSON
        console.print("[bold red]❌ Failed to parse JSON response. Raw output:[/bold red]")
        console.print(response_text)
    except Exception as e:
        console.print(f"[bold red]❌ An error occurred during display: {e}[/bold red]")
        console.print(response_text)

    console.print(f"\n[dim]⏱️  Inference Time: {end_time - start_time:.2f}s[/dim]\n")

if __name__ == "__main__":
    print("🚀 GHT-qCXR MVP: Visual Cortex Initializing on M4...")
    
    model, processor, config = load_unified_model(MODEL_PATH)
    
    # Updated Categorical Query for cleaner classification
    query = """
    Analyze this chest X-ray. 
    Return a strictly valid JSON object. Do not use Markdown code blocks. 
    
    Output must adhere strictly to this schema:
    {
      "rotation_detected": (boolean, true if the patient is significantly rotated),
      
      "inspiration": (string, choose exactly one: "Under-inflated", "Adequate", "Hyper-inflated"),
      
      "projection": (string, choose exactly one: "PA", "AP"),
      
      "exposure": (string, choose exactly one: "Under-exposed", "Adequate", "Over-exposed"),
      
      "diagnostic_quality": (string, choose exactly one: "Non-diagnostic", "Suboptimal", "Optimal"),
      
      "findings": "Brief summary of clinical findings"
    }
    """

    if os.path.exists(TEST_FILE):
        analyze_cxr(model, processor, config, TEST_FILE, query)
    else:
        print(f"⚠️  Test file '{TEST_FILE}' not found.")