import os
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
from rich.console import Console
from rich.table import Table

# --- CONFIGURATION ---
# 1. Paths
MODEL_PATH = "medsiglip/medsiglip-google-448"
IMAGE_PATH = "data/cxr/cxr_TB_1.jpg"

# 2. Device (Apple Silicon Optimization)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 3. Define the "Zero-Shot" Labels
# MedSigLIP calculates the distance between the image and these text prompts.
# OPTIMIZED LABELS FOR "IS IT A CXR?"
CANDIDATE_LABELS = [
    "Chest X-ray",           # The target
    "Abdominal X-ray",       # Closest confusing medical image
    "Extremity X-ray",       # Bones/Limbs (Hands, Feet, Knees)
    "CT Scan",               # Different modality
    "Non-medical image"      # Random noise/photos
]

def main():
    console = Console()
    console.print(f"[bold green]🚀 CXR View Detector initialized on {DEVICE.upper()}[/bold green]")

    # --- STEP 1: LOAD MODEL ---
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]❌ Model not found at {MODEL_PATH}[/red]")
        return

    try:
        console.print(f"⏳ Loading MedSigLIP (880M params)...")
        # Load Model to MPS
        model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
        model.eval()
        # Load Processor (Handles Resizing & Normalization automatically)
        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    except Exception as e:
        console.print(f"[red]❌ Load Error:[/red] {e}")
        return

    # --- STEP 2: LOAD IMAGE ---
    if not os.path.exists(IMAGE_PATH):
        console.print(f"[red]❌ Image not found at {IMAGE_PATH}[/red]")
        return
    
    # We load the full High-Res image. 
    # The processor will automatically resize it to 448x448, preserving the content logic.
    raw_image = Image.open(IMAGE_PATH).convert("RGB")
    width, height = raw_image.size
    console.print(f"🖼️  Image Loaded: [bold]{os.path.basename(IMAGE_PATH)}[/bold] ({width}x{height} pixels)")

    # --- STEP 3: RUN ZERO-SHOT CLASSIFICATION ---
    console.print("🧪 Running Zero-Shot Inference...")
    
    # Process inputs (Text + Image)
    inputs = processor(
        text=CANDIDATE_LABELS,
        images=raw_image,
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        # The model returns 'logits_per_image' which is the similarity score 
        # between the image and EACH text label.
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Shape: [1, num_labels]
        
        # Convert logits to Probabilities (Softmax)
        probs = logits_per_image.softmax(dim=1) # Shape: [1, num_labels]
        
        # Move back to CPU for printing
        probs_list = probs.cpu().tolist()[0]

    # --- STEP 4: REPORT ---
    # Create a nice table for the results
    table = Table(title="View Classification Results")
    table.add_column("View Type", style="cyan")
    table.add_column("Confidence", justify="right", style="magenta")
    table.add_column("Bar", style="green")

    # Sort results by confidence
    results = sorted(zip(CANDIDATE_LABELS, probs_list), key=lambda x: x[1], reverse=True)

    for label, score in results:
        # Create a simple visual bar
        bar_length = int(score * 20)
        bar = "█" * bar_length
        table.add_row(label, f"{score:.2%}", bar)

    console.print(table)

    # Final Decision
    top_label, top_score = results[0]
    if "Frontal" in top_label and top_score > 0.8:
        console.print(f"\n[bold green]✅ CONFIRMED: Valid Frontal CXR ({top_score:.1%})[/bold green]")
    else:
        console.print(f"\n[bold red]⚠️ WARNING: Image might not be a Frontal CXR (Detected: {top_label})[/bold red]")

if __name__ == "__main__":
    main()