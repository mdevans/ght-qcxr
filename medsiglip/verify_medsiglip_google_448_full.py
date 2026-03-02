import os
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
from rich.console import Console

# 1. SETUP
# Point this to your specific folder
LOCAL_MODEL_PATH = "medsiglip/medsiglip-google-448"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
console = Console()

console.print(f"[bold green]🔍 Checking Full MedSigLIP at: {LOCAL_MODEL_PATH}[/bold green]")

if not os.path.exists(LOCAL_MODEL_PATH):
    console.print(f"[red]❌ Error: Folder not found at {LOCAL_MODEL_PATH}[/red]")
    exit(1)

# 2. LOAD FULL MODEL (Vision + Text)
try:
    console.print(f"⏳ Loading Full Model (Dual-Encoder) to {DEVICE}...")
    
    # Load the joint model
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True).to(DEVICE)
    model.eval()
    
    # Load the Processor (Handles both Image resizing & Text tokenization)
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, use_fast=True)
    
    console.print("✅ Model & Processor Loaded Successfully!")
except Exception as e:
    console.print(f"[bold red]❌ FAILED to load:[/bold red] {e}")
    if "SentencePiece" in str(e):
        console.print("[yellow]💡 TIP: Run 'uv add sentencepiece protobuf' in your terminal.[/yellow]")
    exit(1)

# 3. VERIFY ARCHITECTURE
print("-" * 30)
console.print("[bold]📊 MODEL ARCHITECTURE[/bold]")
print("-" * 30)

# Check for both towers
has_vision = hasattr(model, "vision_model")
has_text = hasattr(model, "text_model")

console.print(f"Vision Tower: {'✅ Present' if has_vision else '❌ Missing'}")
console.print(f"Text Tower:   {'✅ Present' if has_text else '❌ Missing'}")
console.print(f"Params:       {model.num_parameters() / 1e6:.1f} M")

# 4. RUN DUAL-MODAL INFERENCE TEST
print("-" * 30)
console.print(f"[bold]🧪 RUNNING DUAL-ENCODER TEST on {DEVICE}[/bold]")
print("-" * 30)

try:
    # A. Prepare Inputs
    dummy_image = Image.new('RGB', (448, 448), color='black')
    dummy_text = ["Chest X-Ray", "Normal lung parenchyma"]
    
    # Process both text and image together
    inputs = processor(
        text=dummy_text, 
        images=dummy_image, 
        padding="max_length", 
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        # B. Run the Model
        # This runs both encoders and calculates similarity scores
        outputs = model(**inputs)
        
        # Extract Embeddings
        img_embeds = outputs.image_embeds  # Shape: [1, 1152]
        txt_embeds = outputs.text_embeds   # Shape: [2, 1152]
        
        # Calculate Zero-Shot Probabilities (Softmax of dot product)
        logits_per_image = outputs.logits_per_image # [1, 2]
        probs = logits_per_image.softmax(dim=1)

    # C. Report Results
    console.print(f"Image Embed Shape: {img_embeds.shape} ✅")
    console.print(f"Text Embed Shape:  {txt_embeds.shape}  ✅")
    console.print(f"Logits Shape:      {logits_per_image.shape} (Image vs 2 Texts)")
    
    if img_embeds.shape[1] == 1152 and txt_embeds.shape[1] == 1152:
        console.print("\n[bold green]✅ SUCCESS: Full Dual-Encoder is operational.[/bold green]")
    else:
        console.print("\n[bold red]❌ FAIL: Unexpected embedding dimensions.[/bold red]")

except Exception as e:
    console.print(f"[bold red]❌ CRASHED during inference:[/bold red] {e}")