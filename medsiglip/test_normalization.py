import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from rich.console import Console
from rich.table import Table

# --- CONFIG ---
MODEL_PATH = "medsiglip/medsiglip-google-448"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TEST_IMAGE = "data/AP_CXR_448"  # We will pick the first file found here
PROMPTS = ["A chest x-ray", "A photo of a cat"]

console = Console()

def get_image_tensor(path, strategy):
    img = Image.open(path).convert("RGB")
    if img.size != (448, 448):
        img = img.resize((448, 448), resample=Image.Resampling.LANCZOS)
    
    arr = np.array(img).transpose(2, 0, 1) # CxHxW
    tensor = torch.tensor(arr, dtype=torch.float32) / 255.0 # 0-1 range
    
    if strategy == "siglip_standard":
        # (x - 0.5) / 0.5 -> Range [-1, 1]
        tensor = (tensor - 0.5) / 0.5
        
    elif strategy == "imagenet":
        # ImageNet Mean/Std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
    elif strategy == "raw":
        # No normalization, just 0-1
        pass

    return tensor.unsqueeze(0).to(DEVICE)

def main():
    console.print(f"[bold cyan]🔍 Preprocessing Diagnostic ({DEVICE})[/bold cyan]")
    
    # 1. Load Model
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    # 2. Find a Test Image
    files = [f for f in os.listdir(TEST_IMAGE) if f.endswith('.png')]
    if not files:
        console.print("[red]No images found to test![/red]")
        return
    img_path = os.path.join(TEST_IMAGE, files[0])
    console.print(f"Testing on: [yellow]{files[0]}[/yellow]")

    # 3. Prepare Text
    text_inputs = tokenizer(PROMPTS, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)

    # 4. Run Strategies
    strategies = ["siglip_standard", "imagenet", "raw"]
    
    table = Table(title="Logit Response by Strategy")
    table.add_column("Strategy", style="bold")
    table.add_column("CXR Logit", justify="right")
    table.add_column("Cat Logit", justify="right")
    table.add_column("Sigmoid(CXR)", justify="right", style="green")
    
    for strat in strategies:
        pixel_values = get_image_tensor(img_path, strat)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, **text_inputs)
            logits = outputs.logits_per_image[0] # [logit_cxr, logit_cat]
            probs = torch.sigmoid(logits)
            
        logit_cxr = logits[0].item()
        logit_cat = logits[1].item()
        prob_cxr = probs[0].item()
        
        # Color coding: Green if > 0 (Model is active), Red if < -5 (Model is dead)
        style = "green" if logit_cxr > 0 else ("red" if logit_cxr < -5 else "yellow")
        
        table.add_row(
            strat,
            f"[{style}]{logit_cxr:.2f}[/{style}]",
            f"{logit_cat:.2f}",
            f"{prob_cxr:.1%}"
        )

    console.print(table)
    console.print("\n[dim]Goal: Find the strategy where 'CXR Logit' is POSITIVE (> 0.0).[/dim]")

if __name__ == "__main__":
    main()