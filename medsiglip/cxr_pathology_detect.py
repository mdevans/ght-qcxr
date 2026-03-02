import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageOps
import numpy as np
from rich.console import Console
from rich.table import Table

# CONFIG
MODEL_PATH = "medsiglip/medsiglip-google-448"
# IMAGE_PATH = "data/cxr/cxr_TB_2.jpg" # Your TB image
IMAGE_PATH = "data/cxr/cxr_normal_0.jpg" # Your Normal image
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# TEST: Can it distinguish TB from Normal?
PROMPTS = [
    "A normal chest X-ray with clear lungs",
    "A chest X-ray showing Tuberculosis or opacities",
    "A chest X-ray showing pleural effusion",
    "Non-medical image"
]

def main():
    console = Console()
    console.print(f"[bold green]🦠 Running Pathology Detector ({DEVICE})[/bold green]")

    # 1. LOAD
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # 2. IMAGE PREP (Simple Pipeline)
    raw_img = Image.open(IMAGE_PATH).convert("RGB")
    img_fixed = ImageOps.autocontrast(raw_img) 
    img_resized = img_fixed.resize((448, 448), resample=Image.Resampling.LANCZOS)
    
    # To Tensor
    img_array = np.array(img_resized).transpose(2, 0, 1) 
    pixel_values = torch.tensor(img_array, dtype=torch.float32) / 255.0
    pixel_values = (pixel_values - 0.5) / 0.5
    pixel_values = pixel_values.unsqueeze(0).to(DEVICE)

    # 3. TEXT PREP
    text_inputs = tokenizer(PROMPTS, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)

    # 4. INFERENCE
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, **text_inputs)
        logits = outputs.logits_per_image[0]
        probs = F.softmax(logits, dim=0)

    # 5. REPORT
    table = Table(title="Pathology Analysis")
    table.add_column("Diagnosis", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right", style="bold yellow")
    
    results = sorted(zip(PROMPTS, logits.tolist(), probs.tolist()), key=lambda x: x[2], reverse=True)
    
    for label, score, prob in results:
        color = "green" if prob > 0.5 else "dim"
        table.add_row(label, f"{score:.2f}", f"[{color}]{prob:.2%}[/{color}]")

    console.print(table)
    
    # Verdict
    top_diagnosis = results[0][0]
    if "Tuberculosis" in top_diagnosis:
        console.print("\n[bold green]✅ SUCCESS: Model correctly identified the pathology![/bold green]")
    elif "normal" in top_diagnosis:
        console.print("\n[bold red]❌ FAILURE: Model missed the disease (False Negative).[/bold red]")

if __name__ == "__main__":
    main()