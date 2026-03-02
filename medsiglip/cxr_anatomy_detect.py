import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageOps
import numpy as np
from rich.console import Console
from rich.table import Table

# CONFIG
MODEL_PATH = "medsiglip/medsiglip-google-448"
#IMAGE_PATH = "data/cxr/cxr_TB_2.jpg"
#IMAGE_PATH = "data/cxr/cxr_normal_0.jpg"
#IMAGE_PATH = "data/cxr/cxr_rib_fracture_pnemo_hemo.png"
IMAGE_PATH = "data/AP_CXR_448/0c34a9ed-5711c7e7-246b8acb-0dc1127d-23518190.png"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# PROMPTS = ["Chest X-Ray", "Knee X-Ray", "Skull X-Ray", "Other"]
PROMPTS = ["An anterior-posterior chest x-ray", "An posterior-anterior chest x-ray", "A lateral chest x-ray", "A non-medical image"]

def main():
    console = Console()
    console.print(f"[bold green]🛠️  Manual Simple Pipeline ({DEVICE})[/bold green]")

    # 1. LOAD (Simple)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # 2. IMAGE PREP (Simple & Correct)
    raw_img = Image.open(IMAGE_PATH).convert("RGB")
    
    # A. Contrast Fix: Stretch histogram to fix "muddy" greys
    #img_fixed = ImageOps.autocontrast(raw_img) 
    
    # B. Resize: High-quality downsample to target 448x448
    #img_resized = img_fixed.resize((448, 448), resample=Image.Resampling.LANCZOS)
    
    # C. Save Debug (So you can see what the model sees)
    #img_resized.save("debug_input_simple.jpg")
    
    # D. To Tensor (Manual Normalization 0.5/0.5)
    img_array = np.array(raw_img).transpose(2, 0, 1) # HWC -> CHW
    pixel_values = torch.tensor(img_array, dtype=torch.float32) / 255.0
    pixel_values = (pixel_values - 0.5) / 0.5
    pixel_values = pixel_values.unsqueeze(0).to(DEVICE)

    # 3. TEXT PREP
    # We let the tokenizer decide what to return.
    text_inputs = tokenizer(
        PROMPTS, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    ).to(DEVICE)

    # 4. INFERENCE
    console.print("🧪 Running Inference...")
    with torch.no_grad():
        # FIX: We unpack **text_inputs instead of asking for keys that don't exist
        outputs = model(pixel_values=pixel_values, **text_inputs)
        
        logits = outputs.logits_per_image[0]
        probs = F.softmax(logits, dim=0)

    # 5. REPORT
    table = Table(title="Simple Pipeline Results")
    table.add_column("Class", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right", style="bold yellow")
    
    results = sorted(zip(PROMPTS, logits.tolist(), probs.tolist()), key=lambda x: x[2], reverse=True)
    
    for label, score, prob in results:
        color = "green" if prob > 0.8 else "red"
        # Positive score = Good match. Negative score = Bad match.
        score_color = "green" if score > 0 else "red"
        
        table.add_row(
            label, 
            f"[{score_color}]{score:.2f}[/{score_color}]", 
            f"[{color}]{prob:.2%}[/{color}]"
        )

    console.print(table)

if __name__ == "__main__":
    main()