import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from collections import defaultdict

# --- CONFIGURATION ---
MODEL_PATH = "medsiglip/medsiglip-google-448"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Directory Paths (Adjust these to match your actual folder structure)
DIRS = {
    "AP":  "data/AP_CXR_448",   # Folder with 100 AP images
    "PA":  "data/PA_CXR_448",   # Folder with 100 PA images
    "LAT": "data/LAT_CXR_448"   # Folder with 100 Lateral images
}

# The Prompts (Classes 0, 1, 2, 3)
PROMPTS = [
    "An anterior-posterior chest x-ray", # 0: AP
    "An posterior-anterior chest x-ray", # 1: PA
    "A lateral chest x-ray",             # 2: Lateral
    "A non-medical image"                # 3: Other
]

# Map directory keys to expected Prompt Index
LABEL_MAP = {
    "AP": 0,
    "PA": 1,
    "LAT": 2
}

def get_image_tensor(image_path):
    """
    Loads and preprocesses an image exactly like the single-image test.
    """
    try:
        raw_img = Image.open(image_path).convert("RGB")
        # Ensure 448x448 (Assuming input is already resized, but safety check)
        if raw_img.size != (448, 448):
            raw_img = raw_img.resize((448, 448), resample=Image.Resampling.LANCZOS)

        img_array = np.array(raw_img).transpose(2, 0, 1) # HWC -> CHW
        pixel_values = torch.tensor(img_array, dtype=torch.float32) / 255.0
        # Normalization (0.5 mean/std)
        pixel_values = (pixel_values - 0.5) / 0.5
        return pixel_values.unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def main():
    console = Console()
    console.print(f"[bold cyan]🧪 MedSigLIP Batch Evaluator ({DEVICE})[/bold cyan]")
    
    # 1. LOAD MODEL
    console.print("   Loading Model...", style="dim")
    try:
        model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return

    # 2. PREPARE TEXT EMBEDDINGS (Done once)
    text_inputs = tokenizer(
        PROMPTS, padding="max_length", truncation=True, return_tensors="pt"
    ).to(DEVICE)
    
    # 3. RUN EVALUATION
    def create_stats():
        return {
            "total": 0,
            "correct": 0,
            "frontal_correct": 0,
            "predicted": defaultdict(int)
        }

    results = defaultdict(create_stats)
    
    total_images = 0
    
    for class_name, folder_path in DIRS.items():
        if not os.path.exists(folder_path):
            console.print(f"[yellow]⚠️  Skipping {class_name}: Folder not found ({folder_path})[/yellow]")
            continue
            
        expected_idx = LABEL_MAP[class_name]
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        console.print(f"   Processing [bold]{class_name}[/bold] ({len(files)} images)...")
        
        for filename in track(files, description=f"Evaluating {class_name}..."):
            img_path = os.path.join(folder_path, filename)
            pixel_values = get_image_tensor(img_path)
            
            if pixel_values is None: 
                continue

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, **text_inputs)
                logits = outputs.logits_per_image[0]
                probs = F.softmax(logits, dim=0)
                pred_idx = torch.argmax(probs).item()

            # Record Stats
            results[class_name]["total"] += 1
            results[class_name]["predicted"][pred_idx] += 1
            
            # Strict Accuracy Check (Must be exact class)
            if pred_idx == expected_idx:
                results[class_name]["correct"] += 1
            
            # "Frontal" Accuracy Check (AP or PA counts as correct for Frontal classes)
            # If input is AP or PA, and prediction is AP or PA (0 or 1), it's "Frontal Correct"
            is_frontal_input = expected_idx in [0, 1]
            is_frontal_pred = pred_idx in [0, 1]
            
            if is_frontal_input and is_frontal_pred:
                results[class_name]["frontal_correct"] += 1
            elif expected_idx == 2 and pred_idx == 2: # Lateral must be Lateral
                results[class_name]["frontal_correct"] += 1

            total_images += 1

    # 4. FINAL REPORT
    print_summary(console, results)

def print_summary(console, results):
    # Main Statistics Table
    table = Table(title="MedSigLIP Zero-Shot Performance")
    
    table.add_column("Class (True)", style="cyan", justify="left")
    table.add_column("Total", justify="center")
    table.add_column("Strict Acc %", justify="right", style="bold")
    table.add_column("Frontal/Lat Acc %", justify="right", style="green")
    table.add_column("Confused With", justify="left", style="dim")

    grand_total = 0
    grand_correct = 0
    grand_frontal_correct = 0

    for class_name, stats in results.items():
        total = stats["total"]
        if total == 0: continue
        
        strict_acc = (stats["correct"] / total) * 100
        frontal_acc = (stats["frontal_correct"] / total) * 100
        
        grand_total += total
        grand_correct += stats["correct"]
        grand_frontal_correct += stats["frontal_correct"]

        # Confusion String
        confusions = []
        for pred_idx, count in stats["predicted"].items():
            if pred_idx != LABEL_MAP[class_name] and count > 0:
                label = PROMPTS[pred_idx].split()[1] # Grab "anterior-posterior" or "lateral"
                confusions.append(f"{label}: {count}")
        
        conf_str = ", ".join(confusions) if confusions else "None"

        table.add_row(
            class_name,
            str(total),
            f"{strict_acc:.1f}%",
            f"{frontal_acc:.1f}%",
            conf_str
        )

    console.print("\n")
    console.print(table)
    
    if grand_total > 0:
        overall_strict = (grand_correct / grand_total) * 100
        overall_robust = (grand_frontal_correct / grand_total) * 100
        
        console.print(f"\n[bold]Overall Strict Accuracy (AP vs PA vs LAT):[/bold] [yellow]{overall_strict:.1f}%[/yellow]")
        console.print(f"[bold]Overall Robust Accuracy (Frontal vs LAT):[/bold]   [green]{overall_robust:.1f}%[/green]")
        console.print("[dim]Note: 'Strict' requires AP/PA distinction. 'Robust' groups AP+PA as Frontal.[/dim]")

if __name__ == "__main__":
    main()