import os
import torch
import joblib
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModel
from rich.console import Console
from rich.table import Table
from rich.progress import track

# --- CONFIG ---
MODEL_PATH = "medsiglip/medsiglip-google-448"
CLASSIFIER_PATH = "models/cxr_classifier_production.joblib"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TARGET_SIZE = 448

# TEST_DIRS = {
#     "PADCHEST_NORMAL": {
#         "path": "data/TB_Chest_Radiography_Database/Normal",
#         "expected": "CXR_FRONTAL"
#     },
#     "LATERALS": {
#         "path": "data/lateral",
#         "expected": "CXR_LAT"
#     }
# }

TEST_DIRS = {
    # "CheXPert_Valid_Frontal": {
    #     "path": "data/CheXpert-v1/valid/frontal",
    #     "expected": "CXR_FRONTAL"
    # },
    # "CheXPert_Valid_Lateral": {
    #     "path": "data/CheXpert-v1/valid/lateral",
    #     "expected": "CXR_LATERAL"
    # },
    "Other": {
        "path": "data/other",
        "expected": "OTHER"
    }

}

LABELS = ["CXR_FRONTAL", "CXR_LATERAL", "OTHER"]

console = Console()

def preprocess_image_letterbox(image_path):
    """
    Reads image and applies aspect-ratio preserving resize (Letterbox)
    matching the training pipeline exactly.
    """
    try:
        # 1. Read Image (cv2 reads as BGR)
        pixel_array = cv2.imread(image_path)
        if pixel_array is None: return None
        
        # Convert BGR to RGB (Crucial for MedSigLIP)
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)

        # 2. Aspect-Ratio Preserving Resize ("Letterbox")
        h, w = pixel_array.shape[:2]
        scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use high-quality interpolation
        resized = cv2.resize(pixel_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate Padding
        delta_w = TARGET_SIZE - new_w
        delta_h = TARGET_SIZE - new_h
        
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        # Add Black Border
        final_image = cv2.copyMakeBorder(
            resized, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0] # Black padding
        )
        # --- YOUR CODE END ---

        # 3. Normalization (Matches Training/JSON)
        # Scale 0-255 -> 0-1
        tensor = torch.tensor(final_image, dtype=torch.float32) / 255.0
        
        # Transpose (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
        
        # Normalize (x - 0.5) / 0.5 -> Range -1 to 1
        tensor = (tensor - 0.5) / 0.5
        
        return tensor.unsqueeze(0).to(DEVICE)
        
    except Exception as e:
        console.print(f"[red]Error processing {image_path}: {e}[/red]")
        return None

def main():
    console.print(f"[bold cyan]🧪 Production Classifier Test v2 (Letterbox) ({DEVICE})[/bold cyan]")

    # 1. Load Models
    try:
        model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
        clf = joblib.load(CLASSIFIER_PATH)
        console.print("[green]✅ Models Loaded Successfully[/green]")
    except Exception as e:
        console.print(f"[bold red]Failed to load models:[/bold red] {e}")
        return

    results = []

    # 2. Run Inference
    for name, info in TEST_DIRS.items():
        folder = info["path"]
        expected = info["expected"]
        
        if not os.path.exists(folder):
            console.print(f"[yellow]Skipping {name}: Folder not found[/yellow]")
            continue
            
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files: continue

        # Limit PadChest for speed if needed
        if name == "PADCHEST_NORMAL": files = files[:50]

        for f in track(files, description=f"Testing {name} ({expected})..."):
            path = os.path.join(folder, f)
            
            # A. Get Features using Letterbox Preprocessing
            pixel_values = preprocess_image_letterbox(path)
            if pixel_values is None: continue
            
            with torch.no_grad():
                outputs = model.vision_model(pixel_values=pixel_values)
                emb = outputs.pooler_output.cpu().numpy().flatten()
            
            # B. Predict
            pred_idx = clf.predict([emb])[0]
            probs = clf.predict_proba([emb])[0]
            
            pred_label = LABELS[pred_idx]
            confidence = probs[pred_idx]
            is_correct = (pred_label == expected)
            
            results.append({
                "file": f,
                "source": name,
                "expected": expected,
                "predicted": pred_label,
                "confidence": confidence,
                "is_correct": is_correct
            })

    # 3. Report
    table = Table(title="Production Test Results (Letterbox)")
    table.add_column("Source", style="cyan")
    table.add_column("File", style="dim")
    table.add_column("Prediction", style="bold")
    table.add_column("Conf", justify="right")
    table.add_column("Result", justify="center")

    correct_count = 0
    total_count = 0
    stats = {name: {"total": 0, "correct": 0} for name in TEST_DIRS}

    for r in results:
        total_count += 1
        stats[r["source"]]["total"] += 1
        
        if r["is_correct"]: 
            correct_count += 1
            stats[r["source"]]["correct"] += 1

        # Show ALL failures, and a few successes
        show_row = not r["is_correct"]
        if r["is_correct"] and stats[r["source"]]["correct"] <= 9:
            show_row = True
            
        if show_row:
            color = "green" if r["is_correct"] else "bold red"
            table.add_row(
                r["source"],
                r["file"][:30],
                r["predicted"],
                f"{r['confidence']:.1%}",
                f"[{color}]{'PASS' if r['is_correct'] else 'FAIL'}[/{color}]"
            )

    console.print(table)
    
    console.print("\n[bold]Summary per Dataset:[/bold]")
    for name, data in stats.items():
        if data["total"] > 0:
            acc = data["correct"] / data["total"]
            color = "green" if acc > 0.95 else ("red" if acc < 0.8 else "yellow")
            console.print(f"   {name}: [{color}]{data['correct']}/{data['total']} ({acc:.1%})[/{color}]")

    console.print(f"\n[bold]Overall Accuracy: {correct_count}/{total_count} ({correct_count/total_count:.1%})[/bold]")

if __name__ == "__main__":
    main()