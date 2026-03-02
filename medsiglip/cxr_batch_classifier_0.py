import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.progress import track

# --- CONFIG ---
MODEL_PATH = "medsiglip/medsiglip-google-448"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. PROMPTS (The "Sensors")
PROMPTS = [
    "An anterior-posterior chest x-ray",  # 0 -> Frontal
    "A posterior-anterior chest x-ray",   # 1 -> Frontal
    "A lateral chest x-ray",              # 2 -> Lateral
    "A medical x-ray of a bone or joint", # 3 -> Other
    "A non-medical image",                # 4 -> Other
    "A digital screenshot or document"    # 5 -> Other
]

# 2. LOGIC (Mapping Sensors to Output)
CLASS_MAP = {
    0: "CXR_FRONTAL", 1: "CXR_FRONTAL",
    2: "CXR_LAT",
    3: "OTHER", 4: "OTHER", 5: "OTHER"
}

# 3. DIRECTORIES (Updated to your structure)
DIRS = {
    "AP":    "data/AP_CXR_448",    # Expect CXR_FRONTAL
    "PA":    "data/PA_CXR_448",    # Expect CXR_FRONTAL
    "LAT":   "data/LAT_CXR_448",   # Expect CXR_LAT
    "OTHER": "data/other"          # Expect OTHER
}

console = Console()

def main():
    console.print(f"[bold cyan]🧪 MedSigLIP Batch Test: Fixed Normalization ({DEVICE})[/bold cyan]")

    # 1. Load Model & Processor
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

    # --- CRITICAL FIX: FORCE CORRECT NORMALIZATION ---
    # AutoProcessor often defaults to ImageNet (Mean=0.485). 
    # We must force it to 0.5 to match the JSON config we found.
    try:
        # Access the underlying image processor configuration
        if hasattr(processor, "image_processor"):
            processor.image_processor.image_mean = [0.5, 0.5, 0.5]
            processor.image_processor.image_std = [0.5, 0.5, 0.5]
            processor.image_processor.rescale_factor = 1/255
            processor.image_processor.do_resize = True
            processor.image_processor.size = {"height": 448, "width": 448}
        console.print("[green]✅ Processor successfully patched to Mean/Std [0.5, 0.5, 0.5][/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not patch processor: {e}[/yellow]")

    results = []
    
    # 2. Run Test
    for source, folder in DIRS.items():
        if not os.path.exists(folder):
            console.print(f"[yellow]Skipping {source}: Folder not found ({folder})[/yellow]")
            continue
        
        # Limit CXRs to 20 for speed, check ALL others
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if source in ["AP", "PA", "LAT"]: 
            files = files[:20]

        for f in track(files, description=f"Scanning {source}..."):
            path = os.path.join(folder, f)
            try:
                image = Image.open(path).convert("RGB")
                
                # AutoProcessor now uses the patched values
                inputs = processor(
                    text=PROMPTS,
                    images=image,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits_per_image[0]
                    
                    # --- CRITICAL FIX: SOFTMAX ---
                    # Softmax forces a relative choice, solving the "0% confidence" issue
                    probs = F.softmax(logits, dim=0).cpu().numpy()

                # Logic: Winner -> Class
                winner_idx = np.argmax(probs)
                pred_class = CLASS_MAP[winner_idx]
                
                # Aggregate Confidence (Sum probabilities of all prompts for the predicted class)
                # e.g. 0.35 AP + 0.35 PA = 0.70 Frontal Confidence
                confidence = sum(probs[i] for i, cls in CLASS_MAP.items() if cls == pred_class)
                
                # Verification
                expected = "OTHER"
                if source in ["AP", "PA"]: expected = "CXR_FRONTAL"
                elif source == "LAT": expected = "CXR_LAT"
                
                results.append({
                    "file": f, "source": source, "pred": pred_class,
                    "conf": confidence, "correct": (pred_class == expected)
                })

            except Exception as e:
                console.print(f"[red]Error on {f}: {e}[/red]")

    # 3. Report
    table = Table(title="Corrected AutoProcessor Results")
    table.add_column("Source", style="cyan")
    table.add_column("Prediction", style="bold")
    table.add_column("Conf", justify="right")
    table.add_column("Status", justify="center")

    correct = 0
    total = 0

    for r in results:
        total += 1
        if r["correct"]: correct += 1
        
        # Show Failures OR Other OR Sample Successes (to keep table readable)
        if not r["correct"] or r["source"] == "OTHER" or (total % 5 == 0 and total < 50):
            color = "green" if r["correct"] else "bold red"
            table.add_row(
                f"{r['source']}/{r['file'][:8]}",
                r['pred'],
                f"{r['conf']:.1%}",
                f"[{color}]{'PASS' if r['correct'] else 'FAIL'}[/{color}]"
            )

    console.print(table)
    console.print(f"\n[bold]Overall Accuracy: {correct}/{total} ({correct/total:.1%})[/bold]")
    
    # Per-Class Stats
    for cls in ["CXR_FRONTAL", "CXR_LAT", "OTHER"]:
        subset = [r for r in results if 
                  (r["source"] in ["AP", "PA"] and cls=="CXR_FRONTAL") or
                  (r["source"]=="LAT" and cls=="CXR_LAT") or
                  (r["source"]=="OTHER" and cls=="OTHER")]
        
        if subset:
            sub_correct = sum(1 for r in subset if r["correct"])
            console.print(f"   {cls}: [green]{sub_correct}/{len(subset)} ({sub_correct/len(subset):.1%})[/green]")

if __name__ == "__main__":
    main()