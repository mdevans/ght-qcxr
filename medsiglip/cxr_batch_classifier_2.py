import os
import torch
import numpy as np
import joblib
from transformers import AutoModel
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from rich.console import Console
from rich.table import Table
from rich.progress import track

# --- CONFIG ---
MODEL_PATH = "medsiglip/medsiglip-google-448"
OUTPUT_MODEL = "models/cxr_classifier_production.joblib"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)

# 3-Class System (Robust)
# 0 = Frontal (AP + PA)
# 1 = Lateral
# 2 = Other
DIRS = {
    "AP":    {"path": "data/AP_CXR_448",  "label": 0, "class_name": "CXR_FRONTAL"},
    "PA":    {"path": "data/PA_CXR_448",  "label": 0, "class_name": "CXR_FRONTAL"},
    "LAT":   {"path": "data/LAT_CXR_448", "label": 1, "class_name": "CXR_LAT"},
    "OTHER": {"path": "data/other",       "label": 2, "class_name": "OTHER"}
}

CLASS_NAMES = ["CXR_FRONTAL", "CXR_LAT", "OTHER"]

console = Console()

def get_embedding(model, image_path):
    """
    Extracts the 512-dim feature vector using the VALIDATED manual preprocessing.
    """
    try:
        # 1. Load & Resize (Strict 448x448)
        img = Image.open(image_path).convert("RGB")
        if img.size != (448, 448):
            img = img.resize((448, 448), resample=Image.Resampling.LANCZOS)
        
        # 2. Manual Preprocessing (Matches JSON Config)
        # Scale 0-255 -> 0-1
        arr = np.array(img).transpose(2, 0, 1)
        tensor = torch.tensor(arr, dtype=torch.float32) / 255.0
        
        # Normalize (x - 0.5) / 0.5 -> Range -1 to 1
        tensor = (tensor - 0.5) / 0.5
        
        # 3. Extract Features
        pixel_values = tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # Use vision_model directly to get raw pooled output (before projection)
            # This contains the richest information for training a classifier.
            outputs = model.vision_model(pixel_values=pixel_values)
            embedding = outputs.pooler_output.cpu().numpy().flatten()
            
        return embedding
    except Exception as e:
        console.print(f"[red]Error processing {image_path}: {e}[/red]")
        return None

def main():
    console.print(f"[bold cyan]🏭 Building Production Classifier ({DEVICE})[/bold cyan]")
    
    # 1. Load Feature Extractor
    console.print("[dim]Loading MedSigLIP model...[/dim]")
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    
    X = []
    y = []
    sources = [] # Keep track of source folder for debugging
    
    # 2. Extract Embeddings
    console.print("\n[bold]1. Extracting Features...[/bold]")
    
    for key, info in DIRS.items():
        if not os.path.exists(info["path"]):
            console.print(f"[yellow]Skipping {key}: Path not found ({info['path']})[/yellow]")
            continue
            
        files = [f for f in os.listdir(info["path"]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files: 
            console.print(f"[yellow]Skipping {key}: No images found[/yellow]")
            continue

        # Debug: Show start of each folder
        console.print(f"   Processing [cyan]{key}[/cyan] ({len(files)} images)...")

        for f in track(files, description=f"   -> {key}"):
            path = os.path.join(info["path"], f)
            emb = get_embedding(model, path)
            if emb is not None:
                X.append(emb)
                y.append(info["label"])
                sources.append(key)
    
    X = np.array(X)
    y = np.array(y)
    
    console.print(f"[green]✓ Data Collection Complete.[/green] Total Samples: {len(X)}")
    console.print(f"   Shape: {X.shape} (Samples, Features)")

    # 3. Train Classifier
    console.print("\n[bold]2. Training Logistic Regression...[/bold]")
    
    # Stratify ensures we split even small classes (like OTHER) fairly
    # random_state=42 ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    console.print(f"   Training Set: {len(X_train)} samples")
    console.print(f"   Test Set:     {len(X_test)} samples")
    
    # C=1.0 is standard regularization. 
    # class_weight='balanced' automatically handles if you have few 'Other' images vs many CXRs.
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    # 4. Detailed Validation
    console.print("\n[bold]3. Validation Results[/bold]")
    
    acc = clf.score(X_test, y_test)
    console.print(f"   [bold green]Overall Accuracy: {acc:.2%}[/bold green]")
    
    y_pred = clf.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    console.print("\n[bold]Confusion Matrix:[/bold]")
    
    cm_table = Table(show_header=True, header_style="bold magenta")
    cm_table.add_column("True \\ Pred", style="dim")
    for name in CLASS_NAMES:
        cm_table.add_column(name, justify="right")
    
    for i, row in enumerate(cm):
        row_str = [str(x) for x in row]
        cm_table.add_row(CLASS_NAMES[i], *row_str)
        
    console.print(cm_table)
    
    # Detailed Report
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
    console.print(f"\n[dim] {report} [/dim]")
    
    # 5. Save
    console.print(f"\n[bold]4. Saving Model...[/bold]")
    joblib.dump(clf, OUTPUT_MODEL)
    console.print(f"   [green]Saved to: {OUTPUT_MODEL}[/green]")
    console.print("[dim]   (Size: ~10-50KB)[/dim]")

if __name__ == "__main__":
    main()


"""
michaelevans@Michaels-Mac-mini ght-qcxr % uv run medsiglip/cxr_batch_classifier_2.py
🏭 Building Production Classifier (mps)
Loading MedSigLIP model...
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 888/888 [00:00<00:00, 3932.04it/s, Materializing param=vision_model.post_layernorm.weight]

1. Extracting Features...
   Processing AP (246 images)...
   -> AP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:59
   Processing PA (160 images)...
   -> PA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:19
   Processing LAT (159 images)...
   -> LAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:19
   Processing OTHER (81 images)...
   -> OTHER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:39
✓ Data Collection Complete. Total Samples: 646
   Shape: (646, 1152) (Samples, Features)

2. Training Logistic Regression...
   Training Set: 516 samples
   Test Set:     130 samples

3. Validation Results
   Overall Accuracy: 100.00%

Confusion Matrix:
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ True \ Pred ┃ CXR_FRONTAL ┃ CXR_LAT ┃ OTHER ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ CXR_FRONTAL │          82 │       0 │     0 │
│ CXR_LAT     │           0 │      32 │     0 │
│ OTHER       │           0 │       0 │    16 │
└─────────────┴─────────────┴─────────┴───────┘

               precision    recall  f1-score   support

 CXR_FRONTAL       1.00      1.00      1.00        82
     CXR_LAT       1.00      1.00      1.00        32
       OTHER       1.00      1.00      1.00        16

    accuracy                           1.00       130
   macro avg       1.00      1.00      1.00       130
weighted avg       1.00      1.00      1.00       130
 

4. Saving Model...
   Saved to: models/cxr_classifier_production.joblib
   (Size: ~10-50KB)
"""