import os
import shutil
import torch
import numpy as np
import cv2
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.progress import track

# --- CONFIG ---
MODEL_PATH = "medsiglip/medsiglip-google-448"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TARGET_SIZE = 448

DIRS = {
    "Centered": {"path": "data/rotation/centered", "label": 0},
    "Rotated": {"path": "data/rotation/rotated", "label": 1}
}

# Output directories for the failures
FAILURES_DIR = "data/rotation_failures"
FP_DIR = os.path.join(FAILURES_DIR, "false_positives_true_centered_pred_rotated")
FN_DIR = os.path.join(FAILURES_DIR, "false_negatives_true_rotated_pred_centered")

console = Console()

def setup_directories():
    """Create fresh failure directories, clearing old ones if they exist."""
    if os.path.exists(FAILURES_DIR):
        shutil.rmtree(FAILURES_DIR)
    os.makedirs(FP_DIR)
    os.makedirs(FN_DIR)

def get_embedding(model, image_path):
    try:
        # 1. Read & Convert
        img = cv2.imread(image_path)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Aspect-Ratio Preserving Resize (Letterbox)
        h, w = img.shape[:2]
        if h == TARGET_SIZE and w == TARGET_SIZE:
            final_img = img
        else:
            scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            delta_w = TARGET_SIZE - new_w
            delta_h = TARGET_SIZE - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            final_img = cv2.copyMakeBorder(
                resized, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        # 3. Normalize & Extract FULL GLOBAL EMBEDDING
        tensor = torch.from_numpy(final_img).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        pixel_values = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.vision_model(pixel_values=pixel_values)
            # GLOBAL POOLER OUTPUT
            embedding = outputs.pooler_output.cpu().numpy().flatten()
            
        return embedding
    except Exception as e:
        console.print(f"[red]Error processing {image_path}: {e}[/red]")
        return None

def main():
    console.print(f"[bold cyan]🎯 Extracting Global Embedding Failures ({DEVICE})[/bold cyan]")
    
    setup_directories()
    
    # 1. Load Feature Extractor
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    
    X = []
    y = []
    all_paths = []
    
    # 2. Extract Features
    console.print("\n[bold]1. Extracting Features (1152 dims)...[/bold]")
    for key, info in DIRS.items():
        if not os.path.exists(info["path"]): continue
            
        files = [f for f in os.listdir(info["path"]) if f.lower().endswith(('.png', '.jpg'))]
        
        for f in track(files, description=f"   Processing {key}"):
            full_path = os.path.join(info["path"], f)
            emb = get_embedding(model, full_path)
            if emb is not None:
                X.append(emb)
                y.append(info["label"])
                all_paths.append(full_path)
    
    X = np.array(X)
    y = np.array(y)

    # 3. Train Classifier
    console.print("\n[bold]2. Training Logistic Regression & Testing...[/bold]")
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, all_paths, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    # Get raw probabilities for Class 1 (Rotated)
    y_proba = clf.predict_proba(X_test)[:, 1] 
    y_pred = (y_proba >= 0.50).astype(int) # Explicit 50% threshold
    
    # 4. Route and Copy Failures
    console.print("\n[bold]3. Sorting Failures...[/bold]")
    fp_count = 0
    fn_count = 0
    
    for true_label, pred_label, prob, path in zip(y_test, y_pred, y_proba, paths_test):
        # Format the new filename
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        prob_pct = int(prob * 100) # Convert 0.743 to 74
        
        new_filename = f"{name}_{prob_pct}%{ext}"
        
        if true_label == 0 and pred_label == 1:
            # False Positive
            shutil.copy(path, os.path.join(FP_DIR, new_filename))
            fp_count += 1
            
        elif true_label == 1 and pred_label == 0:
            # False Negative
            shutil.copy(path, os.path.join(FN_DIR, new_filename))
            fn_count += 1

    console.print(f"\n[bold green]✓ Done![/bold green]")
    console.print(f"Copied [bold red]{fp_count}[/bold red] False Positives to: {FP_DIR}")
    console.print(f"Copied [bold yellow]{fn_count}[/bold yellow] False Negatives to: {FN_DIR}")

if __name__ == "__main__":
    main()