import os
import torch
import numpy as np
import joblib
import cv2
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from rich.console import Console
from rich.table import Table
from rich.progress import track

# --- CONFIG ---
MODEL_PATH = "medsiglip/medsiglip-google-448"
OUTPUT_MODEL = "models/cxr_projection_classifier.joblib"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TARGET_SIZE = 448

# We only care about these two now
DIRS = {
    "AP": {"path": "data/AP_CXR_448", "label": 0, "name": "AP"},
    "PA": {"path": "data/PA_CXR_448", "label": 1, "name": "PA"}
}
CLASS_NAMES = ["AP", "PA"]

console = Console()

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

        # 3. Normalize & Extract
        tensor = torch.from_numpy(final_img).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        pixel_values = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.vision_model(pixel_values=pixel_values)
            embedding = outputs.pooler_output.cpu().numpy().flatten()
            
        return embedding
    except Exception as e:
        return None

def main():
    console.print(f"[bold cyan]🎯 Training AP vs PA Specialist ({DEVICE})[/bold cyan]")
    
    # 1. Load Feature Extractor
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    
    X = []
    y = []
    
    # 2. Extract Features
    console.print("\n[bold]1. Extracting Features (1152 dims)...[/bold]")
    for key, info in DIRS.items():
        if not os.path.exists(info["path"]): continue
            
        files = [f for f in os.listdir(info["path"]) if f.lower().endswith(('.png', '.jpg'))]
        
        for f in track(files, description=f"   Processing {key}"):
            emb = get_embedding(model, os.path.join(info["path"], f))
            if emb is not None:
                X.append(emb)
                y.append(info["label"])
    
    X = np.array(X)
    y = np.array(y)
    
    console.print(f"[green]✓ Data Collected.[/green] Shape: {X.shape}")

    # 3. Train Classifier
    console.print("\n[bold]2. Training Logistic Regression...[/bold]")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # C=0.5: Slightly stronger regularization because features are subtle and we have fewer samples
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    # 4. Results
    acc = clf.score(X_test, y_test)
    console.print(f"\n[bold green]Test Accuracy: {acc:.2%}[/bold green]")
    
    y_pred = clf.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    table = Table(title="Confusion Matrix", show_header=True)
    table.add_column("True \\ Pred")
    table.add_column("AP")
    table.add_column("PA")
    
    table.add_row("AP", str(cm[0][0]), str(cm[0][1]))
    table.add_row("PA", str(cm[1][0]), str(cm[1][1]))
    console.print(table)
    
    # 5. Save
    joblib.dump(clf, OUTPUT_MODEL)
    console.print(f"\n[bold]Saved to {OUTPUT_MODEL}[/bold]")

if __name__ == "__main__":
    main()