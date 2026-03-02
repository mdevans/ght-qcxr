import multiprocessing as mp
# CRITICAL: This must be the very first line to prevent the hang on macOS
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch
import joblib
import cv2
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from rich.progress import Progress
from rich.console import Console

# --- CONFIG ---
MODEL_PATH = "medsiglip/medsiglip-google-448"
CLASSIFIER_PATH = "models/cxr_classifier_production.joblib"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# SPEED SETTINGS
BATCH_SIZE = 64        # Big batches for GPU
NUM_WORKERS = 4        # Use 4 CPU cores (Now safe with 'spawn')
PREFETCH_FACTOR = 2    # Buffer batches

console = Console()

class CXRDataset(Dataset):
    def __init__(self, file_paths, target_size=448):
        self.file_paths = file_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            # Prevent OpenCV from fighting with PyTorch workers
            cv2.setNumThreads(0)
            
            # 1. Read
            img = cv2.imread(path)
            if img is None: return torch.zeros((3, self.target_size, self.target_size)), "ERROR"
            
            # 2. Check Dimensions
            h, w = img.shape[:2]

            # --- FAST PATH (99% of your data) ---
            if h == w:
                # Image is square (512x512). Just shrink to 448x448.
                # No padding, no math, no black bars needed.
                final_img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            
            # --- SAFETY PATH (For those 7 odd laterals) ---
            else:
                scale = min(self.target_size / w, self.target_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                delta_w = self.target_size - new_w
                delta_h = self.target_size - new_h
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                
                final_img = cv2.copyMakeBorder(
                    resized, top, bottom, left, right, 
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

            # 3. Normalize & Convert (Float16)
            # BGR -> RGB is implicitly handled if we trained on RGB. 
            # Note: cv2 reads BGR. We should convert.
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            
            tensor = torch.from_numpy(final_img).half() / 255.0 
            tensor = (tensor - 0.5) / 0.5
            tensor = tensor.permute(2, 0, 1) 
            
            return tensor, path
            
        except Exception:
            return torch.zeros((3, self.target_size, self.target_size)), "ERROR"

def main():
    console.print(f"[bold cyan]🚀 Turbo Square Classifier ({DEVICE})[/bold cyan]")
    
    # 1. Load Models (Float16)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).half().eval()
    clf = joblib.load(CLASSIFIER_PATH)
    labels = ["CXR_FRONTAL", "CXR_LAT", "OTHER"]
    
    # 2. Get Files
    target_dir = "data/TB_Chest_Radiography_Database/Normal"
    all_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.png')]
    # Test on 1000 to measure sustained speed
    all_files = all_files[:1000]
    
    console.print(f"Processing {len(all_files)} images...")

    # 3. Optimized Loader
    dataset = CXRDataset(all_files)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True, 
        pin_memory=True
    )

    results = []
    start_time = time.time()
    
    with Progress() as progress:
        task = progress.add_task("[green]Inferencing...", total=len(all_files))
        
        with torch.no_grad():
            for batch_imgs, batch_paths in loader:
                valid_mask = [p != "ERROR" for p in batch_paths]
                if not any(valid_mask): continue
                
                batch_imgs = batch_imgs[valid_mask].to(DEVICE)
                
                # Vision Model
                outputs = model.vision_model(pixel_values=batch_imgs)
                embeddings = outputs.pooler_output.float().cpu().numpy()
                
                # Classifier
                probs = clf.predict_proba(embeddings)
                preds = np.argmax(probs, axis=1)
                
                # Stats
                progress.update(task, advance=len(batch_imgs))
                results.extend(preds)

    total_time = time.time() - start_time
    fps = len(results) / total_time
    
    console.print(f"\n[bold green]Finished![/bold green]")
    console.print(f"Time: {total_time:.2f}s")
    console.print(f"Speed: [bold cyan]{fps:.2f} images/sec[/bold cyan]")

if __name__ == "__main__":
    main()