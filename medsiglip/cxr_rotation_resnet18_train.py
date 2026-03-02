import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from rich.console import Console
from rich.table import Table

# --- CONFIG ---
DATA_DIR = "data/rotation"
OUTPUT_MODEL = "models/cxr_rotation_resnet18.pth"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4

DIRS = {
    "Centered": {"path": "data/rotation/centered", "label": 0},
    "Rotated": {"path": "data/rotation/rotated", "label": 1}
}
CLASS_NAMES = ["Centered", "Rotated"]

console = Console()

# --- CUSTOM DATASET ---
class CXRDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        # Convert to RGB because ResNet expects 3 channels
        image = Image.open(img_path).convert("RGB") 
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def main():
    console.print(f"[bold cyan]🎯 Training ResNet18 Rotation Specialist ({DEVICE})[/bold cyan]")
    
    # 1. Load File Paths
    all_paths = []
    all_labels = []
    
    for class_name, info in DIRS.items():
        if not os.path.exists(info["path"]):
            console.print(f"[red]Warning: Path {info['path']} not found.[/red]")
            continue
            
        files = [f for f in os.listdir(info["path"]) if f.lower().endswith(('.png', '.jpg'))]
        for f in files:
            all_paths.append(os.path.join(info["path"], f))
            all_labels.append(info["label"])
            
    if not all_paths:
        console.print("[bold red]No images found. Exiting.[/bold red]")
        return
        
    console.print(f"[green]✓ Found {len(all_paths)} total images.[/green]")

    # 2. Stratified Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # Calculate class weights for PyTorch CrossEntropyLoss
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    # Weight formula: total_samples / (num_classes * count)
    weights = total_samples / (2.0 * class_counts)
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    
    console.print(f"Training set: {len(X_train)} images | Validation set: {len(X_val)} images")

    # 3. Data Transforms
    # NO ROTATION OR FLIPS ALLOWED! We only adjust pixels, not geometry.
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CXRDataset(X_train, y_train, transform=train_transforms)
    val_dataset = CXRDataset(X_val, y_val, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    console.print("\n[bold]Loading Pre-trained ResNet18...[/bold]")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 5. Training Loop
    console.print("\n[bold]Starting Training...[/bold]")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total
        console.print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.2%}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_MODEL)

    console.print(f"\n[bold green]✓ Training Complete. Best Val Accuracy: {best_acc:.2%}[/bold green]")

    # 6. Final Evaluation with Raw Probabilities
    console.print("\n[bold cyan]--- Evaluating Validation Probabilities ---[/bold cyan]")
    model.load_state_dict(torch.load(OUTPUT_MODEL, weights_only=True))
    model.eval()
    
    # Create a rich table to display the raw probabilities
    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Image Index", justify="right")
    results_table.add_column("True Label")
    results_table.add_column("Prob (Rotated)", justify="right")
    results_table.add_column("Prob (Centered)", justify="right")
    
    all_preds = []
    all_labels = []
    
    # We will use 0.50 just as a baseline for color-coding the output
    BASELINE_THRESHOLD = 0.50 
    
    img_idx = 1
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            # Convert raw logits to probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            for i in range(len(labels)):
                true_label = labels[i].item()
                prob_centered = probabilities[i, 0].item()
                prob_rotated = probabilities[i, 1].item()
                
                # Baseline prediction
                predicted = 1 if prob_rotated >= BASELINE_THRESHOLD else 0
                
                all_preds.append(predicted)
                all_labels.append(true_label)
                
                # Formatting for the table
                true_class = "Rotated" if true_label == 1 else "Centered"
                color = "green" if predicted == true_label else "red"
                
                results_table.add_row(
                    str(img_idx),
                    true_class,
                    f"[{color}]{prob_rotated:.4f}[/{color}]",
                    f"{prob_centered:.4f}"
                )
                img_idx += 1
                
    console.print(results_table)
    

if __name__ == "__main__":
    main()