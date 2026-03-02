import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
from rich.console import Console
from rich.table import Table

# CONFIG
MODEL_PATH = "medsiglip/medsiglip-google-448"
IMAGE_PATH = "data/cxr/cxr_TB_1.jpg"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- PROMPT ENSEMBLING ---
# We map 1 "Concept" to multiple "Text Descriptions"
CLASSES = {
    "Chest X-Ray": [
        "A frontal chest X-ray",
        "A radiograph of the chest and lungs",
        "Anterior-Posterior chest X-ray",
        "Medical image of the heart and lungs"
    ],
    "Abdominal X-Ray": [
        "An X-ray of the abdomen",
        "Abdominal radiograph showing bowels",
        "KUB X-ray",
        "Medical image of the stomach"
    ],
    "Limb/Bone X-Ray": [
        "An X-ray of a bone fracture",
        "Radiograph of the hand or foot",
        "Musculoskeletal X-ray",
        "Extremity X-ray"
    ],
    "Other/Noise": [
        "A random noise image",
        "A blurry non-medical photo",
        "A CT scan slice",
        "An MRI scan"
    ]
}

def main():
    console = Console()
    console.print(f"[bold green]🚀 Ensemble Detector initialized on {DEVICE}[/bold green]")

    # 1. Load
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)

    # 2. Process Image
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    # 3. Process Text (Flatten the list first)
    all_texts = []
    class_indices = [] # Keeps track of which text belongs to which class
    
    current_idx = 0
    class_names = list(CLASSES.keys())
    
    for name in class_names:
        prompts = CLASSES[name]
        all_texts.extend(prompts)
        # Store start/end indices for this class
        class_indices.append((current_idx, current_idx + len(prompts)))
        current_idx += len(prompts)

    # 4. Run Inference
    inputs = processor(
        text=all_texts, 
        images=image, 
        padding="max_length", 
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for (1 Image) vs (All Prompts)
        logits = outputs.logits_per_image[0] # Shape: [Total_Prompts]
        
        # 5. Aggregate Scores per Class
        class_scores = []
        for start, end in class_indices:
            # Take the mean logit for all prompts in this class
            # We average the *logits* (raw scores) before softmax for stability
            avg_logit = logits[start:end].mean()
            class_scores.append(avg_logit)
            
        # Convert averaged logits to probability
        class_scores_tensor = torch.tensor(class_scores)
        probs = torch.softmax(class_scores_tensor, dim=0)

    # 6. Report
    table = Table(title="Ensemble Classification Results")
    table.add_column("Class", style="cyan")
    table.add_column("Confidence", justify="right", style="green")
    
    # Sort
    results = sorted(zip(class_names, probs.tolist()), key=lambda x: x[1], reverse=True)
    
    for name, score in results:
        bar = "█" * int(score * 20)
        table.add_row(name, f"{score:.2%}  {bar}")
        
    console.print(table)

if __name__ == "__main__":
    main()