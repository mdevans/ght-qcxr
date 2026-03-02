import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageOps
import numpy as np
import json
from rich.console import Console
from rich.table import Table

# CONFIG
MODEL_PATH = "medsiglip/medsiglip-google-448"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class RIPE_Agent:
    def __init__(self, model_path, device):
        self.device = device
        self.console = Console()
        
        self.console.print(f"[bold green]🤖 Initializing RIPE Agent on {device}...[/bold green]")
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # DEFINING THE "RULER" (The Semantic Pairs)
        # Format: "Category": [("Positive/Pass Label", "Negative/Fail Label")]
        self.criteria = {
            "Rotation": [
                "Spinous processes centered exactly between clavicle heads", 
                "Patient is rotated with asymmetric clavicles"
            ],
            "Inspiration": [
                "Good inspiration with diaphragm below the 9th rib", 
                "Poor inspiration with high diaphragm and crowded lung markings"
            ],
            "Projection": [
                "PA view with scapulae retracted outside lung fields", 
                "AP view with scapulae overlapping the lungs"
            ],
            "Exposure": [
                "Proper exposure with thoracic spine faintly visible behind heart", 
                "Underexposed image, spine not visible behind heart",
                "Overexposed image, lungs look completely black" # Exposure has 3 states!
            ]
        }

    def preprocess(self, image_path):
        """Standardized Pipeline: AutoContrast -> Resize -> Normalize"""
        raw_img = Image.open(image_path).convert("RGB")
        img = ImageOps.autocontrast(raw_img) # Crucial for Exposure check
        img = img.resize((448, 448), Image.Resampling.LANCZOS)
        
        img_array = np.array(img).transpose(2, 0, 1)
        pixel_values = torch.tensor(img_array, dtype=torch.float32) / 255.0
        pixel_values = ((pixel_values - 0.5) / 0.5).unsqueeze(0).to(self.device)
        return pixel_values

    def assess(self, image_path):
        pixel_values = self.preprocess(image_path)
        report = {"filename": image_path, "status": "PASS", "details": {}}
        
        # Flatten prompts for batch processing
        all_prompts = []
        mapping = [] # To track which prompt belongs to which category
        
        for category, prompts in self.criteria.items():
            for p in prompts:
                all_prompts.append(p)
                mapping.append(category)

        # Tokenize & Inference
        text_inputs = self.tokenizer(all_prompts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, **text_inputs)
            # Use SOFTMAX within each category group effectively
            # But here we just get raw logits and compare them manually
            logits = outputs.logits_per_image[0]

        # Process Results per Category
        current_idx = 0
        for category, prompts in self.criteria.items():
            n_options = len(prompts)
            # Get logits for this specific group (e.g., Good Insp vs Bad Insp)
            group_logits = logits[current_idx : current_idx + n_options]
            
            # Softmax ONLY within this group to force a decision
            probs = F.softmax(group_logits, dim=0)
            
            # Who won?
            winner_idx = probs.argmax().item()
            winner_prob = probs[winner_idx].item()
            winner_text = prompts[winner_idx]
            
            # Logic: Index 0 is always the "Good/Pass" prompt in my dictionary
            is_pass = (winner_idx == 0)
            
            # Alert Logic
            status = "OK" if is_pass else "FLAG"
            if not is_pass:
                report["status"] = "REVIEW" # Downgrade overall status
            
            report["details"][category] = {
                "verdict": status,
                "confidence": round(winner_prob * 100, 1),
                "description": winner_text
            }
            
            current_idx += n_options

        return report

# --- RUN THE AGENT ---
if __name__ == "__main__":
    agent = RIPE_Agent(MODEL_PATH, DEVICE)
    
    # Test on your image
    result = agent.assess("data/cxr/cxr_normal_0.jpg")
    
    # Pretty Print Report
    console = Console()
    table = Table(title=f"RIPE Assessment: {result['status']}")
    table.add_column("Check", style="cyan")
    table.add_column("Verdict", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Model Finding", style="dim")

    for cat, data in result["details"].items():
        color = "green" if data["verdict"] == "OK" else "red"
        table.add_row(
            cat, 
            f"[{color}]{data['verdict']}[/{color}]", 
            f"{data['confidence']}%", 
            data['description']
        )
    
    console.print(table)