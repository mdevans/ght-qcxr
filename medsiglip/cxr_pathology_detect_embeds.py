import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageOps
import numpy as np

# CONFIG
MODEL_PATH = "medsiglip/medsiglip-google-448"
#IMAGE_PATH = "data/cxr/cxr_TB_1.jpg"
IMAGE_PATH = "data/cxr/cxr_normal_0.jpg"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROMPTS = [
    "A chest X-ray showing Tuberculosis",
    "A normal chest X-ray",
    "Non-medical image"
]

def main():
    print(f"Running Cosine Check on {DEVICE}...")

    # 1. Load Model
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # 2. Image Prep (Simple Pipeline)
    raw_img = Image.open(IMAGE_PATH).convert("RGB")
    # Fix contrast and resize
    img = ImageOps.autocontrast(raw_img).resize((448, 448), Image.Resampling.LANCZOS)
    
    # Manual Tensor Construction
    img_array = np.array(img).transpose(2, 0, 1)
    pixel_values = torch.tensor(img_array, dtype=torch.float32) / 255.0
    pixel_values = ((pixel_values - 0.5) / 0.5).unsqueeze(0).to(DEVICE)

    # 3. Text Prep
    text_inputs = tokenizer(PROMPTS, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        # Run the full model once
        outputs = model(pixel_values=pixel_values, **text_inputs)
        
        # EXTRACT TENSORS SAFELY
        # The model output object holds the embeddings in these attributes:
        img_embeds = outputs.image_embeds  # Shape: [1, 1152]
        text_embeds = outputs.text_embeds  # Shape: [3, 1152]
        
        # Normalize (just to be 100% sure they are length 1.0)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # 5. Calculate Raw Cosine Similarity (-1.0 to 1.0)
        # Matrix multiplication: [1, 1152] @ [1152, 3] -> [1, 3]
        cosine_sim = torch.matmul(img_embeds, text_embeds.t())
        
        # Get the final biased logits for comparison
        logits = outputs.logits_per_image

    # 6. Report
    print(f"\n{'Label':<35} | {'Cosine (Raw)':<12} | {'Logit (Final)':<12}")
    print("-" * 65)
    
    for i, label in enumerate(PROMPTS):
        raw = cosine_sim[0][i].item()   # The pure angle
        logit = logits[0][i].item()     # The scaled + biased output
        print(f"{label:<35} | {raw:+.4f}       | {logit:.2f}")

    print("\nINTERPRETATION:")
    print("Cosine > 0.0 means 'Positive Correlation' (Vector points roughly same direction).")
    print("Logit is negative because SigLIP subtracts a massive bias to prevent False Positives.")

if __name__ == "__main__":
    main()