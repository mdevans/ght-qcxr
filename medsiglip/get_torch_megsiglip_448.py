from transformers import AutoModel, AutoProcessor

# 1. Define Model ID and Local Folder
MODEL_ID = "google/medsiglip-448"
LOCAL_PATH = "./medsiglip_google_448"

print(f"⬇️  Downloading {MODEL_ID}...")

# 2. Download Model & Processor
# trust_remote_code=True is required for SigLIP architecture
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 3. Save to Local Directory
print(f"💾 Saving to {LOCAL_PATH}...")
model.save_pretrained(LOCAL_PATH)
processor.save_pretrained(LOCAL_PATH)

print("✅ Download Complete.")