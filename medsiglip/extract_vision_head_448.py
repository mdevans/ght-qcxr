import os
import torch
from transformers import AutoModel, AutoProcessor, SiglipVisionModel

# CONFIGURATION
INPUT_DIR = "./medsiglip-google-448" 
OUTPUT_DIR = "./medsiglip-vision-v1"

print(f"🚀 Starting Optimization (Structure Fix)...")
print(f"   Source: {INPUT_DIR}")
print(f"   Target: {OUTPUT_DIR}")

# 1. LOAD FULL MODEL (Float16)
print("⏳ Loading full model source...")
# We load the full dual-encoder model (Text + Vision)
full_model = AutoModel.from_pretrained(INPUT_DIR, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(INPUT_DIR)

# 2. CREATE THE PROPER CONTAINER
print("🏗️  Constructing standalone Vision Chassis...")
# We explicitly create the class that AutoModel expects to see
vision_config = full_model.config.vision_config
vision_head: SiglipVisionModel = SiglipVisionModel(config=vision_config)

# Ensure the new container is also Float16
vision_head = vision_head.to(dtype=torch.float16) # type: ignore

# 3. SURGERY: TRANSPLANT WEIGHTS
print("✂️  Transplanting Vision Tower...")
# We copy the weights from the full model's vision engine (vision_model)
# into the new container's vision engine (vision_model).
# Since we are mapping sub-module to sub-module, the keys match perfectly (1:1).
vision_head.vision_model.load_state_dict(full_model.vision_model.state_dict())

# 4. SAVE LIGHTWEIGHT VERSION
print(f"💾 Saving to {OUTPUT_DIR}...")
# Now we save the CONTAINER, which preserves the correct architecture hierarchy
vision_head.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# 5. FINAL REPORT
original_size = sum(os.path.getsize(os.path.join(INPUT_DIR, f)) for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f)))
new_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f)))

print("-" * 30)
print(f"✅ OPTIMIZATION COMPLETE")
print(f"   Original Size: {original_size / 1e9:.2f} GB")
print(f"   New Size:      {new_size / 1e9:.2f} GB")
print(f"   Structure:     CORRECTED (Wrapped in SiglipVisionModel)")
print("-" * 30)