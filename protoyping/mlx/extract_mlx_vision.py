import mlx.core as mx
import json
import glob
import os

# CONFIG
SOURCE_MODEL = "mlx-community/medsiglip-448"
TARGET_DIR = "./medsiglip-vision-mlx"

print(f"🚀 Starting MLX Surgery (V3 - No Transpose)...")

# 1. FIND SOURCE
try:
    if os.path.exists(SOURCE_MODEL):
        base_path = SOURCE_MODEL
    else:
        home = os.path.expanduser("~")
        cache_path = f"{home}/.cache/huggingface/hub/models--mlx-community--medsiglip-448/snapshots"
        snapshots = glob.glob(f"{cache_path}/*")
        if not snapshots: raise FileNotFoundError("Model not found.")
        base_path = snapshots[0]
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# 2. LOAD WEIGHTS
print("⏳ Loading full model weights...")
weights = mx.load(f"{base_path}/model.safetensors")

# 3. SURGERY loop
clean_weights = {}
print("✂️  Copying weights (Filtering only)...")

for k, v in weights.items():
    # A. REMOVE UNUSED
    if "text_model" in k: continue
    if "vision_model.head" in k: continue
    if "logit" in k: continue
    
    # B. RENAME KEYS (Normalize)
    new_k = k.replace("vision_model.", "").replace("encoder.layers.", "blocks.")
    
    # Flatten Embeddings for our simple class
    if "embeddings.patch_embedding.weight" in new_k:
        new_k = "embeddings.weight"
    elif "embeddings.patch_embedding.bias" in new_k:
        new_k = "embeddings.bias"
    elif "embeddings.position_embedding.weight" in new_k:
        new_k = "position_embedding.weight"
    
    # C. COPY AS-IS (Force Float16, but NO TRANSPOSE)
    clean_weights[new_k] = v.astype(mx.float16)

# 4. SAVE CONFIG
os.makedirs(TARGET_DIR, exist_ok=True)
with open(f"{base_path}/config.json", "r") as f:
    full_config = json.load(f)
    vision_config = full_config["vision_config"]

with open(f"{TARGET_DIR}/config.json", "w") as f:
    json.dump(vision_config, f, indent=2)

# 5. SAVE WEIGHTS
print(f"💾 Saving to {TARGET_DIR}/model.safetensors...")
mx.save_safetensors(f"{TARGET_DIR}/model.safetensors", clean_weights)

print("-" * 30)
print("✅ EXTRACTION COMPLETE")