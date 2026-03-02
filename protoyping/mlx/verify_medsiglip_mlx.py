import os
import mlx.core as mx
from mlx_vlm import load

MODEL_REPO = "mlx-community/medsiglip-448"
LOCAL_DIR = "medsiglip-448"

# Load and Verify
print("Loading model to check precision...")
model, processor = load(LOCAL_DIR, tokenizer_config={"trust_remote_code": True, "use_fast": True})

if model is None:
    raise RuntimeError(f"Failed to load model from {LOCAL_DIR}. Check if the path is correct and the model files are present.")

# Check the data type of the first layer
first_weight = model.vision_model.embeddings.patch_embedding.weight
precision = first_weight.dtype

print(f"\n--- Verification Report ---")
print(f"✅ Input Resolution: {processor.image_processor.size['height']}x{processor.image_processor.size['width']}")
print(f"✅ Model Precision:  {precision} (Should be bfloat16 or float16)")
print(f"✅ Model Location:   {os.path.abspath(LOCAL_DIR)}")