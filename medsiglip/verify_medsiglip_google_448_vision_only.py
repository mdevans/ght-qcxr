import time
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

# CONFIG
MODEL_PATH = "medsiglip/medsiglip-vision-v1"
# Force use of M4 GPU (Metal Performance Shaders)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"🚀 Device Selected: {DEVICE.upper()}")

# --- STEP 1: LOAD MODEL ---
print(f"🔎 Loading model from {MODEL_PATH}...")
t0 = time.perf_counter()

model = AutoModel.from_pretrained(MODEL_PATH)
model.to(DEVICE) # Move model to M4 GPU
model.eval()     # Set to evaluation mode (optimization)

t1 = time.perf_counter()
load_time = t1 - t0
print(f"✅ Model Loaded in: {load_time:.4f} seconds")

# Verify Precision
dtype = model.vision_model.embeddings.patch_embedding.weight.dtype
print(f"   Precision: {dtype}")
print(f"   Params:    {model.num_parameters()/1e6:.1f} M")

# --- STEP 2: PREPARE INPUT ---
print("\n🖼️  Preparing Input...")
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
image = Image.new('RGB', (448, 448), color='black')

t2 = time.perf_counter()
# Note: Moving inputs to device is critical for speed
inputs = processor(images=image, return_tensors="pt").to(DEVICE)
t3 = time.perf_counter()
print(f"   Preprocessing: {t3 - t2:.4f} seconds")

# --- STEP 3: RUN INFERENCE ---
print(f"\n🧪 Running Inference on {DEVICE.upper()}...")

# Warmup run (optional, compiles the graph on MPS)
# with torch.no_grad():
#     _ = model(**inputs)

t4 = time.perf_counter()

with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.pooler_output

# Sync MPS to get accurate timing (MPS is asynchronous)
if DEVICE == "mps":
    torch.mps.synchronize()

t5 = time.perf_counter()
inference_time = t5 - t4

print(f"✅ Inference Complete in: {inference_time:.4f} seconds")
print(f"   Output Shape: {embedding.shape}")

