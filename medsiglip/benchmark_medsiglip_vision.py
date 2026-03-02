import time
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image

# CONFIG
MODEL_PATH = "./medsiglip-vision-v1"
BATCH_SIZE = 1 # ViT models scale well, but let's test latency first
NUM_LOOPS = 50

# 1. SETUP M4 GPU
DEVICE = "mps"
print(f"🚀 Device: {DEVICE.upper()} (Apple M4)")

# 2. LOAD MODEL
print("⏳ Loading Model...")
# Ensure model is strictly Float16
model = AutoModel.from_pretrained(MODEL_PATH, dtype=torch.float16).to(DEVICE)
model.eval()

# 3. PREPARE INPUT (THE FIX)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
image = Image.new('RGB', (448, 448), color='black')

# Create inputs
inputs_pt = processor(images=image, return_tensors="pt")

# CRITICAL STEP: Cast input to Float16 manually
# This removes the casting overhead from the inference loop
inputs = {
    k: v.to(DEVICE).to(dtype=torch.float16) 
    for k, v in inputs_pt.items()
}

print(f"   Input Dtype: {inputs['pixel_values'].dtype} (Should be float16)")

# 4. WARMUP
print("🔥 Warming up...")
with torch.no_grad():
    # Run twice to be sure shaders are compiled
    _ = model(**inputs)
    _ = model(**inputs)
    torch.mps.synchronize()

print("✅ Warmup Complete. Starting Benchmark...")

# 5. RUN BENCHMARK
latencies = []

print(f"🏃 Running {NUM_LOOPS} iterations...")

# Use inference_mode (slightly faster than no_grad)
with torch.inference_mode():
    for i in range(NUM_LOOPS):
        t0 = time.perf_counter()
        
        # Inference
        _ = model(**inputs)
        
        torch.mps.synchronize() # Sync is required for accurate timing
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

# 6. RESULTS
avg_latency = np.mean(latencies)
p99_latency = np.percentile(latencies, 99)
throughput = 1000 / avg_latency

print("\n" + "="*40)
print(f"📊 OPTIMIZED M4 REPORT")
print("="*40)
print(f"Avg Latency:    {avg_latency:.2f} ms")
print(f"Throughput:     {throughput:.1f} images/sec")
print("="*40)

