import time
import mlx.core as mx
from mlx_vlm import load
from PIL import Image
import numpy as np

# CONFIG
# This will auto-download to ~/.cache/huggingface/hub unless local_dir is set
MODEL_REPO = "mlx-community/medsiglip-448"

print(f"🚀 Initializing MLX...")
print(f"⬇️  Downloading/Loading {MODEL_REPO}...")

# 1. LOAD (Auto-downloads if needed)
# The 'trust_remote_code' is not usually needed for MLX community models as they are native
model, processor = load(MODEL_REPO)

print("✅ Model Loaded Successfully on GPU/Metal")

# 2. PREPARE INPUT
print("🖼️  Creating Dummy Input (448x448)...")
image = Image.new('RGB', (448, 448), color='black')
inputs = processor(images=image, return_tensors="np")
pixel_values = mx.array(inputs['pixel_values'])

# 3. WARMUP
print("🔥 Warming up (Compiling Shaders)...")
# SigLIP's vision tower is usually accessed via .vision_model
# We run one pass to compile the Metal graph
out = model.vision_model(pixel_values)
mx.eval(out) 

print("✅ Warmup Complete. Starting Benchmark...")

# 4. BENCHMARK
latencies = []
NUM_LOOPS = 50

print(f"🏃 Running {NUM_LOOPS} iterations...")
for i in range(NUM_LOOPS):
    t0 = time.perf_counter()
    
    out = model.vision_model(pixel_values)
    mx.eval(out) # Force execution
    
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

# 5. REPORT
avg_latency = np.mean(latencies)
throughput = 1000 / avg_latency

print("\n" + "="*40)
print(f"🍏 MLX PERFORMANCE REPORT")
print("="*40)
print(f"Avg Latency:    {avg_latency:.2f} ms")
print(f"Throughput:     {throughput:.1f} images/sec")
print("="*40)

if avg_latency < 50:
    print("⚡ Conclusion: REAL-TIME READY")
else:
    print("⚠️ Conclusion: Slow (Check background tasks)")