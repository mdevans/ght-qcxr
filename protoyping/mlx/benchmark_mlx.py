import time
import mlx.core as mx
from mlx_vlm import load, apply_chat_template
from PIL import Image
import numpy as np

# CONFIG
# This matches the model you already have cached
MODEL_PATH = "mlx-community/medsiglip-448" 

print(f"🚀 Loading MLX Model: {MODEL_PATH}...")
# 1. Load Model (Native Apple Silicon weights)
model, processor = load(MODEL_PATH)

# 2. Prepare Dummy Input
print("🖼️  Preparing Input...")
# MLX Processor handles the 448x448 resize/norm automatically
image = Image.new('RGB', (448, 448), color='black')
inputs = processor(images=image, return_tensors="np")

# Convert to MLX Array (puts it on the GPU)
pixel_values = mx.array(inputs['pixel_values'])

# 3. Warmup (Critical for MLX)
print("🔥 Warming up (Compiling Metal Shaders)...")
# Run once to compile
vision_output = model.vision_model(pixel_values)
# Force evaluation (MLX is lazy)
mx.eval(vision_output)

print("✅ Warmup Complete. Starting Benchmark...")

# 4. Run Benchmark
latencies = []
NUM_LOOPS = 50

print(f"🏃 Running {NUM_LOOPS} iterations...")

for i in range(NUM_LOOPS):
    t0 = time.perf_counter()
    
    # Inference
    vision_output = model.vision_model(pixel_values)
    
    # Extract Embedding (Pooler Output)
    # SigLIP typically uses the first token or a probe, depending on implementation.
    # We force eval here to measure calculation time.
    mx.eval(vision_output)
    
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

# 5. Results
avg_latency = np.mean(latencies)
throughput = 1000 / avg_latency

print("\n" + "="*40)
print(f"🍏 MLX PERFORMANCE REPORT")
print("="*40)
print(f"Avg Latency:    {avg_latency:.2f} ms")
print(f"Throughput:     {throughput:.1f} images/sec")
print("="*40)

if avg_latency < 50:
    print("⚡ Status: SUCCESS (Real-time Ready)")
else:
    print("⚠️ Status: Unexpectedly Slow (Check Memory)")