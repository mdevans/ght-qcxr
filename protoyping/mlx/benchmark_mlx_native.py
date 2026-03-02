import time
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# CONFIG
MODEL_PATH = "./medsiglip-vision-mlx"

print(f"🚀 Initializing Benchmark (Native MLX)...")

# --- 1. ARCHITECTURE ---
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.self_attn = Attention(config["hidden_size"], config["num_attention_heads"])
        self.layer_norm2 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.mlp = MLP(config["hidden_size"], config["intermediate_size"])

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Simplified structure matches our extraction script V2
        self.embeddings = nn.Conv2d(
            in_channels=3, out_channels=config["hidden_size"],
            kernel_size=config["patch_size"], stride=config["patch_size"]
        )
        num_patches = (config["image_size"] // config["patch_size"]) ** 2
        self.position_embedding = nn.Embedding(num_patches, config["hidden_size"])
        
        self.blocks = [EncoderLayer(config) for _ in range(config["num_hidden_layers"])]
        self.post_layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])

    def __call__(self, x):
        x = self.embeddings(x) 
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C) + self.position_embedding.weight
        for block in self.blocks:
            x = block(x)
        return self.post_layernorm(x).mean(axis=1)

# --- 2. LOAD ---
print("⏳ Loading Clean Weights...")
with open(f"{MODEL_PATH}/config.json", "r") as f:
    config = json.load(f)

model = VisionTransformer(config)
weights = mx.load(f"{MODEL_PATH}/model.safetensors")

# This should now work instantly without error
model.update(tree_unflatten(list(weights.items())))
mx.eval(model.parameters())
print("✅ Weights Loaded.")

# --- 3. BENCHMARK ---
print("🔥 Warming up (Compiling)...")
x = mx.random.uniform(shape=(1, 448, 448, 3)).astype(mx.float16)

@mx.compile
def run_model(inputs):
    return model(inputs)

mx.eval(run_model(x))

print("🏃 Running 50 Iterations...")
latencies = []
for _ in range(50):
    t0 = time.perf_counter()
    out = run_model(x)
    mx.eval(out)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

avg = np.mean(latencies)
print("\n" + "="*40)
print(f"🍏 FINAL RESULT")
print("="*40)
print(f"Avg Latency:    {avg:.2f} ms")
print(f"Throughput:     {1000/avg:.1f} fps")
print("="*40)