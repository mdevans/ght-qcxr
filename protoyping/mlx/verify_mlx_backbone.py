import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoModel
from mlx.utils import tree_unflatten
import json

# CONFIG
PYTORCH_PATH = "./medsiglip-vision-v1"
MLX_PATH = "./medsiglip-vision-mlx"

print("🧪 VERIFYING BACKBONE (Last Hidden State)")
print("=========================================")

# --- 1. INPUT ---
input_np = np.zeros((1, 3, 448, 448), dtype=np.float32)

# --- 2. PYTORCH (Reference) ---
print("🔥 Running PyTorch...")
pt_model = AutoModel.from_pretrained(PYTORCH_PATH).float().eval()
with torch.no_grad():
    pt_out = pt_model(pixel_values=torch.tensor(input_np))
    # Grab the sequence of tokens, NOT the pooled vector
    pt_last_hidden = pt_out.last_hidden_state.numpy()

print(f"   PyTorch Shape: {pt_last_hidden.shape}")

# --- 3. MLX (Candidate) ---
print("🍏 Running MLX...")

# Same Architecture as before
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
        x = self.post_layernorm(x)
        return x # Return sequence (B, 1024, 1152)

with open(f"{MLX_PATH}/config.json", "r") as f:
    config = json.load(f)

model = VisionTransformer(config)
weights = mx.load(f"{MLX_PATH}/model.safetensors")
model.update(tree_unflatten(list(weights.items())))

# Run
mlx_input = mx.array(input_np.transpose(0, 2, 3, 1))
mlx_last_hidden = np.array(model(mlx_input))

print(f"   MLX Shape:     {mlx_last_hidden.shape}")

# --- 4. COMPARE ---
print("\n📊 COMPARISON")

# Compare mean of first token to be safe
pt_flat = pt_last_hidden.flatten()
mlx_flat = mlx_last_hidden.flatten()

dot = np.dot(pt_flat, mlx_flat)
norm_a = np.linalg.norm(pt_flat)
norm_b = np.linalg.norm(mlx_flat)
similarity = dot / (norm_a * norm_b)

print(f"   Cosine Similarity: {similarity:.6f}")

if similarity > 0.99:
    print("\n✅ MATCH CONFIRMED: Backbone is perfect.")
else:
    print("\n❌ MISMATCH: Weights are definitely wrong.")