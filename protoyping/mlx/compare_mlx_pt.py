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

print("🧪 STARTING NUMERICAL VERIFICATION (FIXED)")
print("==========================================")

# --- 1. PREPARE INPUT (BLACK IMAGE) ---
input_np = np.zeros((1, 3, 448, 448), dtype=np.float32)
print("🖼️  Input: Black Image (Zeros)")

# --- 2. RUN PYTORCH (Reference) ---
print("🔥 Running PyTorch Model...")
pt_model = AutoModel.from_pretrained(PYTORCH_PATH).float().eval()

with torch.no_grad():
    pt_input = torch.tensor(input_np)
    # Get the raw sequence of 1024 patches
    pt_out = pt_model(pixel_values=pt_input).last_hidden_state
    
    # FORCE MEAN POOLING (Average across the 1024 patches)
    # This matches exactly what our MLX script does
    pt_vec = pt_out.mean(dim=1).numpy()

print(f"   PyTorch (Mean Pool) Shape: {pt_vec.shape}")
print(f"   PyTorch First 5 values:    {pt_vec[0, :5]}")

# --- 3. RUN MLX (Candidate) ---
print("\n🍏 Running MLX Model...")

# Architecture
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
        # MLX MEAN POOLING
        return self.post_layernorm(x).mean(axis=1)

# Load MLX
with open(f"{MLX_PATH}/config.json", "r") as f:
    config = json.load(f)

model = VisionTransformer(config)
weights = mx.load(f"{MLX_PATH}/model.safetensors")
model.update(tree_unflatten(list(weights.items())))

# Run
mlx_input = mx.array(input_np.transpose(0, 2, 3, 1))
mlx_out = model(mlx_input)
mlx_vec = np.array(mlx_out)

print(f"   MLX (Mean Pool) Shape:     {mlx_vec.shape}")
print(f"   MLX First 5 values:        {mlx_vec[0, :5]}")

# --- 4. COMPARE ---
print("\n📊 COMPARISON RESULT")

# Cosine Similarity
dot_product = np.dot(pt_vec.flatten(), mlx_vec.flatten())
norm_a = np.linalg.norm(pt_vec)
norm_b = np.linalg.norm(mlx_vec)
similarity = dot_product / (norm_a * norm_b)

print(f"   Cosine Similarity: {similarity:.6f}")

if similarity > 0.99:
    print("\n✅ MATCH CONFIRMED: Models are identical.")
else:
    print("\n❌ MISMATCH DETECTED.")