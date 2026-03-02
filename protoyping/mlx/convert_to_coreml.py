import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModel
import transformers.models.siglip.modeling_siglip as siglip_code

# CONFIG
MODEL_PATH = "./medsiglip-vision-v1"
COREML_PATH = "./medsiglip_448_patched.mlpackage"

print("🩹 Preparing Monkey Patch...")

# --- THE MONKEY PATCH ---
# We overwrite the 'forward' method of the Pooling Head.
# Instead of asking "What is the shape?", we tell it: "The shape is 1."
def fixed_forward(self, hidden_states: torch.Tensor, attention_mask=None):
    # HARDCODED CONSTANTS FOR TRACING
    # This prevents the JIT tracer from seeing dynamic shapes
    batch_size = 1
    seq_len = 1024  # (448/14)^2 = 32*32 = 1024 patches
    hidden_dim = 1152
    
    # 1. Expand Probe (Query) using hardcoded batch_size
    # Original: query = self.probe.expand(batch_size, -1, -1)
    query = self.probe.expand(batch_size, -1, -1)
    
    # 2. Key and Value are the input features
    key = hidden_states
    value = hidden_states

    # 3. Multi-head Attention
    # We use the internal attention module directly
    attn_output, _ = self.attention(
        query,
        key,
        value,
        need_weights=False, # We don't need the maps
        is_causal=False
    )

    # 4. Layer Norm
    attn_output = self.layernorm(attn_output)

    # 5. MLP
    attn_output = self.mlp(attn_output)

    # 6. Squeeze to [Batch, Dim]
    return attn_output.squeeze(1)

# APPLY THE PATCH
print("💉 Injecting static logic into Transformers library...")
siglip_code.SiglipMultiheadAttentionPoolingHead.forward = fixed_forward

# --- STANDARD CONVERSION FLOW ---

print(f"🚀 Loading PyTorch Model from {MODEL_PATH}...")
# Load Float32 (CPU) for cleanest trace
hf_model = AutoModel.from_pretrained(MODEL_PATH).float().eval()

# Wrapper to strip dictionary outputs (Core ML hates dicts)
class VisionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.pooler_output

print("📸 Tracing model graph...")
dummy_input = torch.rand(1, 3, 448, 448)
wrapper_model = VisionWrapper(hf_model).eval()

# Tracing should now be smooth because there are no dynamic shapes in the head
traced_model = torch.jit.trace(wrapper_model, dummy_input)

print("⚙️  Converting to Core ML (Neural Engine)...")
model_coreml = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="pixel_values", shape=dummy_input.shape)],
    outputs=[ct.TensorType(name="embedding")],
    minimum_deployment_target=ct.target.iOS17, # Targets M3/M4
    compute_precision=ct.precision.FLOAT16,    # Forces ANE
    convert_to="mlprogram"
)

print(f"💾 Saving to {COREML_PATH}...")
model_coreml.save(COREML_PATH)

print("✅ SUCCESS! You survived Core ML Hell.")