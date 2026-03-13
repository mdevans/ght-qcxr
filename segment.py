import torch
import cv2
import numpy as np
import json
from pathlib import Path
import torchxrayvision as xrv

# Import your CXAS architecture (Adjust import path if your project structure differs)
from cxas.models.UNet.backbone_unet import BackboneUNet

# Import your bulletproof preprocessing module
from preprocessing import (
    KEY_BASE_IMAGE,
    KEY_CXAS_TENSOR,
    KEY_XRV_TENSOR,
    preprocess_image,
)

# --- CONFIGURATION ---
INPUT_DIR = Path("data/input_images")
OUTPUT_DIR = Path("data/output_masks")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CXAS_WEIGHTS_PATH = Path.home() / ".cxas" / "weights" / "UNet_ResNet50_default.pth"

# ==========================================
# 1. SPECIFIC MODEL LOGIC
# ==========================================

def load_cxas_model(device):
    print(f"🚀 Loading CXAS UNet on {device}...")
    model = BackboneUNet(model_name="UNet_ResNet50_default", classes=159)
    try:
        checkpoint = torch.load(CXAS_WEIGHTS_PATH, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    except FileNotFoundError:
        print(f"⚠️ CXAS weights not found at {CXAS_WEIGHTS_PATH}. Ensure they are downloaded before inference.")
    
    model.to(device).eval()
    return model

def predict_cxas(model, tensor: torch.Tensor) -> dict[str, np.ndarray]:
    extracted_masks = {}
    with torch.no_grad():
        output_dict = model({"data": tensor})
        # pred shape is (159, 512, 512)
        pred = output_dict["segmentation_preds"][0] 
        
        # CXAS natively outputs pre-thresholded booleans/logits in this tensor
        masks = pred.bool().cpu().numpy()
        
        for class_idx in range(masks.shape[0]):
            if class_idx == 0: 
                continue # Skip background
                
            binary_mask = masks[class_idx].astype(np.uint8)
            
            # Only save if this anatomy actually exists in the image
            if binary_mask.max() > 0:
                extracted_masks[f"class_{class_idx:03d}"] = binary_mask

    return extracted_masks

def load_xrv_model(device):
    print(f"🚀 Loading XRV PSPNet on {device}...")
    model = xrv.baseline_models.chestx_det.PSPNet()
    model.to(device).eval()
    return model

def predict_xrv(model, tensor: torch.Tensor) -> dict[str, np.ndarray]:
    extracted_masks = {}
    with torch.no_grad():
        output = model(tensor).squeeze(0) # Shape: (14, 512, 512)
        
        for i, target_name in enumerate(model.targets):
            binary_mask = (output[i] > 0.5).cpu().numpy().astype(np.uint8)
            if binary_mask.max() > 0:
                safe_name = target_name.replace(" ", "_")
                extracted_masks[safe_name] = binary_mask

    return extracted_masks


# ==========================================
# 2. CORE GEOMETRY & POST-PROCESSING
# ==========================================

def resize_mask_back_to_orig(mask_512: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Stretches the 512x512 mask directly back to the original image dimensions."""
    orig_h, orig_w = original_shape[:2]
    # Simple direct resize reverses the geometric squash applied in preprocessing
    return cv2.resize(mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

def extract_polygons_from_mask(binary_mask: np.ndarray, min_area: float = 50.0) -> list:
    """Extracts boundaries as a list of polygons (each polygon is a list of [x, y] coordinates)."""
    mask_8u = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Safely unpack the OpenCV (N, 1, 2) shape into a standard Python list of [x, y]
            poly = [pt[0].tolist() for pt in contour]
            polygons.append(poly)
            
    return polygons


# ==========================================
# 3. BATCH ORCHESTRATOR
# ==========================================

def process_directory(input_dir: Path, output_dir: Path, model_configs: list[dict]):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    valid_extensions = {".png", ".jpg", ".jpeg", ".dcm"}
    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]

    print(f"\n🎯 Starting pipeline on {len(image_files)} images...")

    for img_path in image_files:
        print(f"  Processing: {img_path.name}")
        
        try:
            # 1. Preprocess
            preprocessed_data = preprocess_image(img_path)
            orig_shape = preprocessed_data[KEY_BASE_IMAGE].shape

            img_out_dir = output_dir / img_path.stem
            img_out_dir.mkdir(parents=True, exist_ok=True)

            # 2. Execute Models
            for config in model_configs:
                input_tensor = preprocessed_data[config["tensor_key"]].to(DEVICE)
                mask_dict_512 = config["predict_fn"](config["model"], input_tensor)
                
                model_out_dir = img_out_dir / config["name"]
                model_out_dir.mkdir(exist_ok=True)
                
                numeric_masks_to_save = {}
                json_polygons_to_save = {}
                
                # 3. Format and Extract Features
                for anatomy_name, raw_mask_512 in mask_dict_512.items():
                    final_mask = resize_mask_back_to_orig(raw_mask_512, orig_shape)
                    
                    numeric_masks_to_save[anatomy_name] = final_mask
                    json_polygons_to_save[anatomy_name] = extract_polygons_from_mask(final_mask)
                    
                    # (Optional) Save PNG for quick visual inspection
                    cv2.imwrite(str(model_out_dir / f"{anatomy_name}.png"), final_mask * 255)

                # 4. Save Aggregated Output Files
                if numeric_masks_to_save:
                    archive_path = model_out_dir / f"{config['name']}_masks.npz"
                    np.savez_compressed(archive_path, **numeric_masks_to_save)
                    
                if json_polygons_to_save:
                    json_path = model_out_dir / f"{config['name']}_polygons.json"
                    with open(json_path, "w") as f:
                        json.dump(json_polygons_to_save, f, indent=2)

        except Exception as e:
            print(f"🚨 Error processing {img_path.name}: {e}")

    print("✅ Batch processing complete.")

# ==========================================
# 4. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    active_models = [
        {
            "name": "cxas",
            "tensor_key": KEY_CXAS_TENSOR,
            "model": load_cxas_model(DEVICE),
            "predict_fn": predict_cxas
        },
        {
            "name": "xrv",
            "tensor_key": KEY_XRV_TENSOR,
            "model": load_xrv_model(DEVICE),
            "predict_fn": predict_xrv
        }
    ]
    
    process_directory(INPUT_DIR, OUTPUT_DIR, active_models)