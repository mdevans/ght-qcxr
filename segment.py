import torch
import numpy as np
import cv2
import re
import shutil
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
import torchxrayvision as xrv

from cxas.models.UNet.backbone_unet import BackboneUNet

# --- CONFIGURATION ---
CXAS_WEIGHTS_PATH = Path.home() / ".cxas" / "weights" / "UNet_ResNet50_default.pth"
IMAGE_DIR = Path("data/TB_Chest_Radiography_Database/Normal")
OUTPUT_DIR = Path("segment/segmentation_output") # Main output directory
SAVE_VISUAL_MASKS = True # Parameter to turn on/off visual mask saving

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- LABEL DICTIONARIES FOR VISUALIZATION ---
CXAS_LABEL_DICT = {
    2: "spine", 29: "sternum", 31: "clavicle_left", 32: "clavicle_right",
    105: "diaphragm", 135: "lung_right", 136: "lung_left"
}
XRV_LABEL_DICT = {
    0: "clavicle_left", 1: "clavicle_right", 13: "spine"
}

# --- HELPER & PREPROCESSING FUNCTIONS ---

def normalize_cxas(array: torch.Tensor) -> torch.Tensor:
    return (array - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / (
           torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

def normalize_xrv(img_array):
    return xrv.datasets.normalize(img_array, 255)

def load_cxas_model():
    print("  -> Loading CXAS model...")
    model = BackboneUNet(model_name="UNet_ResNet50_default", classes=159)
    checkpoint = torch.load(CXAS_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

def load_xrv_model():
    print("  -> Loading TorchXRayVision model...")
    model = xrv.baseline_models.chestx_det.PSPNet().to(DEVICE).eval()
    return model

# --- BATCH PROCESSOR ---

def process_batch():
    print("--- Segmentation Generation Module ---")
    
    print(f"🧹 Clearing and creating output directory: {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print("🧠 Loading segmentation models...")
    cxas_model = load_cxas_model()
    xrv_model = load_xrv_model()

    valid_extensions = {'.png', '.jpg', '.jpeg'}
    def natural_sort_key(path_obj):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path_obj.name)]

    all_files = sorted([p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions], key=natural_sort_key)
    if not all_files:
        print(f"❌ No images found in {IMAGE_DIR}")
        return

    test_files = all_files[:100]

    print(f"\n🧪 Starting segmentation on {len(test_files)} images...")
    print(f"{'FILENAME':<30} | {'STATUS'}")
    print("-" * 80)

    for image_path in test_files:
        filename = image_path.name
        try:
            # Create the specific output directory for this image
            image_output_dir = OUTPUT_DIR / image_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)

            pil_img = Image.open(image_path).convert("RGB")
            orig_h, orig_w = pil_img.height, pil_img.width
            cv_img_raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            with torch.no_grad():
                # CXAS Inference
                cxas_array = np.transpose(np.array(pil_img), [2, 0, 1])
                cxas_tensor = normalize_cxas(torch.tensor(cxas_array).float() / 255.0).unsqueeze(0).to(DEVICE)
                cxas_tensor_interp = F.interpolate(cxas_tensor, size=(512, 512))
                cxas_pred = cxas_model({"data": cxas_tensor_interp})["segmentation_preds"][0]
                cxas_masks = F.interpolate(cxas_pred.unsqueeze(0), size=(orig_h, orig_w), mode="nearest")[0].bool().cpu().numpy()

                # XRV Inference
                xrv_img_resized = cv2.resize(cv_img_raw, (512, 512))
                xrv_tensor = torch.from_numpy(normalize_xrv(xrv_img_resized)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                xrv_output = xrv_model(xrv_tensor)
                xrv_masks_raw = torch.sigmoid(xrv_output[0]).cpu().numpy()
                xrv_masks_resized = np.array([cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) for m in xrv_masks_raw])
                xrv_masks = (xrv_masks_resized > 0.5)

            # --- Save Combined Masks to a Single .npz File (for fast loading) ---
            npz_save_path = image_output_dir / "masks.npz"
            np.savez_compressed(npz_save_path, cxas_masks=cxas_masks, xrv_masks=xrv_masks)
            
            # --- Save Visual Masks (optional, for debugging) ---
            if SAVE_VISUAL_MASKS:
                cxas_visual_dir = image_output_dir / "cxas"
                cxas_visual_dir.mkdir(exist_ok=True)
                for idx, label in CXAS_LABEL_DICT.items():
                    if idx < len(cxas_masks) and np.sum(cxas_masks[idx]) > 0:
                        cv2.imwrite(str(cxas_visual_dir / f"{idx:03d}_{label}.png"), cxas_masks[idx].astype(np.uint8) * 255)

                xrv_visual_dir = image_output_dir / "xrv"
                xrv_visual_dir.mkdir(exist_ok=True)
                for idx, label in XRV_LABEL_DICT.items():
                    if idx < len(xrv_masks) and np.sum(xrv_masks[idx]) > 0:
                        cv2.imwrite(str(xrv_visual_dir / f"{idx:03d}_{label}.png"), xrv_masks[idx].astype(np.uint8) * 255)
            
            print(f"{filename:<30} | ✅ Segmented and Saved to {image_output_dir.relative_to(Path.cwd())}")

        except Exception as e:
            print(f"{filename:<30} | 🚨 ERROR: {e}")

    print("\n" + "="*40)
    print(f"🎯 COMPLETE | Masks saved to subdirectories in: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    process_batch()

