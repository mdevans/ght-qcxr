import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from cxas.models.UNet.backbone_unet import BackboneUNet

from preprocessing import KEY_BASE_IMAGE, preprocess_image, KEY_CXAS_TENSOR

# --- CONFIG ---
WEIGHTS_PATH = Path.home() / ".cxas" / "weights" / "UNet_ResNet50_default.pth"
# TEST_IMAGE = "data/cxr/cxr_normal_0.jpg"
TEST_IMAGE = "tests/test_data/cxr1.jpg"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


LABEL_DICT = {
    0: "spine", 
    31: "clavicle_left", 
    32: "clavicle_right"
}

def normalize(array: torch.Tensor) -> torch.Tensor:
    """Perfect clone of cxas FileLoader.normalize"""
    array = (array - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / (
        torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    )
    return array

def run_standalone_sanity_check():
    print("1. Loading Model Blueprint...")
    model = BackboneUNet(model_name="UNet_ResNet50_default", classes=159)
    
    print("2. Loading Weights...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # print("3. Preprocessing Image (Exact cxas FileLoader Clone)...")
    # try:
    #     # 1. Load using PIL
    #     pil_img = Image.open(TEST_IMAGE).convert(mode="RGB")
    #     orig_file_size = (pil_img.size[1], pil_img.size[0]) # (H, W)
    # except Exception as e:
    #     print(f"❌ Error loading image: {e}")
    #     return
        
    # array = np.array(pil_img)
    # array = np.transpose(array, [2, 0, 1])
    # array = torch.tensor(array).float() / 255.0
    # array = normalize(array).to(DEVICE)
    
    # # cxas defaults base_size to 512! We missed this in the previous code.
    # # We were forcing it to 1024, but their code says: self.base_size = 512
    # img_tensor = F.interpolate(array.unsqueeze(0), size=(512, 512))
    
    pp_result = preprocess_image(Path(TEST_IMAGE))
    img_tensor = pp_result[KEY_CXAS_TENSOR].to(DEVICE)
    orig_file_size = pp_result[KEY_BASE_IMAGE].shape[:2] # (H, W)
    
    input_dict = {"data": img_tensor}

    print("4. Running Inference...")
    with torch.no_grad():
        output_dict = model(input_dict)
        
    # Get raw boolean predictions
    segmentation = output_dict["segmentation_preds"][0]

    # 5. Exact clone of cxas resize_to_numpy
    pred = segmentation.float()
    pred = F.interpolate(pred.unsqueeze(0), size=orig_file_size, mode="nearest")
    masks = pred[0].bool().cpu().numpy()

    print("6. Saving Masks (Exact cxas FileSaver Clone)...")
    out_dir = Path("cxas_test_output_3")
    out_dir.mkdir(exist_ok=True)
    
    for idx, label in LABEL_DICT.items():
        # EXACT clone of `export_prediction_as_png`
        # Using PIL's 1-bit binary conversion to guarantee perfect edges
        out_path = out_dir / f"{idx:03d}_{label}.png"
        Image.fromarray(masks[idx]).convert("1").save(str(out_path))
            
    print(f"✅ Success! Check the '{out_dir.name}' folder.")
    print("The edges should now be perfectly crisp and identical to the CLI.")

if __name__ == "__main__":
    run_standalone_sanity_check()