import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.transforms.functional import normalize
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from cxas.models.UNet.backbone_unet import BackboneUNet

# --- CONFIG ---
WEIGHTS_PATH = Path.home() / ".cxas" / "weights" / "UNet_ResNet50_default.pth"
TEST_IMAGE = "data/cxr/cxr_normal_0.jpg"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "UNet_ResNet50_default" 
NUM_CLASSES = 159 

LABEL_DICT = {
    "0": "spine", "1": "cervical spine", "2": "thoracic spine", "3": "lumbar spine", "4": "vertebrae C1", "5": "vertebrae C2", "6": "vertebrae C3", "7": "vertebrae C4", "8": "vertebrae C5", "9": "vertebrae C6", "10": "vertebrae C7", "11": "vertebrae T1", "12": "vertebrae T2", "13": "vertebrae T3", "14": "vertebrae T4", "15": "vertebrae T5", "16": "vertebrae T6", "17": "vertebrae T7", "18": "vertebrae T8", "19": "vertebrae T9", "20": "vertebrae T10", "21": "vertebrae T11", "22": "vertebrae T12", "23": "vertebrae L1", "24": "vertebrae L2", "25": "vertebrae L3", "26": "vertebrae L4", "27": "vertebrae L5", "28": "rib_cartilage", "29": "sternum", "30": "clavicles", "31": "clavicle left", "32": "clavicle right", "33": "scapulas", "34": "scapula left", "35": "scapula right", "36": "posterior 12th rib right", "37": "posterior 12th rib left", "38": "anterior 11th rib right", "39": "posterior 11th rib right", "40": "anterior 11th rib left", "41": "posterior 11th rib left", "42": "anterior 10th rib right", "43": "posterior 10th rib right", "44": "anterior 10th rib left", "45": "posterior 10th rib left", "46": "anterior 9th rib right", "47": "posterior 9th rib right", "48": "anterior 9th rib left", "49": "posterior 9th rib left", "50": "anterior 8th rib right", "51": "posterior 8th rib right", "52": "anterior 8th rib left", "53": "posterior 8th rib left", "54": "anterior 7th rib right", "55": "posterior 7th rib right", "56": "anterior 7th rib left", "57": "posterior 7th rib left", "58": "anterior 6th rib right", "59": "posterior 6th rib right", "60": "anterior 6th rib left", "61": "posterior 6th rib left", "62": "anterior 5th rib right", "63": "posterior 5th rib right", "64": "anterior 5th rib left", "65": "posterior 5th rib left", "66": "anterior 4th rib right", "67": "posterior 4th rib right", "68": "anterior 4th rib left", "69": "posterior 4th rib left", "70": "anterior 3rd rib right", "71": "posterior 3rd rib right", "72": "anterior 3rd rib left", "73": "posterior 3rd rib left", "74": "anterior 2nd rib right", "75": "posterior 2nd rib right", "76": "anterior 2nd rib left", "77": "posterior 2nd rib left", "78": "anterior 1st rib right", "79": "posterior 1st rib right", "80": "anterior 1st rib left", "81": "posterior 1st rib left", "82": "12th rib", "83": "posterior 11th rib", "84": "anterior 11th rib", "85": "posterior 10th rib", "86": "anterior 10th rib", "87": "posterior 9th rib", "88": "anterior 9th rib", "89": "posterior 8th rib", "90": "anterior 8th rib", "91": "posterior 7th rib", "92": "anterior 7th rib", "93": "posterior 6th rib", "94": "anterior 6th rib", "95": "posterior 5th rib", "96": "anterior 5th rib", "97": "posterior 4th rib", "98": "anterior 4th rib", "99": "posterior 3rd rib", "100": "anterior 3rd rib", "101": "posterior 2nd rib", "102": "anterior 2nd rib", "103": "posterior 1st rib", "104": "anterior 1st rib", "105": "diaphragm", "106": "left hemidiaphragm", "107": "right hemidiaphragm", "108": "stomach", "109": "small bowel", "110": "duodenum", "111": "liver", "112": "pancreas", "113": "kidney left", "114": "kidney right", "115": "cardiomediastinum", "116": "upper mediastinum", "117": "lower mediastinum", "118": "anterior mediastinum", "119": "middle mediastinum", "120": "posterior mediastinum", "121": "heart", "122": "heart atrium left", "123": "heart atrium right", "124": "heart myocardium", "125": "heart ventricle left", "126": "heart ventricle right", "127": "aorta", "128": "ascending aorta", "129": "descending aorta", "130": "aortic arch", "131": "pulmonary artery", "132": "inferior vena cava", "133": "esophagus", "134": "lung", "135": "right lung", "136": "left lung", "137": "lung base", "138": "mid zone lung", "139": "upper zone lung", "140": "apical zone lung", "141": "right upper zone lung", "142": "right mid zone lung", "143": "right lung base", "144": "right apical zone lung", "145": "left upper zone lung", "146": "left mid zone lung", "147": "left lung base", "148": "left apical zone lung", "149": "lung lower lobe left", "150": "lung upper lobe left", "151": "lung lower lobe right", "152": "lung middle lobe right", "153": "lung upper lobe right", "154": "trachea", "155": "tracheal bifurcation", "156": "breast", "157": "breast left", "158": "breast right"
}

def run_sanity_check():
    print("1. Loading Model Blueprint (ResNet50 Backbone)...")
    model = BackboneUNet(model_name=MODEL_NAME, classes=NUM_CLASSES)
    
    print("2. Loading Weights...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    print("3. Preprocessing Image (Strict CXAS Replication)...")
    
    # 1. Load using PIL and convert to RGB (Exactly like cxas)
    try:
        pil_img = Image.open(TEST_IMAGE).convert(mode="RGB")
    except Exception as e:
        print(f"❌ Could not load image from {TEST_IMAGE}! Error: {e}")
        return
        
    array = np.array(pil_img)
    
    # 2. Transpose to (Channels, Height, Width)
    array = np.transpose(array, [2, 0, 1])
    
    # 3. Convert to tensor and divide by 255.0
    tensor_array = torch.tensor(array).float() / 255.0
    
    # 4. ImageNet Normalization (This is what self.normalize does)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor_normalized = normalize(tensor_array, mean=mean, std=std)
    
    # Move to GPU and add batch dimension (1, C, H, W)
    tensor_gpu = tensor_normalized.unsqueeze(0).to(DEVICE)
    
    # 5. Interpolate to the base size using PyTorch (Exactly like cxas)
    # The default base_size for this model is 1024x1024. 
    # mode="bilinear" and align_corners=False are standard PyTorch defaults for 2D interpolation.
    img_tensor = F.interpolate(tensor_gpu, size=(1024, 1024), mode='bilinear', align_corners=False)

    input_dict = {"data": img_tensor}

    print("4. Running Inference...")
    with torch.no_grad():
        output_dict = model(input_dict)
        
    # Get the raw 1024x1024 boolean masks
    raw_masks = output_dict["segmentation_preds"].float() 
    
    # NEW: "Un-squish" the masks back to the original image size!
    # (Matches the cxas `resize_to_numpy` function exactly)
    orig_size = (pil_img.size[1], pil_img.size[0]) # PIL size is (W, H), PyTorch needs (H, W)
    resized_masks = F.interpolate(raw_masks, size=orig_size, mode="nearest")
    
    # Convert back to numpy arrays
    masks = resized_masks.squeeze().cpu().numpy()

    print("5. Saving visual masks...")
    out_dir = Path("cxas_test_output")
    out_dir.mkdir(exist_ok=True)
    
    # REMOVED the >0 filter so you get exactly 159 files just like the CLI
    for i in range(masks.shape[0]):
        channel_mask = (masks[i]).astype(np.uint8) * 255
        
        safe_label = LABEL_DICT.get(str(i), f"unknown_{i}").replace(" ", "_")
        filename = f"{i:03d}_{safe_label}.png"
        cv2.imwrite(str(out_dir / filename), channel_mask)
            
    print(f"✅ Success! Saved all {NUM_CLASSES} classes to the '{out_dir.name}' folder.")
    print("The shapes will now perfectly match the CLI output!")

if __name__ == "__main__":
    run_sanity_check()