import cv2
import numpy as np
from pathlib import Path

from segment import load_cxas_model, predict_cxas, DEVICE
from preprocessing import preprocess_image, KEY_CXAS_TENSOR

def run_visual_debugger():
    image_path = Path("tests/test_data/cxr3.png")
    out_dir = Path("debug_cxas_masks")
    out_dir.mkdir(exist_ok=True)

    print(f"🔍 Debugging CXAS outputs for {image_path.name}...")

    # 1. Preprocess and run inference
    data = preprocess_image(image_path)
    tensor = data[KEY_CXAS_TENSOR].to(DEVICE)
    model = load_cxas_model(DEVICE)
    masks = predict_cxas(model, tensor)

    print(f"✅ CXAS found {len(masks)} distinct anatomical structures.")

    # 2. Load the base image to create context overlays
    # (Assuming we have a fallback if cv2 can't read DCM directly without pydicom)
    # We will just use a blank black background if the base image fails to load 
    # so you can at least see the mask shape.
    base_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if base_img is not None:
        base_img = cv2.resize(base_img, (512, 512))
        base_img_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        print("Note: Using black background for overlays (cv2 couldn't read raw dcm).")
        base_img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)

    # 3. Export every single mask as its own image
    for class_key, mask in masks.items():
        if mask.max() > 0:
            overlay = base_img_bgr.copy()
            
            # Draw the mask in bright yellow
            overlay[mask > 0] = (0, 255, 255) 
            
            # Print the class name huge and red in the top corner
            cv2.putText(overlay, class_key, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            save_path = out_dir / f"{class_key}.png"
            cv2.imwrite(str(save_path), overlay)
            
    print(f"🎯 Done! Open the '{out_dir}' folder and look for the Sternum.")

if __name__ == "__main__":
    run_visual_debugger()