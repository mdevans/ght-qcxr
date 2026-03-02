import os
import cv2
import numpy as np
import urllib.request

def build_other_directory():
    output_dir = "data/other"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created directory: {output_dir}")

    # --- 1. Generate Technical Edge Cases ---
    print("⚙️ Generating technical edge cases (blank, white, noise)...")
    cv2.imwrite(os.path.join(output_dir, "blank_black.jpg"), np.zeros((448, 448, 3), dtype=np.uint8))
    cv2.imwrite(os.path.join(output_dir, "blank_white.jpg"), np.ones((448, 448, 3), dtype=np.uint8) * 255)
    cv2.imwrite(os.path.join(output_dir, "random_noise.jpg"), np.random.randint(0, 256, (448, 448, 3), dtype=np.uint8))

    # --- 2. Download Real-World Out-of-Distribution Images ---
    print("🌐 Downloading external test images...")
    external_images = {
        "xray_hand.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Normal_hand_x-ray.jpg/500px-Normal_hand_x-ray.jpg",
        "xray_knee.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Knee_X-ray.jpg/500px-Knee_X-ray.jpg",
        "mri_brain.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Brain_mri_sag_mac.jpg/500px-Brain_mri_sag_mac.jpg",
        "ct_chest_axial.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/CT_scan_of_the_chest.jpg/500px-CT_scan_of_the_chest.jpg",
        "ultrasound_fetus.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Fetal_profile_ultrasound.jpg/500px-Fetal_profile_ultrasound.jpg",
        "natural_cat.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/500px-Cat03.jpg",
        "natural_car.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/2015_Mazda_MX-5_ND_2.0_SKYACTIV-G_160_i-ELOOP_Rubinrot-Metallic_Vorderansicht.jpg/500px-2015_Mazda_MX-5.jpg",
        "scanned_document.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Sample_of_a_scanned_document.jpg/500px-Sample_of_a_scanned_document.jpg"
    }

    for filename, url in external_images.items():
        filepath = os.path.join(output_dir, filename)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"  ✅ Downloaded: {filename}")
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")

    print("\n🎉 'OTHER' directory is ready! You can now run your classifier test script.")

if __name__ == "__main__":
    build_other_directory()