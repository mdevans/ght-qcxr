import os
import random
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from cxr_pipeline import load_and_preprocess

# CONFIG
SOURCE_IMAGE = "data/cxr/cxr_normal_0.jpg"
OUTPUT_COMPARE = "debug_rotation_clavicle_independent.jpg"

def apply_clavicle_shift(pil_img):
    """
    Simulates rotation by shifting the Clavicles horizontally while keeping the Spine pinned.
    
    Mechanism:
    - Target the Left Clavicle region (approx 25% width, 15% height).
    - Target the Right Clavicle region (approx 75% width, 15% height).
    - Shift BOTH towards the same direction (e.g., Right).
    - Result: One gap shrinks, the other expands.
    """
    img_arr = np.array(pil_img)
    h, w, c = img_arr.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    # 1. Define Anatomical Targets
    # Clavicle Heads are roughly at 25% and 75% width, 15% down from top
    left_clav_y, left_clav_x   = int(h * 0.15), int(w * 0.25)
    right_clav_y, right_clav_x = int(h * 0.15), int(w * 0.75)
    
    # Sigma: Control the spread so it affects the shoulder/clavicle but NOT the spine
    # We need a tight horizontal sigma to avoid moving the spine (at 50%)
    sigma_y = h * 0.20  # Vertical spread (affects top of lung)
    sigma_x = w * 0.10  # Horizontal spread (stops before midline)
    
    # 2. Create Masks for Left and Right Clavicles
    # Left Mask
    dist_L = ((y_grid - left_clav_y)**2 / (2*sigma_y**2)) + ((x_grid - left_clav_x)**2 / (2*sigma_x**2))
    mask_L = np.exp(-dist_L)
    
    # Right Mask
    dist_R = ((y_grid - right_clav_y)**2 / (2*sigma_y**2)) + ((x_grid - right_clav_x)**2 / (2*sigma_x**2))
    mask_R = np.exp(-dist_R)
    
    # Combine Masks (We treat them as separate influencers)
    # Note: At x=50% (Spine), both masks should be near zero.
    
    # 3. Apply Shift
    # Direction: -1 (Move Left) or 1 (Move Right)
    direction = random.choice([-1, 1]) 
    strength = random.uniform(40, 70) * direction
    
    # Calculate Displacement
    # If we shift Right (+), pixels are pulled from Left.
    # Left Clavicle (starts at 25%) moves towards 50% (Spine) -> GAP SHRINKS
    # Right Clavicle (starts at 75%) moves towards 100% (Edge) -> GAP GROWS
    displacement = (mask_L + mask_R) * strength
    
    x_new = x_grid - displacement
    
    # 4. Warp
    warped_img = np.zeros_like(img_arr)
    for i in range(c):
        warped_img[:, :, i] = map_coordinates(
            img_arr[:, :, i], 
            [y_grid, x_new], 
            order=1, 
            mode='nearest'
        )
        
    side_note = "Right" if direction > 0 else "Left" # Direction the ANATOMY moved
    return Image.fromarray(warped_img), side_note

def create_comparison(img_pass, img_fail, note):
    w, h = img_pass.size
    comp = Image.new("RGB", (w * 2, h + 50), (0, 0, 0))
    comp.paste(img_pass, (0, 50))
    comp.paste(img_fail, (w, 50))
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comp)
    try: font = ImageFont.truetype("Arial.ttf", 40)
    except: font = ImageFont.load_default()
        
    draw.text((20, 10), "PASS (Symmetric)", fill="#00FF00", font=font)
    draw.text((w + 20, 10), f"FAIL (Clavicles shifted {note})", fill="#FF0000", font=font)
    
    # Helper Lines: Draw vertical lines at clavicle heads and spine
    # We estimate positions for visualization
    center = w // 2
    clav_L = int(w * 0.25)
    clav_R = int(w * 0.75)
    
    # Draw reference lines on BOTH images
    for offset in [0, w]:
        # Spine (Yellow)
        draw.line([(center + offset, 50), (center + offset, h+50)], fill="yellow", width=2)
        # Clavicle zones (Cyan) - rough guides
        draw.line([(clav_L + offset, 50), (clav_L + offset, h+50)], fill="cyan", width=1)
        draw.line([(clav_R + offset, 50), (clav_R + offset, h+50)], fill="cyan", width=1)

    return comp

def main():
    if not os.path.exists(SOURCE_IMAGE):
        print(f"❌ Error: {SOURCE_IMAGE} not found.")
        return

    print(f"🧪 Generating Independent Clavicle Shift: {SOURCE_IMAGE}")

    # 1. Load 
    _, img_pass = load_and_preprocess(SOURCE_IMAGE)
    if img_pass is None: return

    # 2. Apply Warp
    img_fail, direction = apply_clavicle_shift(img_pass)
    
    # 3. Save Comparison
    comp = create_comparison(img_pass, img_fail, direction)
    comp.save(OUTPUT_COMPARE)
    
    print(f"✅ Comparison saved to: [bold]{OUTPUT_COMPARE}[/bold]")
    print("   Open this file.")
    print("   Look at the distance between the Cyan Line (Clavicle) and Yellow Line (Spine).")
    print("   One side should be significantly narrower than the other.")

if __name__ == "__main__":
    main()