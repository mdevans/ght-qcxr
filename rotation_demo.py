import cv2
import numpy as np
from pathlib import Path
from PIL import ImageColor

# --- Pipeline Imports ---
from segment import load_cxas_model, load_xrv_model, predict_cxas, predict_xrv, resize_mask_back_to_orig, DEVICE
from preprocessing import preprocess_image, KEY_BASE_IMAGE, KEY_CXAS_TENSOR, KEY_XRV_TENSOR
from ensemble import blend_patient_masks
from tests.test_utils import TEST_IMAGES

# --- QA Module Imports ---
from rotation import assess_rotation, RotationClass

# --- CONFIGURATION ---
OUTPUT_DIR = Path("demo_output/rotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- VISUALIZATION CONSTANTS ---
def get_bgr(color_name: str) -> tuple[int, int, int]:
    """Converts a standard W3C color name to a strictly typed BGR 3-tuple for OpenCV."""
    rgb = ImageColor.getrgb(color_name)
    return rgb[2], rgb[1], rgb[0]  # RGB to BGR

COLOR_TEXT        = get_bgr("white")
COLOR_SPINE       = get_bgr("red")
COLOR_STERNUM     = get_bgr("lime")
COLOR_LUNG        = get_bgr("dodgerblue")
COLOR_CLAVICLE    = get_bgr("magenta")
COLOR_BG          = get_bgr("black")

def draw_rotation_overlay(base_img: np.ndarray, masks: dict, result: dict) -> np.ndarray:
    """Draws all three tiers of rotation analysis with color-coded geometry."""
    if len(base_img.shape) == 2:
        overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base_img.copy()

    h, w = overlay.shape[:2]
    scale = max(0.5, w / 1000.0)
    thick = max(1, int(2 * scale))
    font_scale = 0.7 * scale
    pad = int(8 * scale)
    point_radius = max(3, int(6 * scale))
    
    # 1. MASK OVERLAYS (Low Opacity)
    mask_layer = np.zeros_like(overlay)
    segment_map = {
        "Spine": COLOR_SPINE,
        "Sternum": COLOR_STERNUM,
        "Lung_Right": COLOR_LUNG,
        "Lung_Left": COLOR_LUNG,
        "Clavicle_Right": COLOR_CLAVICLE,
        "Clavicle_Left": COLOR_CLAVICLE
    }
    
    for seg_name, color in segment_map.items():
        if seg_name in masks and np.any(masks[seg_name]):
            mask_layer[masks[seg_name] > 0] = color
        
    cv2.addWeighted(mask_layer, 0.25, overlay, 0.75, 0, overlay)

    # 2. TIER VISUALS
    if result["status"] != RotationClass.ERROR:
        m = result["metrics"]
        
        # Tier 1 & 3 Midline Visuals
        if "visuals" in m:
            vis = m["visuals"]
            sp_x, st_x, eval_y = vis["eval_pt"]
            
            # Draw Spine Line (Red)
            vx_sp, vy_sp, x0_sp, y0_sp = vis["spine_line"]
            top_sp = (int(x0_sp - y0_sp * (vx_sp / vy_sp)), 0) if abs(vy_sp) > 1e-6 else (int(x0_sp), 0)
            bot_sp = (int(x0_sp + (h - y0_sp) * (vx_sp / vy_sp)), h) if abs(vy_sp) > 1e-6 else (int(x0_sp), h)
            cv2.line(overlay, top_sp, bot_sp, COLOR_SPINE, thick, cv2.LINE_AA)
            
            # Draw Sternum Line (Lime)
            vx_st, vy_st, x0_st, y0_st = vis["sternum_line"]
            top_st = (int(x0_st - y0_st * (vx_st / vy_st)), 0) if abs(vy_st) > 1e-6 else (int(x0_st), 0)
            bot_st = (int(x0_st + (h - y0_st) * (vx_st / vy_st)), h) if abs(vy_st) > 1e-6 else (int(x0_st), h)
            cv2.line(overlay, top_st, bot_st, COLOR_STERNUM, thick, cv2.LINE_AA)

            # Parallax Bridge
            cv2.line(overlay, (sp_x, eval_y), (st_x, eval_y), COLOR_TEXT, thick)
            cv2.circle(overlay, (sp_x, eval_y), point_radius, COLOR_SPINE, -1)
            cv2.circle(overlay, (st_x, eval_y), point_radius, COLOR_STERNUM, -1)

        # Tier 2: Lung Outlines (Perimeter Resilience)
        for lung_side in ["Lung_Right", "Lung_Left"]:
            if lung_side in masks and np.any(masks[lung_side]):
                cnts, _ = cv2.findContours(masks[lung_side].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, COLOR_LUNG, thick)

    # 3. METRICS OVERLAY
    is_pass = result["status"] == RotationClass.PASS
    status_label_color = (0, 255, 0) if is_pass else (0, 0, 255) # Green/Red status marker
    
    info_lines = [f"Verdict: {result['status'].upper()}"]
    if result["status"] != RotationClass.ERROR:
        if m.get("parallax_ratio") is not None:
            info_lines.append(f"T1 Parallax: {m['parallax_ratio']:+.1%}")
        if m.get("lung_contour_skew") is not None:
            info_lines.append(f"T2 Lung Skew: {m['lung_contour_skew']:.1%}")
        if m.get("clavicle_skew") is not None:
            info_lines.append(f"T3 Clavicle Skew: {m['clavicle_skew']:.1%}")

    start_x, y_offset = int(20 * scale), int(40 * scale)
    for i, line in enumerate(info_lines):
        (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        cv2.rectangle(overlay, (start_x - pad, y_offset - th - pad), (start_x + tw + pad, y_offset + base + pad), COLOR_BG, -1)
        # First line (Verdict) gets status color, rest stay white
        txt_color = status_label_color if i == 0 else COLOR_TEXT
        cv2.putText(overlay, line, (start_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thick, cv2.LINE_AA)
        y_offset += th + base + (pad * 2)

    # 4. REASONING (Bottom Left)
    reasoning = result["reasoning"]
    r_scale = font_scale * 0.8
    (rw, rh), rb = cv2.getTextSize(reasoning, cv2.FONT_HERSHEY_SIMPLEX, r_scale, thick)
    rx, ry = start_x, h - int(40 * scale)
    cv2.rectangle(overlay, (rx - pad, ry - rh - pad), (rx + rw + pad, ry + rb + pad), COLOR_BG, -1)
    cv2.putText(overlay, reasoning, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, r_scale, COLOR_TEXT, thick, cv2.LINE_AA)

    return overlay

def main():
    print(f"🚀 Initializing Rotation Demo on {DEVICE}...")
    cxas_model = load_cxas_model(DEVICE)
    xrv_model = load_xrv_model(DEVICE)

    for img_path in TEST_IMAGES:
        data = preprocess_image(img_path)
        base_img = data[KEY_BASE_IMAGE]
        
        # Segmentation & Ensemble
        cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(DEVICE))
        xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(DEVICE))
        blended = blend_patient_masks(cxas_raw, xrv_raw)
        
        # Post-processing
        masks = {k: resize_mask_back_to_orig(v, base_img.shape) for k, v in blended.items() if v is not None}
        
        # QA Logic
        result = assess_rotation(masks, base_img.shape)
        
        # Visualization
        overlay = draw_rotation_overlay(base_img, masks, result)
        out_path = OUTPUT_DIR / f"rot_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"✅ {img_path.name} -> {result['status'].upper()}")

if __name__ == "__main__":
    main()