import numpy as np
import cv2

# --- ROTATION CONSTANTS ---
THRESHOLD_PARALLAX_PA_PASS: float = 0.03  # 3% of image width (~3 degrees)
THRESHOLD_PARALLAX_AP_PASS: float = 0.05  # 5% for portable magnified parallax (~5 degrees)

# Tier 2: Volumetric Symmetry (Lungs)
THRESHOLD_LUNG_AREA_SYMMETRY: float = 0.15   # Max 15% difference in area
THRESHOLD_LUNG_WIDTH_SYMMETRY: float = 0.12  # Max 12% difference in bounding box width

class RotationClass:
    PASS = "pass"
    REJECT = "reject_rotated"
    ERROR = "error"

def calculate_parallax_offset(segment_masks: dict[str, np.ndarray], img_shape: tuple[int, int]) -> dict:
    """Refactored core math for Spine-Sternum parallax detection."""
    h, w = img_shape[:2]
    
    # 1. Extraction 
    spine = segment_masks.get("Spine")
    sternum = segment_masks.get("Sternum")
    
    # 2. Detailed Structural Sanity Reporting
    missing_landmarks = []
    if spine is None or spine.max() == 0: missing_landmarks.append("Spine")
    if sternum is None or sternum.max() == 0: missing_landmarks.append("Sternum")

    if missing_landmarks:
        return {"valid": False, "msg": f"Missing landmarks: {', '.join(missing_landmarks)}"}
    
    assert spine is not None and sternum is not None

    # 3. Line Fitting (Vertical structural axes)
    def get_axis(mask):
        y, x = np.where(mask > 0)
        points = np.column_stack((x, y)).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return vx, vy, x0, y0

    vx_sp, vy_sp, x0_sp, y0_sp = get_axis(spine)
    vx_st, vy_st, x0_st, y0_st = get_axis(sternum)
    
    # Evaluate at the sternum's vertical centroid
    eval_y = y0_st
    
    # Protect against divide-by-zero if spine is perfectly horizontal
    if abs(vy_sp) < 1e-6:
        spine_x = x0_sp
    else:
        spine_x = x0_sp + ((eval_y - y0_sp) / vy_sp) * vx_sp
        
    stern_x = x0_st
    
    # --- SIMPLIFIED SIGNED RATIO ---
    # Positive (+) = Sternum left of spine = Rotated Right
    # Negative (-) = Sternum right of spine = Rotated Left
    pixel_diff_signed = spine_x - stern_x
    offset_ratio_signed = pixel_diff_signed / w
    
    return {
        "valid": True,
        "offset_ratio": float(offset_ratio_signed),
        "spine_line": (vx_sp, vy_sp, x0_sp, y0_sp),
        "sternum_line": (vx_st, vy_st, x0_st, y0_st),
        "eval_pt": (int(spine_x), int(stern_x), int(eval_y))
    }

def calculate_lung_symmetry(segment_masks: dict[str, np.ndarray]) -> dict:
    """Calculates volumetric symmetry between the left and right lungs."""
    lung_r = segment_masks.get("Lung_Right")
    lung_l = segment_masks.get("Lung_Left")
    
    missing_landmarks = []
    if lung_r is None or lung_r.max() == 0: missing_landmarks.append("Lung_Right")
    if lung_l is None or lung_l.max() == 0: missing_landmarks.append("Lung_Left")

    if missing_landmarks:
        return {"valid": False, "msg": f"Missing landmarks: {', '.join(missing_landmarks)}"}
        
    assert lung_r is not None and lung_l is not None

    # Calculate Areas
    area_r = np.sum(lung_r > 0)
    area_l = np.sum(lung_l > 0)
    
    # Calculate Widths via Bounding Boxes
    _, _, w_r, _ = cv2.boundingRect((lung_r > 0).astype(np.uint8))
    _, _, w_l, _ = cv2.boundingRect((lung_l > 0).astype(np.uint8))
    
    # Calculate Skew Ratios (0.0 = perfect symmetry)
    # Using max() in denominator to safely yield a percentage difference between 0.0 and 1.0
    area_ratio = abs(area_r - area_l) / max(area_r, area_l)
    width_ratio = abs(w_r - w_l) / max(w_r, w_l)
    
    # Determine which side is larger (Useful for directional rotation context)
    larger_side = "Right" if area_r > area_l else "Left"

    return {
        "valid": True,
        "area_ratio": float(area_ratio),
        "width_ratio": float(width_ratio),
        "larger_side": larger_side,
        "visuals": {
            "lung_r_w": int(w_r),
            "lung_l_w": int(w_l)
        }
    }

def assess_rotation(segment_masks: dict[str, np.ndarray], img_shape: tuple[int, int], projection: str = "PA") -> dict:
    """Orchestrates the rotation assessment using a signed parallax ratio."""
    parallax = calculate_parallax_offset(segment_masks, img_shape)
    
    if not parallax["valid"]:
        return {"status": RotationClass.ERROR, "metrics": {}, "reasoning": parallax["msg"]}
    
    signed_ratio = parallax["offset_ratio"]
    abs_ratio = abs(signed_ratio)
    
    limit = THRESHOLD_PARALLAX_PA_PASS if projection == "PA" else THRESHOLD_PARALLAX_AP_PASS
    
    # Use the absolute value to check against the strict positive threshold limits
    status = RotationClass.PASS if abs_ratio <= limit else RotationClass.REJECT
    
    if status == RotationClass.PASS:
        reasoning = f"Parallax offset of {signed_ratio:.1%} is within limits for {projection}."
    else:
        reasoning = f"Rotation detected: Spine-Sternum parallax ({signed_ratio:.1%}) exceeds {projection} threshold."

    return {
        "status": status,
        "metrics": {
            "parallax_ratio": signed_ratio,
            "projection": projection,
            "visuals": parallax 
        },
        "reasoning": reasoning
    }