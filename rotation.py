import numpy as np
import cv2
from typing import Any

# ==========================================
# --- ROTATION THRESHOLDS & CONSTANTS ---
# ==========================================

# Tier 1: Skeletal Parallax
THRESHOLD_PARALLAX_PA_PASS: float = 0.03 
THRESHOLD_PARALLAX_AP_PASS: float = 0.05 

# Tier 2: Area Symmetry
THRESHOLD_LUNG_AREA_SYMMETRY: float = 0.10 

# --- Tier 0: Anatomic Sanity Constraints ---
MIN_SKELETAL_AREA_RATIO: float = 1.5   # Spine area should be > 1.5x larger than Sternum
MEDIASTINAL_CORRIDOR_RATIO: float = 0.3 # Landmarks must be within 30% of image centerline
MIN_LUNG_SKELETAL_AREA_RATIO: float = 2.0   # Each lung area should be > 2x the Spine area

# ==========================================

class RotationClass:
    PASS = "pass"
    REJECT = "reject_rotated"
    ERROR = "error"

# ==========================================
# 0. TIER 0: ANATOMIC SANITY GATE
# ==========================================

def passes_anatomic_sanity(segment_masks: dict[str, np.ndarray], img_shape: tuple[int, int]) -> tuple[bool, str]:
    """Validates anatomical logic and relative mask areas before performing QA math."""
    h, w = img_shape[:2]
    mid_x = w // 2
    corridor = w * MEDIASTINAL_CORRIDOR_RATIO

    # 1. Extraction & Existence
    spine = segment_masks.get("Spine")
    sternum = segment_masks.get("Sternum")
    lung_r = segment_masks.get("Lung_Right")
    lung_l = segment_masks.get("Lung_Left")

    missing = [k for k, v in {"Spine": spine, "Sternum": sternum, "Lung_R": lung_r, "Lung_L": lung_l}.items() 
               if v is None or np.count_nonzero(v) == 0]
    
    if missing:
        return False, f"Anatomic Failure: Missing landmarks ({', '.join(missing)})"
    
    assert spine is not None and sternum is not None and lung_r is not None and lung_l is not None, "All required masks must be provided."

    # 2. Area-Based Logic Checks
    # np.count_nonzero is used to ensure we are counting pixels regardless of dtype
    area_sp = int(np.count_nonzero(spine > 0))
    area_st = int(np.count_nonzero(sternum > 0))
    area_lr = int(np.count_nonzero(lung_r > 0))
    
    if area_sp < (area_st * MIN_SKELETAL_AREA_RATIO):
        return False, "Anatomic Failure: Sternum area is impossibly large relative to Spine."
    
    if area_lr < (area_sp * MIN_LUNG_SKELETAL_AREA_RATIO):
        return False, "Anatomic Failure: Skeletal area is disproportionate to Lung area."

    # 3. Spatial Centering (Mediastinal Corridor)
    def get_centroid_x(mask):
        M = cv2.moments((mask > 0).astype(np.uint8))
        return int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0

    sp_x = get_centroid_x(spine)
    st_x = get_centroid_x(sternum)

    if abs(sp_x - mid_x) > corridor or abs(st_x - mid_x) > corridor:
        return False, "Anatomic Failure: Skeletal landmarks outside central mediastinal corridor."

    return True, "Passed"

# ==========================================
# 1. ENGINE A: SKELETAL PARALLAX
# ==========================================

def calculate_parallax_offset(segment_masks: dict[str, np.ndarray], img_shape: tuple[int, int]) -> dict[str, Any]:
    h, w = img_shape[:2]
    spine, sternum = segment_masks["Spine"], segment_masks["Sternum"]

    def get_axis(mask):
        y, x = np.where(mask > 0)
        pts = np.column_stack((x, y)).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return vx, vy, x0, y0

    vx_sp, vy_sp, x0_sp, y0_sp = get_axis(spine)
    vx_st, vy_st, x0_st, y0_st = get_axis(sternum)
    
    eval_y = y0_st
    spine_x = x0_sp if abs(vy_sp) < 1e-6 else x0_sp + ((eval_y - y0_sp) / vy_sp) * vx_sp
    
    # Positive (+) = Right rotation, Negative (-) = Left rotation
    offset_ratio = (spine_x - x0_st) / w
    
    return {
        "valid": True,
        "offset_ratio": float(offset_ratio),
        "visuals": {
            "spine_line": (vx_sp, vy_sp, x0_sp, y0_sp),
            "sternum_line": (vx_st, vy_st, x0_st, y0_st),
            "eval_pt": (int(spine_x), int(x0_st), int(eval_y))
        }
    }

# ==========================================
# 2. ENGINE B: AREA SYMMETRY
# ==========================================

def calculate_lung_symmetry(segment_masks: dict[str, np.ndarray]) -> dict[str, Any]:
    """Calculates area skew relative to total lung area for geometric stability."""
    lung_r, lung_l = segment_masks["Lung_Right"], segment_masks["Lung_Left"]
    area_r = int(np.count_nonzero(lung_r > 0))
    area_l = int(np.count_nonzero(lung_l > 0))
    total_area = area_r + area_l
    
    area_skew = abs(area_r - area_l) / total_area if total_area > 0 else 0.0

    return {
        "valid": True,
        "area_ratio": float(area_skew),
        "larger_side": "Right" if area_r > area_l else "Left"
    }

# ==========================================
# 3. CONSENSUS JUDGE
# ==========================================

def resolve_rotation_consensus(parallax: dict[str, Any], symmetry: dict[str, Any], projection: str) -> tuple[str, str]:
    limit_p = THRESHOLD_PARALLAX_PA_PASS if projection == "PA" else THRESHOLD_PARALLAX_AP_PASS
    
    v_parallax = RotationClass.PASS if abs(parallax["offset_ratio"]) <= limit_p else RotationClass.REJECT
    v_symmetry = RotationClass.PASS if symmetry["area_ratio"] <= THRESHOLD_LUNG_AREA_SYMMETRY else RotationClass.REJECT
        
    # --- UPDATED CONSENSUS: Either pass is a pass ---
    if v_parallax == RotationClass.PASS or v_symmetry == RotationClass.PASS:
        status = RotationClass.PASS
        reasoning = f"PASS: (Parallax: {v_parallax.upper()} {parallax['offset_ratio']:+.1%}, Lung Skew: {v_symmetry.upper()} {symmetry['area_ratio']:.1%})"
    else:
        status = RotationClass.REJECT
        reasoning = f"REJECT: Both engines detected rotation (Parallax {parallax['offset_ratio']:+.1%}, Lung Skew {symmetry['area_ratio']:.1%})"
    
    return status, reasoning

# ==========================================
# 4. ORCHESTRATOR
# ==========================================

def assess_rotation(segment_masks: dict[str, np.ndarray], img_shape: tuple[int, int], projection: str = "PA") -> dict[str, Any]:
    # Tier 0: Anatomic Sanity
    is_sane, sanity_msg = passes_anatomic_sanity(segment_masks, img_shape)
    if not is_sane:
        return {"status": RotationClass.ERROR, "metrics": {}, "reasoning": sanity_msg}

    # Tier 1 & 2
    parallax = calculate_parallax_offset(segment_masks, img_shape)
    symmetry = calculate_lung_symmetry(segment_masks)
    
    # Consensus
    status, reasoning_str = resolve_rotation_consensus(parallax, symmetry, projection)
    
    metrics: dict[str, Any] = {
        "projection": projection,
        "parallax_ratio": parallax["offset_ratio"],
        "lung_area_skew": symmetry["area_ratio"],
        "visuals": parallax 
    }

    return {"status": status, "metrics": metrics, "reasoning": reasoning_str}