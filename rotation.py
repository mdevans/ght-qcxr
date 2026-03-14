import numpy as np
import cv2

# ==========================================
# --- ROTATION THRESHOLDS & CONSTANTS ---
# ==========================================

THRESHOLD_PARALLAX_PA_PASS: float = 0.03 
THRESHOLD_PARALLAX_AP_PASS: float = 0.05 
THRESHOLD_LUNG_CONTOUR_SKEW: float = 0.08 
THRESHOLD_CLAVICLE_SKEW: float = 0.02 

# --- Modular Anatomic Constraints ---
MIN_SKELETAL_AREA_RATIO: float = 1.5   
MEDIASTINAL_CORRIDOR_RATIO: float = 0.3 
MIN_LUNG_SKELETAL_AREA_RATIO: float = 2.0 
MIN_CLAVICLE_TO_LUNG_AREA_RATIO: float = 0.01 

class RotationClass:
    PASS = "pass"
    REJECT = "reject_rotated"
    ERROR = "error"

# ==========================================
# 0. MODULAR ANATOMIC VALIDATORS
# ==========================================

def is_parallax_sane(masks: dict, img_shape: tuple) -> tuple:
    h, w = img_shape[:2]
    mid_x, corridor = w // 2, w * MEDIASTINAL_CORRIDOR_RATIO
    
    sp = masks.get("Spine")
    st = masks.get("Sternum")

    if sp is None or st is None:
        return False, "Landmarks missing"
    if not np.any(sp) or not np.any(st):
        return False, "Landmarks empty"

    area_sp = int((sp > 0).sum())
    area_st = int((st > 0).sum())
    
    if area_sp < (area_st * MIN_SKELETAL_AREA_RATIO):
        return False, "Improbable Spine-Sternum area ratio"

    def get_x(m):
        M = cv2.moments((m > 0).astype(np.uint8))
        return int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0

    if abs(get_x(sp) - mid_x) > corridor or abs(get_x(st) - mid_x) > corridor:
        return False, "Mediastinal corridor violation"

    return True, "Sane"

def is_symmetry_sane(masks: dict) -> tuple:
    lr, ll = masks.get("Lung_Right"), masks.get("Lung_Left")
    if lr is None or ll is None:
        return False, "Lung landmarks missing"
    if not np.any(lr) or not np.any(ll):
        return False, "Lung landmarks empty"
    return True, "Sane"

def is_clavicle_sane(masks: dict, img_shape: tuple) -> tuple:
    cr, cl = masks.get("Clavicle_Right"), masks.get("Clavicle_Left")
    lr, ll = masks.get("Lung_Right"), masks.get("Lung_Left")
    sp = masks.get("Spine")

    if cr is None or cl is None or lr is None or ll is None or sp is None:
        return False, "Missing mandatory segments"
    if not np.any(cr) or not np.any(cl) or not np.any(lr) or not np.any(ll) or not np.any(sp):
        return False, "Landmarks empty"
    
    total_lung_area = int((lr > 0).sum()) + int((ll > 0).sum())
    if (int((cr > 0).sum()) / total_lung_area) < MIN_CLAVICLE_TO_LUNG_AREA_RATIO:
        return False, "Right clavicle fragmented"
    if (int((cl > 0).sum()) / total_lung_area) < MIN_CLAVICLE_TO_LUNG_AREA_RATIO:
        return False, "Left clavicle fragmented"

    _, xr = np.where(cr > 0); tip_r = np.max(xr)
    _, xl = np.where(cl > 0); tip_l = np.min(xl)
    if tip_r >= tip_l:
        return False, "Clavicle Crossover detected"

    return True, "Sane"

# ==========================================
# 1. ENGINES (MATH ONLY)
# ==========================================

def run_parallax_engine(masks: dict, img_shape: tuple, projection: str) -> dict:
    h, w = img_shape[:2]
    sp, st = masks["Spine"], masks["Sternum"]

    def get_axis(m):
        y, x = np.where(m > 0)
        pts = np.column_stack((x, y)).astype(np.float32)
        return cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    vx_sp, vy_sp, x0_sp, y0_sp = get_axis(sp)
    vx_st, vy_st, x0_st, y0_st = get_axis(st)
    
    spine_x = x0_sp if abs(vy_sp) < 1e-6 else x0_sp + ((y0_st - y0_sp) / vy_sp) * vx_sp
    ratio = float((spine_x - x0_st) / w)
    
    limit = THRESHOLD_PARALLAX_PA_PASS if projection == "PA" else THRESHOLD_PARALLAX_AP_PASS
    status = RotationClass.PASS if abs(ratio) <= limit else RotationClass.REJECT
    
    return {
        "status": status, "value": ratio, 
        "visuals": {"spine_line": (vx_sp, vy_sp, x0_sp, y0_sp), "sternum_line": (vx_st, vy_st, x0_st, y0_st), "eval_pt": (int(spine_x), int(x0_st), int(y0_st))}
    }

def run_symmetry_engine(masks: dict) -> dict:
    lr, ll = masks["Lung_Right"], masks["Lung_Left"]
    
    def get_perimeter(m):
        contours, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.arcLength(contours[0], True) if contours else 0.0

    pr, pl = get_perimeter(lr), get_perimeter(ll)
    skew = float(abs(pr - pl) / (pr + pl)) if (pr + pl) > 0 else 0.0
    return {"status": RotationClass.PASS if skew <= THRESHOLD_LUNG_CONTOUR_SKEW else RotationClass.REJECT, "value": skew}

def run_clavicle_engine(masks: dict, img_shape: tuple) -> dict:
    cr, cl, sp = masks["Clavicle_Right"], masks["Clavicle_Left"], masks["Spine"]

    def get_medial_pt(m, side):
        kernel = np.ones((5,5), np.uint8) 
        eroded = cv2.erode((m > 0).astype(np.uint8), kernel, iterations=1)
        y, x = np.where(eroded > 0 if np.any(eroded) else m > 0)
        return (x[np.argmax(x)], y[np.argmax(x)]) if side == "R" else (x[np.argmin(x)], y[np.argmin(x)])

    (xr, yr), (xl, yl) = get_medial_pt(cr, "R"), get_medial_pt(cl, "L")
    y_sp, x_sp = np.where(sp > 0)
    vx, vy, x0, y0 = cv2.fitLine(np.column_stack((x_sp, y_sp)).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    
    eval_y = (yr + yl) / 2
    spine_x = x0 if abs(vy) < 1e-6 else x0 + ((eval_y - y0) / vy) * vx
    skew = float(abs(abs(spine_x - xr) - abs(xl - spine_x)) / img_shape[1])
    
    return {"status": RotationClass.PASS if skew <= THRESHOLD_CLAVICLE_SKEW else RotationClass.REJECT, "value": skew}

# ==========================================
# 2. HIERARCHICAL CONSENSUS
# ==========================================

def resolve_consensus(active_votes: dict) -> tuple:
    parallax = active_votes.get("parallax")
    
    # Map internal keys to display names for reasoning strings
    display_names = {"parallax": "Parallax", "symmetry": "Lung Contour", "clavicle": "Clavicle"}
    summary = " | ".join([f"{display_names.get(k, k)}: {v['status'].upper()} ({v['value']:.1%})" 
                         for k, v in active_votes.items()])

    if parallax:
        status = parallax["status"]
        return status, f"Skeletal Core Override: {status.upper()} [{summary}]"

    sym, clav = active_votes.get("symmetry"), active_votes.get("clavicle")
    if sym and clav:
        if sym["status"] == RotationClass.PASS and clav["status"] == RotationClass.PASS:
            return RotationClass.PASS, f"Dual Backup PASS [{summary}]"
        return RotationClass.REJECT, f"Dual Backup REJECT (Unanimous required) [{summary}]"

    return RotationClass.ERROR, f"Insufficient backup engines [{summary}]"

# ==========================================
# 3. ORCHESTRATOR
# ==========================================

def assess_rotation(segment_masks: dict, img_shape: tuple, projection: str = "PA") -> dict:
    # 1. Modular Engine Factory
    votes = {}
    
    if is_parallax_sane(segment_masks, img_shape)[0]:
        votes["parallax"] = run_parallax_engine(segment_masks, img_shape, projection)
    
    if is_symmetry_sane(segment_masks)[0]:
        votes["symmetry"] = run_symmetry_engine(segment_masks)
        
    if is_clavicle_sane(segment_masks, img_shape)[0]:
        votes["clavicle"] = run_clavicle_engine(segment_masks, img_shape)

    # 2. Total Failure Check
    if not votes:
        return {"status": RotationClass.ERROR, "metrics": {}, "reasoning": "Anatomic Sanity Failure: All segments fragmented or missing."}

    # 3. Consensus & Metrics Packaging
    status, reasoning = resolve_consensus(votes)
    
    metrics = {
        "projection": projection,
        "parallax_ratio": votes.get("parallax", {}).get("value"),
        "lung_contour_skew": votes.get("symmetry", {}).get("value"),
        "clavicle_skew": votes.get("clavicle", {}).get("value"),
        "visuals": votes.get("parallax", {}).get("visuals")
    }

    return {
        "status": status,
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "reasoning": reasoning
    }