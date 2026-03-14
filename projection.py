import numpy as np
import cv2

# ==========================================
# --- PROJECTION THRESHOLDS & CONSTANTS ---
# ==========================================

MAX_FRONTAL_SKELETAL_OFFSET: float = 0.15 
FOV_MARGIN_PX: int = 5
AP_PA_CLAVICLE_THRESHOLD: float = 0.02 

class ProjectionClass:
    FRONTAL = "frontal"
    LATERAL = "lateral"
    ERROR   = "error"
    AP      = "AP"
    PA      = "PA"

# ==========================================
# 1. THE FRONTAL FILTER
# ==========================================

def is_frontal_view(masks: dict, img_shape: tuple) -> tuple[bool, str]:
    """Determines if the view is Frontal or Lateral based on Spine-Sternum alignment."""
    h, w = img_shape[:2]
    
    sp = masks.get("Spine")
    st = masks.get("Sternum")

    # Explicit None checks for Pylance
    if sp is None or st is None:
        return False, "Skeletal landmarks missing"
    
    # Explicit pixel presence check
    if not np.any(sp) or not np.any(st):
        return False, "Skeletal landmarks empty"

    def get_centroid_x(mask: np.ndarray) -> int:
        M = cv2.moments((mask > 0).astype(np.uint8))
        return int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0

    sp_x = get_centroid_x(sp)
    st_x = get_centroid_x(st)
    
    offset_ratio = float(abs(sp_x - st_x) / w)

    if offset_ratio > MAX_FRONTAL_SKELETAL_OFFSET:
        return False, f"Lateral View Detected: Skeletal offset {offset_ratio:.1%}"

    return True, "Frontal View Confirmed"

# ==========================================
# 2. ANATOMIC COMPLETENESS (FOV GATE)
# ==========================================

def get_fov_completeness(masks: dict, img_shape: tuple) -> dict:
    """Checks for lung clipping at image boundaries."""
    h, w = img_shape[:2]
    lr = masks.get("Lung_Right")
    ll = masks.get("Lung_Left")

    if lr is None or ll is None:
        return {"complete": False, "clipping": [], "reason": "Missing lung masks"}
    
    if not np.any(lr) or not np.any(ll):
        return {"complete": False, "clipping": [], "reason": "Empty lung masks"}

    # Calculate combined bounding box
    combined = ((lr > 0) | (ll > 0))
    y, x = np.where(combined)
    
    y_min, y_max = int(np.min(y)), int(np.max(y))
    x_min, x_max = int(np.min(x)), int(np.max(x))

    clipping = []
    if y_min <= FOV_MARGIN_PX: clipping.append("Apices (Top)")
    if y_max >= (h - FOV_MARGIN_PX): clipping.append("Bases (Bottom)")
    if x_min <= FOV_MARGIN_PX or x_max >= (w - FOV_MARGIN_PX): clipping.append("Lateral Ribs")

    return {
        "complete": len(clipping) == 0,
        "clipping": clipping,
        "reason": f"Clipped: {', '.join(clipping)}" if clipping else "Complete"
    }

# ==========================================
# 3. PROJECTION ENGINE (AP VS PA)
# ==========================================

def determine_ap_pa_projection(masks: dict) -> tuple[str, float]:
    """Heuristic logic to distinguish AP from PA via Clavicle-Apex relationship."""
    lr, ll = masks.get("Lung_Right"), masks.get("Lung_Left")
    cr, cl = masks.get("Clavicle_Right"), masks.get("Clavicle_Left")
    
    # Strict None/Empty checking to satisfy Pylance
    if lr is None or ll is None or cr is None or cl is None:
        return "Unknown", 0.0
    
    if not all(np.any(m) for m in [lr, ll, cr, cl]):
        return "Unknown", 0.0

    # Get Lung Apex (highest point)
    y_apex_r = int(np.min(np.where(lr > 0)[0]))
    y_apex_l = int(np.min(np.where(ll > 0)[0]))
    y_apex = min(y_apex_r, y_apex_l)
    
    # Get Lung Base (lowest point)
    y_base_r = int(np.max(np.where(lr > 0)[0]))
    y_base_l = int(np.max(np.where(ll > 0)[0]))
    y_base = max(y_base_r, y_base_l)
    
    # Get Medial Clavicle Heights
    y_clav_r = float(np.mean(np.where(cr > 0)[0]))
    y_clav_l = float(np.mean(np.where(cl > 0)[0]))
    y_clav_avg = (y_clav_r + y_clav_l) / 2.0
    
    lung_height = float(y_base - y_apex)
    if lung_height <= 0:
        return "Unknown", 0.0

    rel_height = float((y_clav_avg - y_apex) / lung_height)
    
    # AP: Clavicles are high (closer to or above apex)
    # PA: Clavicles are projected lower into the lungs
    proj = ProjectionClass.AP if rel_height < AP_PA_CLAVICLE_THRESHOLD else ProjectionClass.PA
    
    return str(proj), rel_height

# ==========================================
# 4. ORCHESTRATOR
# ==========================================

def assess_projection(segment_masks: dict, img_shape: tuple) -> dict:
    """Final Quality Gate for Projection and Completeness."""
    
    # 1. Frontal Filter
    is_frontal, frontal_msg = is_frontal_view(segment_masks, img_shape)
    if not is_frontal:
        return {
            "status": ProjectionClass.LATERAL if "Lateral" in frontal_msg else ProjectionClass.ERROR,
            "view": ProjectionClass.LATERAL,
            "is_frontal": False,
            "metrics": {},
            "reasoning": frontal_msg
        }

    # 2. Anatomic Gates
    fov_data = get_fov_completeness(segment_masks, img_shape)
    proj_type, rel_h = determine_ap_pa_projection(segment_masks)
    
    metrics = {
        "is_frontal": True,
        "projection": proj_type,
        "clavicle_apex_ratio": rel_h,
        "clipping_list": fov_data.get("clipping", []),
        "is_complete": bool(fov_data["complete"])
    }

    status = ProjectionClass.FRONTAL if fov_data["complete"] else "frontal_incomplete"
    reasoning = f"Confirmed Frontal ({proj_type}). "
    reasoning += "FOV Complete." if fov_data["complete"] else fov_data["reason"]

    return {
        "status": status,
        "view": proj_type,
        "is_frontal": True,
        "metrics": metrics,
        "reasoning": reasoning
    }