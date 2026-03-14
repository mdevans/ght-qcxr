import numpy as np
import cv2

# ==========================================
# --- INSPIRATION THRESHOLDS & CONSTANTS ---
# ==========================================

# --- Skeletal Engine (Ribs) ---
THRESHOLD_RIB_VISIBILITY: float = 0.75  # % of rib that must be clear of diaphragm mask
THRESHOLD_MAX_RIB_GAP: int = 2          # Max numerical jump between valid rib indices
THRESHOLD_MIN_RELIABLE_RIBS: int = 5    # Minimum trusted segments required for reliable skeletal vote
SKELETAL_GAP_WARNING_THRESHOLD: int = 2 # Missing ribs count that triggers a reasoning warning

# --- Structural Trust (Integrity) ---
# A rib must span at least this % of the Right Lung's width to be trusted.
THRESHOLD_MIN_RIB_WIDTH_RATIO: float = 0.35
# Ribs 1-3 (Proximal) are anatomically smaller; we apply this leniency multiplier.
RIB_PROXIMAL_LENIENCY_MULTIPLIER: float = 0.6
INDEX_PROXIMAL_END: int = 3

# --- Projection-Aware Expansion Levels (Rib Count) ---
PA_EXPANSION_POOR_THRESHOLD: int = 9
PA_EXPANSION_OVER_THRESHOLD: int = 10
AP_EXPANSION_POOR_THRESHOLD: int = 7
AP_EXPANSION_OVER_THRESHOLD: int = 9

# --- Geometric Thresholds (Lung Field Aspect Ratio H/W) ---
ASPECT_RATIO_POOR_EXPANSION: float = 0.85
ASPECT_RATIO_OVER_EXPANSION: float = 1.05

# --- Anatomic Sanity Constraints ---
MIN_REQUIRED_LUNG_AREA_RATIO: float = 0.15   # Lungs must occupy > 15% of image
MIN_DIAPHRAGM_HEIGHT_RATIO: float = 0.30     # Diaphragm dome cannot be in top 30% of image

# ==========================================

class InspirationClass:
    POOR = "reject_poor"
    NORMAL = "pass"
    OVER = "reject_over"
    ERROR = "error"

# ==========================================
# 1. ANATOMIC SANITY CHECKS
# ==========================================

def passes_anatomic_sanity(segment_masks: dict[str, np.ndarray], img_shape: tuple[int, int]) -> tuple[bool, str]:
    """Validates fundamental human geometry before performing QA math."""
    h, w = img_shape[:2]
    
    diaphragm = segment_masks.get("Diaphragm_Full")
    if diaphragm is None or diaphragm.max() == 0:
        return False, "Sanity Failure: Diaphragm_Full mask is missing or empty."
        
    y_coords, _ = np.where(diaphragm > 0)
    dome_y = int(np.min(y_coords))
    if dome_y < (h * MIN_DIAPHRAGM_HEIGHT_RATIO):
        return False, f"Sanity Failure: Diaphragm dome (y={dome_y}) is too high."

    r_lung = segment_masks.get("Lung_Right")
    l_lung = segment_masks.get("Lung_Left")
    valid_lungs = [m for m in (r_lung, l_lung) if m is not None and m.max() > 0]
    
    if not valid_lungs:
        return False, "Sanity Failure: No lung masks detected."
        
    total_lung_area = sum(np.sum(m > 0) for m in valid_lungs)
    if total_lung_area < (h * w * MIN_REQUIRED_LUNG_AREA_RATIO):
        return False, "Sanity Failure: Total lung area is impossibly small."

    return True, "Passed"

# ==========================================
# 2. ENGINE A: SKELETAL VALIDATOR (Ribs)
# ==========================================

def get_diaphragm_dome_y(diaphragm_mask: np.ndarray) -> int:
    """Finds the absolute highest y-coordinate (minimum y) of the diaphragm mask."""
    y_coords, _ = np.where(diaphragm_mask > 0)
    return int(np.min(y_coords)) if len(y_coords) > 0 else 0

def is_rib_structurally_sound(rib_mask: np.ndarray, lung_w: int, rib_index: int) -> bool:
    """Verifies rib dimensions relative to lung width to filter DL hallucinations."""
    if lung_w <= 0: return False
    
    _, _, rib_w, _ = cv2.boundingRect((rib_mask > 0).astype(np.uint8))
    
    # Apply anatomical leniency for proximal ribs
    multiplier = RIB_PROXIMAL_LENIENCY_MULTIPLIER if rib_index <= INDEX_PROXIMAL_END else 1.0
    required_width = lung_w * THRESHOLD_MIN_RIB_WIDTH_RATIO * multiplier
    
    return rib_w >= required_width

def calculate_lowest_visible_rib(
    blended_dict: dict[str, np.ndarray], 
    diaphragm_mask: np.ndarray, 
    img_shape: tuple[int, int],
    threshold: float = THRESHOLD_RIB_VISIBILITY
) -> tuple[int, bool, int]:
    """Determines expansion level via 2D anatomical overlap with the diaphragm."""
    lowest_idx = 0
    trusted_count = 0
    
    r_lung = blended_dict.get("Lung_Right")
    lung_w = 0
    if r_lung is not None and r_lung.max() > 0:
        _, _, lung_w, _ = cv2.boundingRect((r_lung > 0).astype(np.uint8))
    
    for i in range(1, 13):
        rib_mask = blended_dict.get(f"Posterior_Rib_{i}_Right")
        if rib_mask is None or rib_mask.max() == 0:
            continue
            
        if not is_rib_structurally_sound(rib_mask, lung_w, i):
            continue
            
        trusted_count += 1
        
        # Guard against broadcasting errors in heterogeneous mask environments
        if rib_mask.shape != diaphragm_mask.shape:
             curr_rib = cv2.resize(rib_mask, (diaphragm_mask.shape[1], diaphragm_mask.shape[0]))
        else:
             curr_rib = rib_mask

        rib_pixels_total = np.sum(curr_rib > 0)
        # Visible = Rib pixels not masked by 2D diaphragm anatomy
        visible_mask = np.logical_and(curr_rib > 0, diaphragm_mask == 0)
        pixels_visible = np.sum(visible_mask)
        
        visibility_ratio = pixels_visible / rib_pixels_total if rib_pixels_total > 0 else 0
        
        if visibility_ratio >= threshold:
            if lowest_idx == 0 or (i - lowest_idx) <= THRESHOLD_MAX_RIB_GAP:
                lowest_idx = i
            
    is_reliable = trusted_count >= THRESHOLD_MIN_RELIABLE_RIBS
    return lowest_idx, is_reliable, trusted_count

# ==========================================
# 3. ENGINE B: GEOMETRIC VALIDATOR (Lungs)
# ==========================================

def calculate_lung_aspect_ratio(segment_masks: dict[str, np.ndarray]) -> float:
    """Measures Height / Width of the combined lung field bounding box."""
    r_lung, l_lung = segment_masks.get("Lung_Right"), segment_masks.get("Lung_Left")
    valid_lungs = [m for m in (r_lung, l_lung) if m is not None and m.max() > 0]
    
    if not valid_lungs:
        return 0.0
        
    combined = valid_lungs[0] > 0 if len(valid_lungs) == 1 else np.logical_or(valid_lungs[0] > 0, valid_lungs[1] > 0)
    y_coords, x_coords = np.where(combined)
    
    if len(y_coords) == 0:
        return 0.0
        
    height = np.max(y_coords) - np.min(y_coords)
    width = np.max(x_coords) - np.min(x_coords)
    
    return float(height / width) if width > 0 else 0.0

# ==========================================
# 4. ORCHESTRATOR
# ==========================================

def assess_inspiration(segment_masks: dict[str, np.ndarray], img_shape: tuple[int, int], projection: str = "PA") -> dict:
    """Ensemble assessment using Skeletal and Geometric engines."""
    is_sane, sanity_msg = passes_anatomic_sanity(segment_masks, img_shape)
    if not is_sane:
        return {"status": InspirationClass.ERROR, "metrics": {}, "reasoning": sanity_msg}

    diaphragm_mask = segment_masks["Diaphragm_Full"]
    rib_level, skeletal_reliable, trusted_count = calculate_lowest_visible_rib(
        segment_masks, diaphragm_mask, img_shape
    )
    aspect_ratio = calculate_lung_aspect_ratio(segment_masks)

    # Threshold Mapping
    poor_rib_thresh = PA_EXPANSION_POOR_THRESHOLD if projection == "PA" else AP_EXPANSION_POOR_THRESHOLD
    over_rib_thresh = PA_EXPANSION_OVER_THRESHOLD if projection == "PA" else AP_EXPANSION_OVER_THRESHOLD

    status = InspirationClass.NORMAL
    reasoning = []

    # Detection Gaps
    missing_rib_gap = rib_level - trusted_count
    if skeletal_reliable and missing_rib_gap >= SKELETAL_GAP_WARNING_THRESHOLD:
        reasoning.append(f"Note: Skeletal gaps detected ({missing_rib_gap} ribs missing/fragmented).")

    if not skeletal_reliable:
        reasoning.append(f"Skeletal tracking unstable (found only {trusted_count} trusted right ribs).")
        if aspect_ratio < ASPECT_RATIO_POOR_EXPANSION:
            status = InspirationClass.POOR
            reasoning.append(f"Lung Aspect Ratio ({aspect_ratio:.2f}) indicates poor expansion.")
        elif aspect_ratio > ASPECT_RATIO_OVER_EXPANSION:
            status = InspirationClass.OVER
            reasoning.append(f"Lung Aspect Ratio ({aspect_ratio:.2f}) indicates hyperinflation.")
    else:
        # Standard Consensus logic
        skeletal_vote = InspirationClass.NORMAL
        if rib_level < poor_rib_thresh:
            skeletal_vote = InspirationClass.POOR
        elif rib_level > over_rib_thresh:
            skeletal_vote = InspirationClass.OVER
            
        geometric_vote = InspirationClass.NORMAL
        if aspect_ratio < ASPECT_RATIO_POOR_EXPANSION:
            geometric_vote = InspirationClass.POOR
        elif aspect_ratio > ASPECT_RATIO_OVER_EXPANSION:
            geometric_vote = InspirationClass.OVER

        if skeletal_vote == geometric_vote:
            status = skeletal_vote
            reasoning.append(f"Unanimous {status.upper()}: Expansion level {rib_level}, Ratio {aspect_ratio:.2f}.")
        elif skeletal_vote == InspirationClass.NORMAL and geometric_vote == InspirationClass.OVER:
            status = InspirationClass.OVER
            reasoning.append(f"Hyperinflation suspected: Normal expansion ({rib_level}) but high ratio ({aspect_ratio:.2f}).")
        elif skeletal_vote == InspirationClass.POOR and geometric_vote == InspirationClass.NORMAL:
            status = InspirationClass.NORMAL
            reasoning.append(f"Low expansion level ({rib_level}) overridden by normal lung geometry ({aspect_ratio:.2f}).")
        else:
            status = skeletal_vote
            reasoning.append(f"Conflict: Ribs ({rib_level}) vs Geometry ({aspect_ratio:.2f}). Defaulting to Skeletal.")

    return {
        "status": status,
        "metrics": {
            "projection": projection,
            "diaphragm_dome_y": get_diaphragm_dome_y(diaphragm_mask),
            "expansion_level_rib": rib_level,
            "trusted_rib_count": trusted_count,
            "skeletal_reliable": skeletal_reliable,
            "lung_aspect_ratio": aspect_ratio
        },
        "reasoning": " ".join(reasoning)
    }