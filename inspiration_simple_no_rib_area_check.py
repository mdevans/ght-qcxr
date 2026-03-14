import numpy as np

class InspirationClass:
    POOR = "reject_poor"
    NORMAL = "pass"
    OVER = "reject_over"
    ERROR = "error"

# ==========================================
# 1. ANATOMIC SANITY CHECKS
# ==========================================

def passes_anatomic_sanity(segment_masks: dict[str, np.ndarray], img_shape: tuple[int, int]) -> tuple[bool, str]:
    """Validates that the inspiration algorithm relevant segments obey fundamental human geometry before doing math."""
    h, w = img_shape[:2]
    
    # Check 1: Do we have a diaphragm?
    diaphragm = segment_masks.get("Diaphragm_Full")
    if diaphragm is None or diaphragm.max() == 0:
        return False, "Sanity Failure: Diaphragm_Full mask is missing or empty."
        
    # Check 2: Is the diaphragm physically in the bottom half of the image?
    y_coords, _ = np.where(diaphragm > 0)
    dome_y = int(np.min(y_coords))
    if dome_y < (h * 0.30):
        return False, f"Sanity Failure: Diaphragm dome (y={dome_y}) is impossibly high in the chest cavity."

    # Check 3: Do we have sufficient lung tissue?
    r_lung = segment_masks.get("Lung_Right")
    l_lung = segment_masks.get("Lung_Left")
    valid_lungs = [m for m in (r_lung, l_lung) if m is not None and m.max() > 0]
    
    if not valid_lungs:
        return False, "Sanity Failure: No lung masks detected."
        
    total_lung_area = sum(np.sum(m > 0) for m in valid_lungs)
    if total_lung_area < (h * w * 0.15): # Lungs must take up at least 15% of the image
        return False, f"Sanity Failure: Total lung area is impossibly small ({total_lung_area} px)."

    return True, "Passed"

# ==========================================
# 2. ENGINE A: SKELETAL VALIDATOR (Ribs)
# ==========================================

def get_diaphragm_dome_y(diaphragm_mask: np.ndarray) -> int:
    """Finds the absolute highest y-coordinate (minimum y) of the diaphragm mask."""
    y_coords, _ = np.where(diaphragm_mask > 0)
    return int(np.min(y_coords)) if len(y_coords) > 0 else 0

def calculate_lowest_visible_rib(blended_dict: dict[str, np.ndarray], dome_y: int, threshold: float = 0.75) -> tuple[int, bool]:
    """
    Returns the index (1-12) of the lowest right posterior rib that is primarily 
    above the diaphragm. Returns a boolean indicating if tracking was reliable.
    """
    highest_idx = 0
    valid_ribs_found = 0
    
    for i in range(1, 13):
        rib_mask = blended_dict.get(f"Posterior_Rib_{i}_Right")
        if rib_mask is None or rib_mask.max() == 0:
            continue
            
        valid_ribs_found += 1
        y_coords, _ = np.where(rib_mask > 0)
        
        # Count pixels above the dome (smaller Y values)
        pixels_above_dome = np.sum(y_coords < dome_y)
        visibility_ratio = pixels_above_dome / len(y_coords)
        
        if visibility_ratio >= threshold:
            highest_idx = max(highest_idx, i)
            
    # Reliability constraint: CXAS must have successfully mapped at least 5 right ribs
    is_reliable = valid_ribs_found >= 5
    return highest_idx, is_reliable

# ==========================================
# 3. ENGINE B: GEOMETRIC VALIDATOR (Lungs)
# ==========================================

def calculate_lung_aspect_ratio(segment_masks: dict[str, np.ndarray]) -> float:
    """Calculates Max Height / Max Width of the combined lung bounding box."""
    r_lung = segment_masks.get("Lung_Right")
    l_lung = segment_masks.get("Lung_Left")
    valid_lungs = [m for m in (r_lung, l_lung) if m is not None]
    
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
    """Dual-Engine evaluation of radiographic inspiration."""
    
    # 1. Anatomic Sanity Gate
    is_sane, sanity_msg = passes_anatomic_sanity(segment_masks, img_shape)
    if not is_sane:
        return {
            "status": InspirationClass.ERROR,
            "metrics": {},
            "reasoning": sanity_msg
        }

    # 2. Execute Engines
    dome_y = get_diaphragm_dome_y(segment_masks["Diaphragm_Full"])
    rib_level, skeletal_reliable = calculate_lowest_visible_rib(segment_masks, dome_y)
    aspect_ratio = calculate_lung_aspect_ratio(segment_masks)

    # 3. Projection-Aware Thresholds
    if projection == "PA":
        poor_rib_thresh = 9
        over_rib_thresh = 10
    else: # AP Projection (Supine / Portable)
        poor_rib_thresh = 7
        over_rib_thresh = 9

    # Geometric Thresholds (Highly stable, scale-invariant)
    poor_aspect_thresh = 0.85 # Short and wide
    over_aspect_thresh = 1.05 # Long and narrow

    # 4. Ensemble Voting Logic
    status = InspirationClass.NORMAL
    reasoning = []

    # If Skeletal tracking failed completely, rely entirely on Geometry
    if not skeletal_reliable:
        reasoning.append(f"Skeletal tracking unstable (found fewer than 5 right ribs). Falling back to geometric analysis.")
        if aspect_ratio < poor_aspect_thresh:
            status = InspirationClass.POOR
            reasoning.append(f"Lung Aspect Ratio ({aspect_ratio:.2f}) indicates poor vertical expansion.")
        elif aspect_ratio > over_aspect_thresh:
            status = InspirationClass.OVER
            reasoning.append(f"Lung Aspect Ratio ({aspect_ratio:.2f}) indicates abnormal hyperinflation.")
        else:
            reasoning.append(f"Lung Aspect Ratio ({aspect_ratio:.2f}) indicates normal expansion.")
            
    # Standard Dual-Engine Voting
    else:
        skeletal_vote = InspirationClass.NORMAL
        if rib_level < poor_rib_thresh:
            skeletal_vote = InspirationClass.POOR
        elif rib_level > over_rib_thresh:
            skeletal_vote = InspirationClass.OVER
            
        geometric_vote = InspirationClass.NORMAL
        if aspect_ratio < poor_aspect_thresh:
            geometric_vote = InspirationClass.POOR
        elif aspect_ratio > over_aspect_thresh:
            geometric_vote = InspirationClass.OVER

        # Consensus Matrix
        if skeletal_vote == geometric_vote:
            status = skeletal_vote
            if status == InspirationClass.NORMAL:
                reasoning.append(f"Adequate {projection} inspiration. {rib_level} posterior ribs visible. Normal lung geometry.")
            else:
                reasoning.append(f"Unanimous {status.upper()}: {rib_level} ribs visible with supporting aspect ratio of {aspect_ratio:.2f}.")
        
        # Conflict: Ribs say Normal, but Lungs are massively hyperinflated (Pathology Catch)
        elif skeletal_vote == InspirationClass.NORMAL and geometric_vote == InspirationClass.OVER:
            status = InspirationClass.OVER
            reasoning.append(f"WARNING: Rib count is normal ({rib_level}), but lung aspect ratio ({aspect_ratio:.2f}) suggests severe structural hyperinflation (e.g., COPD).")
            
        # Conflict: Ribs say Poor, but Lungs look normally expanded
        elif skeletal_vote == InspirationClass.POOR and geometric_vote == InspirationClass.NORMAL:
            status = InspirationClass.NORMAL
            reasoning.append(f"Rib count is low ({rib_level}), but normal lung geometry ({aspect_ratio:.2f}) overrides, confirming adequate volumetric expansion.")
            
        else:
            # Fallback for complex conflicts: Trust the ribs if we are confident
            status = skeletal_vote
            reasoning.append(f"Metrics conflict (Ribs: {rib_level}, Aspect: {aspect_ratio:.2f}). Defaulting to skeletal standard.")

    return {
        "status": status,
        "metrics": {
            "projection": projection,
            "diaphragm_dome_y": dome_y,
            "lowest_visible_rib": rib_level,
            "skeletal_reliable": skeletal_reliable,
            "lung_aspect_ratio": aspect_ratio
        },
        "reasoning": " ".join(reasoning)
    }