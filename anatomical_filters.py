import numpy as np
import cv2

def _keep_largest_blob(mask: np.ndarray) -> np.ndarray:
    """Helper: Removes errant pixels by keeping only the largest connected component."""
    if mask is None or mask.max() == 0:
        return mask
        
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
        
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    return clean_mask

def filter_lungs(masks: dict):
    """Lungs are our primary anchors. Ensure they are solid, contiguous blocks."""
    for side in ["Lung_Right", "Lung_Left"]:
        if side in masks and masks[side] is not None:
            masks[side] = _keep_largest_blob(masks[side])

def filter_spine(masks: dict):
    """Spine models frequently hallucinate small floating blobs. Keep the main column."""
    if "Spine" in masks and masks["Spine"] is not None:
        masks["Spine"] = _keep_largest_blob(masks["Spine"])

def filter_diaphragm(masks: dict):
    """Rule: Delete any diaphragm pixels located above the vertical midpoint of the corresponding lung."""
    for side in ["Right", "Left"]:
        lung_key = f"Lung_{side}"
        diaph_key = f"Diaphragm_{side}"
        
        lung = masks.get(lung_key)
        diaph = masks.get(diaph_key)
        
        if lung is not None and diaph is not None and diaph.max() > 0:
            y_coords, _ = np.where(lung > 0)
            if len(y_coords) > 0:
                lung_midpoint_y = int(np.min(y_coords) + (np.ptp(y_coords) / 2))
                clean_diaph = diaph.copy()
                clean_diaph[:lung_midpoint_y, :] = 0 
                masks[diaph_key] = _keep_largest_blob(clean_diaph)

def filter_clavicles(masks: dict):
    """Rule: Delete any clavicle pixels located below the top 40% of the corresponding lung."""
    for side in ["Right", "Left"]:
        lung_key = f"Lung_{side}"
        clav_key = f"Clavicle_{side}"
        
        lung = masks.get(lung_key)
        clavicle = masks.get(clav_key)
        
        if lung is not None and clavicle is not None and clavicle.max() > 0:
            y_coords, _ = np.where(lung > 0)
            if len(y_coords) > 0:
                lung_top_y = np.min(y_coords)
                lung_height = np.ptp(y_coords)
                cutoff_y = int(lung_top_y + (lung_height * 0.40))
                
                clean_clav = clavicle.copy()
                clean_clav[cutoff_y:, :] = 0
                masks[clav_key] = _keep_largest_blob(clean_clav)

def filter_ribs(masks: dict):
    """Rule: Keep the largest blob per rib, preventing fragmented noise."""
    rib_keys = [k for k in masks.keys() if "Rib" in k]
    for key in rib_keys:
        if masks[key] is not None:
            masks[key] = _keep_largest_blob(masks[key])

def apply_anatomical_filters(blended_dict: dict) -> tuple[dict, list[str]]:
    """
    Master function that runs the discrete anatomical rules in strict dependency order.
    """
    filtered_dict = blended_dict.copy()
    audit_trail = []
    
    # 1. Clean the primary anchors first
    filter_lungs(filtered_dict)
    filter_spine(filtered_dict)
    
    # 2. Use the clean anchors to validate secondary structures
    filter_diaphragm(filtered_dict)
    filter_clavicles(filtered_dict)
    filter_ribs(filtered_dict)
    
    # 3. Generate the Audit Trail by comparing before and after
    for target_name in blended_dict.keys():
        orig_mask = blended_dict[target_name]
        new_mask = filtered_dict.get(target_name)
        
        orig_pixels = int(np.sum(orig_mask > 0)) if orig_mask is not None else 0
        new_pixels = int(np.sum(new_mask > 0)) if new_mask is not None else 0
        
        if orig_pixels != new_pixels:
            removed = orig_pixels - new_pixels
            percent = (removed / orig_pixels) * 100 if orig_pixels > 0 else 0
            audit_trail.append(f"FILTERED [{target_name}]: Removed {removed} pixels ({percent:.1f}%).")
            
    return filtered_dict, audit_trail