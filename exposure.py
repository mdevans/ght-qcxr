import numpy as np
import cv2

# Import our strict type alias to guarantee the data contract
from preprocessing import GrayscaleImage

class ExposureClass:
    UNDER_EXPOSED = "under-exposed"
    NORMAL = "normal"
    OVER_EXPOSED = "over-exposed"

# ==========================================
# 0. LOCAL SANITY & FILTERING (Self-Reliance)
# ==========================================

def _keep_largest_blob(mask: np.ndarray | None) -> np.ndarray | None:
    """Removes AI hallucinations (snow) by strictly keeping only the largest contiguous mass."""
    if mask is None or mask.max() == 0:
        return mask
        
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask_uint8
        
    largest_contour = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask_uint8)
    cv2.drawContours(clean, [largest_contour], -1, 1, thickness=cv2.FILLED)
    return clean

# ==========================================
# 1. MATHEMATICAL HELPERS (Pure Extractors)
# ==========================================

def cxas_thoracic_vertebrae_segmented(segmented_masks: dict[str, np.ndarray]) -> int:
    """Counts how many individual thoracic vertebrae (T1-T12) the AI successfully mapped."""
    target_keys = {f"T{i}" for i in range(1, 13)}
    count = 0
    
    for key in target_keys:
        mask = segmented_masks.get(key)
        if mask is not None:
            # Sanity Check: Ensure we aren't counting a 5-pixel hallucination as a vertebra
            clean_mask = _keep_largest_blob(mask)
            if clean_mask is not None and clean_mask.max() > 0:
                count += 1
                
    return count

def heart_spine_penetration_variance(segmented_masks: dict[str, np.ndarray], gray_img: GrayscaleImage) -> float:
    """Calculates the pixel intensity variance strictly where the heart and spine overlap."""
    # Sanity Check: Clean the solid organs before trusting their overlap
    heart_mask = _keep_largest_blob(segmented_masks.get("Heart"))
    spine_mask = _keep_largest_blob(segmented_masks.get("Spine")) 
    
    if heart_mask is None or spine_mask is None or heart_mask.max() == 0 or spine_mask.max() == 0:
        return 0.0
        
    overlap_mask = np.logical_and(heart_mask > 0, spine_mask > 0)
    if not np.any(overlap_mask):
        return 0.0
        
    overlap_pixels = gray_img[overlap_mask]
    return float(np.var(overlap_pixels))

def lung_field_burnout(segmented_masks: dict[str, np.ndarray], gray_img: GrayscaleImage, black_threshold: int = 15) -> float:
    """Calculates the ratio of the lung field that is burned out (near absolute black)."""
    # Sanity Check: Clean the lungs to prevent extraneous noise from skewing the area ratio
    right_lung = _keep_largest_blob(segmented_masks.get("Lung_Right"))
    left_lung = _keep_largest_blob(segmented_masks.get("Lung_Left"))
    
    valid_masks = [m for m in (right_lung, left_lung) if m is not None and m.max() > 0]
    if not valid_masks:
        return 0.0
        
    combined_lungs = valid_masks[0] > 0 if len(valid_masks) == 1 else np.logical_or(valid_masks[0] > 0, valid_masks[1] > 0)
    lung_pixels = gray_img[combined_lungs]
    
    if len(lung_pixels) == 0:
        return 0.0
        
    black_pixels = np.sum(lung_pixels < black_threshold)
    return float(black_pixels / len(lung_pixels))

# ==========================================
# 2. DECISION LOGIC (Boolean Evaluators)
# ==========================================

def is_over_exposed(burnout_ratio: float, burnout_threshold: float = 0.30) -> bool:
    return burnout_ratio > burnout_threshold

def is_under_exposed(variance: float, vertebrae_count: int, min_variance: float = 50.0, min_vertebrae: int = 8) -> bool:
    return variance < min_variance or vertebrae_count < min_vertebrae

# ==========================================
# 3. PIPELINE ORCHESTRATOR
# ==========================================

def assess_exposure(segmented_masks: dict[str, np.ndarray], gray_img: GrayscaleImage) -> dict:
    """
    Executes the functional Exposure QA pipeline entirely in memory.
    Strictly assumes gray_img and segmented_masks are pre-aligned and formatted.
    """
    vertebrae_count = cxas_thoracic_vertebrae_segmented(segmented_masks)
    variance = heart_spine_penetration_variance(segmented_masks, gray_img)
    burnout_ratio = lung_field_burnout(segmented_masks, gray_img)

    over_exposed = is_over_exposed(burnout_ratio)
    under_exposed = is_under_exposed(variance, vertebrae_count)

    status = ExposureClass.NORMAL
    reasoning = "Image exhibits normal penetration and lung field contrast."

    if over_exposed and under_exposed:
        status = ExposureClass.UNDER_EXPOSED 
        reasoning = "Severe exposure failure: Lungs are burned out but mediastinum is under-penetrated."
    elif over_exposed:
        status = ExposureClass.OVER_EXPOSED
        reasoning = f"Over-penetrated: {burnout_ratio:.1%} of the lung field is burned out, obscuring vascular markings."
    elif under_exposed:
        status = ExposureClass.UNDER_EXPOSED
        reasoning = f"Under-penetrated: Heart/Spine variance is low ({variance:.1f}), and only {vertebrae_count}/12 thoracic vertebrae are visible."

    return {
        "status": status,
        "metrics": {
            "vertebrae_count": vertebrae_count,
            "heart_spine_variance": variance,
            "lung_burnout_ratio": burnout_ratio
        },
        "reasoning": reasoning
    }