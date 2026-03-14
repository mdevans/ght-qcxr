import numpy as np

# Import our strict type alias to guarantee the data contract
from preprocessing import GrayscaleImage

# ==========================================
# --- EXPOSURE THRESHOLDS & CONSTANTS ---
# ==========================================

# Base dynamic range for normalization (8-bit grayscale image)
MAX_8BIT_PIXEL_VALUE: float = 255.0

# Intensity value (0-255) below which a pixel is considered "burned out" (clinically black)
BURNOUT_PIXEL_THRESHOLD: int = 15

# Percentage (0.0 to 1.0) of the lung area that must be burned out to trigger OVER-EXPOSED
THRESHOLD_OVER_EXPOSED_BURNOUT_RATIO: float = 0.30

# Minimum acceptable contrast (Standard Deviation / 255) in the heart-spine intersection.
# Below this percentage (0.0 to 1.0), the image is considered washed out.
THRESHOLD_UNDER_EXPOSED_CONTRAST: float = 0.03

# Minimum number of thoracic vertebrae (integer out of 12) that must be visible 
# to confirm adequate penetration.
THRESHOLD_UNDER_EXPOSED_VERTEBRAE: int = 8

# ==========================================


class ExposureClass:
    UNDER_EXPOSED = "under-exposed"
    NORMAL = "normal"
    OVER_EXPOSED = "over-exposed"

# ==========================================
# 1. MATHEMATICAL HELPERS (Pure Extractors)
# ==========================================

def cxas_thoracic_vertebrae_segmented(segmented_masks: dict[str, np.ndarray]) -> int:
    """Counts how many individual thoracic vertebrae (T1-T12) the AI successfully mapped."""
    target_keys = {f"T{i}" for i in range(1, 13)}
    
    return sum(
        1 for key in target_keys
        if (mask := segmented_masks.get(key)) is not None and mask.max() > 0
    )

def heart_spine_penetration_contrast(segmented_masks: dict[str, np.ndarray], gray_img: GrayscaleImage) -> float:
    """Calculates the contrast (StdDev as a percentage of dynamic range) where the heart and spine overlap."""
    heart_mask = segmented_masks.get("Heart")
    spine_mask = segmented_masks.get("Spine") 
    
    if heart_mask is None or spine_mask is None or heart_mask.max() == 0 or spine_mask.max() == 0:
        return 0.0
        
    overlap_mask = np.logical_and(heart_mask > 0, spine_mask > 0)
    if not np.any(overlap_mask):
        return 0.0
        
    overlap_pixels = gray_img[overlap_mask]
    
    return float(np.std(overlap_pixels) / MAX_8BIT_PIXEL_VALUE)

def lung_field_burnout(
    segmented_masks: dict[str, np.ndarray], 
    gray_img: GrayscaleImage, 
    black_threshold: int = BURNOUT_PIXEL_THRESHOLD
) -> float:
    """Calculates the ratio of the lung field that is burned out (near absolute black)."""
    right_lung = segmented_masks.get("Lung_Right")
    left_lung = segmented_masks.get("Lung_Left")
    
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

def is_over_exposed(
    burnout_ratio: float, 
    burnout_threshold: float = THRESHOLD_OVER_EXPOSED_BURNOUT_RATIO
) -> bool:
    return burnout_ratio > burnout_threshold

def is_under_exposed(
    contrast_ratio: float, 
    vertebrae_count: int, 
    min_contrast: float = THRESHOLD_UNDER_EXPOSED_CONTRAST, 
    min_vertebrae: int = THRESHOLD_UNDER_EXPOSED_VERTEBRAE
) -> bool:
    return contrast_ratio < min_contrast or vertebrae_count < min_vertebrae

# ==========================================
# 3. PIPELINE ORCHESTRATOR
# ==========================================

def assess_exposure(segmented_masks: dict[str, np.ndarray], gray_img: GrayscaleImage) -> dict:
    """
    Executes the functional Exposure QA pipeline entirely in memory.
    Strictly assumes gray_img and segmented_masks are pre-aligned and clean.
    """
    vertebrae_count = cxas_thoracic_vertebrae_segmented(segmented_masks)
    contrast = heart_spine_penetration_contrast(segmented_masks, gray_img)
    burnout_ratio = lung_field_burnout(segmented_masks, gray_img)

    over_exposed = is_over_exposed(burnout_ratio)
    under_exposed = is_under_exposed(contrast, vertebrae_count)

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
        reasoning = f"Under-penetrated: Heart/Spine contrast is severely low ({contrast:.1%}), and only {vertebrae_count}/12 thoracic vertebrae are visible."

    return {
        "status": status,
        "metrics": {
            "vertebrae_count": vertebrae_count,
            "heart_spine_contrast": contrast,
            "lung_burnout_ratio": burnout_ratio
        },
        "reasoning": reasoning
    }