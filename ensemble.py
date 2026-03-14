import numpy as np
import cv2
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from PIL import ImageColor

# --- BLENDING STRATEGIES EXPLAINED ---
# XRV_PRIORITY: Prefer XRV's 2D boundary awareness. If XRV fails, fallback to CXAS.
# CXAS_PRIORITY: Prefer CXAS's 3D volumetric awareness. If CXAS fails, fallback to XRV.
# UNION: Keep a pixel if EITHER model claims it. Maximizes sensitivity.
# INTERSECTION: Keep a pixel ONLY if BOTH models agree. Maximizes specificity.
# CXAS_ONLY: Strictly use CXAS. Ignore XRV entirely.
# XRV_ONLY: Strictly use XRV. Ignore CXAS entirely.

class Strategy(str, Enum):
    XRV_PRIORITY = "xrv_priority"
    CXAS_PRIORITY = "cxas_priority"
    UNION = "union"
    INTERSECTION = "intersection"
    CXAS_ONLY = "cxas_only"
    XRV_ONLY = "xrv_only"

# --- UX-FRIENDLY RECIPE CONFIGURATION ---

@dataclass
class AnatomyRecipe:
    cxas_key: str | None
    xrv_key: str | None
    strategy: Strategy
    color: str = "white" # Standard W3C color name
    expected_segments: int = 1 # Used to filter out AI hallucinations (noise)

# 3. The Comprehensive, Typed Recipe Book mapped to available_classes.md
ANATOMY_RECIPES: dict[str, AnatomyRecipe] = {
    # --- LUNGS & PLEURA (Exposure / Inspiration) ---
    "Lung_Right": AnatomyRecipe("class_135", "Right_Lung", Strategy.UNION, "blue"),
    "Lung_Left": AnatomyRecipe("class_136", "Left_Lung", Strategy.UNION, "blue"),
    
    # --- CARDIAC & VESSELS (Exposure) ---
    "Heart": AnatomyRecipe("class_121", "Heart", Strategy.UNION, "red"),
    
    # --- BONES: ANCHORS & ROTATION ---
    "Spine": AnatomyRecipe("class_000", "Spine", Strategy.CXAS_PRIORITY, "yellow"),
    "Sternum": AnatomyRecipe("class_029", None, Strategy.CXAS_ONLY, "cyan"),
    "Clavicle_Right": AnatomyRecipe("class_032", "Right_Clavicle", Strategy.UNION, "magenta"),
    "Clavicle_Left": AnatomyRecipe("class_031", "Left_Clavicle", Strategy.UNION, "magenta"),
    
    # --- DIAPHRAGM (Inspiration) ---
    "Diaphragm_Right": AnatomyRecipe("class_107", None, Strategy.CXAS_ONLY, "orange"),
    "Diaphragm_Left": AnatomyRecipe("class_106", None, Strategy.CXAS_ONLY, "orange"),
    # The XRV model and full union often yield two distinct blobs (left/right separated by heart)
    "Diaphragm_XRV": AnatomyRecipe(None, "Facies_Diaphragmatica", Strategy.XRV_ONLY, "orange", expected_segments=2),
    "Diaphragm_Full": AnatomyRecipe(None, None, Strategy.UNION, "darkorange", expected_segments=2), 
}

# Programmatically generate all 24 Posterior Ribs for Inspiration Logic
RIGHT_RIBS = {1: 79, 2: 75, 3: 71, 4: 67, 5: 63, 6: 59, 7: 55, 8: 51, 9: 47, 10: 43, 11: 39, 12: 36}
LEFT_RIBS = {1: 81, 2: 77, 3: 73, 4: 69, 5: 65, 6: 61, 7: 57, 8: 53, 9: 49, 10: 45, 11: 41, 12: 37}

for rib_num, cxas_idx in RIGHT_RIBS.items():
    ANATOMY_RECIPES[f"Posterior_Rib_{rib_num}_Right"] = AnatomyRecipe(
        f"class_{cxas_idx:03d}", None, Strategy.CXAS_ONLY, "lime"
    )
for rib_num, cxas_idx in LEFT_RIBS.items():
    ANATOMY_RECIPES[f"Posterior_Rib_{rib_num}_Left"] = AnatomyRecipe(
        f"class_{cxas_idx:03d}", None, Strategy.CXAS_ONLY, "lime"
    )

# Programmatically generate 12 Thoracic Vertebrae (T1-T12) for Exposure QA
for i in range(1, 13):
    cxas_idx = 10 + i
    ANATOMY_RECIPES[f"T{i}"] = AnatomyRecipe(
        f"class_{cxas_idx:03d}", None, Strategy.CXAS_ONLY, "yellow"
    )

# --- CORE BLENDING & FILTERING LOGIC ---

def _keep_top_n_segments(mask: np.ndarray, max_segments: int) -> np.ndarray:
    """
    Scrub AI hallucinations by keeping only the largest `max_segments` contiguous areas.
    If max_segments is 0, the filter is bypassed.
    """
    if max_segments <= 0 or mask is None or mask.max() == 0:
        return mask

    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If the AI generated the expected amount (or fewer), we accept it as is
    if len(contours) <= max_segments:
        return mask_uint8

    # If it generated excess noise, sort contours by pixel area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Draw only the allowed number of largest contours onto a fresh array
    clean_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(clean_mask, contours[:max_segments], -1, 1, thickness=cv2.FILLED)
    
    return clean_mask

def apply_merge_strategy(
    cxas_mask: np.ndarray | None, 
    xrv_mask: np.ndarray | None, 
    strategy: Strategy
) -> np.ndarray: 
    
    if cxas_mask is None and xrv_mask is None: 
        raise ValueError(f"Cannot apply {strategy.value}: Both CXAS and XRV masks are missing.")
    
    if cxas_mask is None: 
        if strategy in [Strategy.CXAS_ONLY, Strategy.INTERSECTION]:
            raise ValueError(f"Strategy '{strategy.value}' strictly requires a CXAS mask, but it was missing.")
        assert xrv_mask is not None
        return xrv_mask 
        
    if xrv_mask is None: 
        if strategy in [Strategy.XRV_ONLY, Strategy.INTERSECTION]:
            raise ValueError(f"Strategy '{strategy.value}' strictly requires an XRV mask, but it was missing.")
        return cxas_mask 
        
    if strategy in [Strategy.CXAS_ONLY, Strategy.CXAS_PRIORITY]: return cxas_mask
    if strategy in [Strategy.XRV_ONLY, Strategy.XRV_PRIORITY]: return xrv_mask
    if strategy == Strategy.UNION: return np.logical_or(cxas_mask, xrv_mask).astype(np.uint8)
    if strategy == Strategy.INTERSECTION: return np.logical_and(cxas_mask, xrv_mask).astype(np.uint8)
    
    raise ValueError(f"Unknown merging strategy encountered: {strategy}")

def blend_patient_masks(cxas_dict: dict[str, np.ndarray], xrv_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Orchestrates the blending of requested anatomical targets and strictly filters AI noise.
    """
    final_masks = {}
    
    for target_name, recipe in ANATOMY_RECIPES.items():
        if target_name == "Diaphragm_Full":
            continue
            
        c_mask = cxas_dict.get(recipe.cxas_key) if recipe.cxas_key else None
        x_mask = xrv_dict.get(recipe.xrv_key) if recipe.xrv_key else None
        
        try:
            blended = apply_merge_strategy(c_mask, x_mask, recipe.strategy)
            
            # --- NEW: Apply the noise filter immediately after blending ---
            if blended is not None and blended.max() > 0:
                clean_mask = _keep_top_n_segments(blended, recipe.expected_segments)
                if clean_mask.max() > 0:
                    final_masks[target_name] = clean_mask
                    
        except ValueError:
            pass
            
    # ==========================================
    # POST-PROCESSING: DERIVED MASKS
    # ==========================================
    
    diaphragm_components = ["Diaphragm_Right", "Diaphragm_Left", "Diaphragm_XRV"]
    available_parts = [
        final_masks[k] for k in diaphragm_components 
        if k in final_masks and final_masks[k] is not None
    ]
    
    if available_parts:
        unified_bool = available_parts[0] > 0
        for part in available_parts[1:]:
            unified_bool = np.logical_or(unified_bool, part > 0)
            
        # Clean the final derived mask using its specific recipe bounds
        clean_diaphragm = _keep_top_n_segments(
            unified_bool.astype(np.uint8), 
            ANATOMY_RECIPES["Diaphragm_Full"].expected_segments
        )
        final_masks["Diaphragm_Full"] = clean_diaphragm

    return final_masks

# --- VISUALIZATION ---
def generate_preview_overlay(base_img_array: np.ndarray, blended_dict: dict[str, np.ndarray], output_path: Path) -> None:
    if base_img_array is None: 
        return
        
    overlay_canvas = cv2.cvtColor(base_img_array, cv2.COLOR_GRAY2BGR)
    overlay = overlay_canvas.copy()
    
    for anatomy, mask in blended_dict.items():
        if mask.shape == overlay_canvas.shape[:2]:
            color_name = ANATOMY_RECIPES[anatomy].color if anatomy in ANATOMY_RECIPES else "lime"
            rgb_tuple = ImageColor.getrgb(color_name)
            bgr_tuple = rgb_tuple[::-1] 
            
            overlay[mask > 0] = bgr_tuple
        
    cv2.addWeighted(overlay, 0.3, overlay_canvas, 0.7, 0, overlay_canvas)
    cv2.imwrite(str(output_path), overlay_canvas)