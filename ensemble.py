import numpy as np
import cv2
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from PIL import ImageColor

# --- BLENDING STRATEGIES EXPLAINED ---
# XRV_PRIORITY: Prefer XRV's 2D boundary awareness. If XRV fails, fallback to CXAS.
# CXAS_PRIORITY: Prefer CXAS's 3D volumetric awareness (e.g., finding the spine behind the heart). If CXAS fails, fallback to XRV.
# UNION: Keep a pixel if EITHER model claims it. Maximizes sensitivity (Great for Heart/Mediastinum).
# INTERSECTION: Keep a pixel ONLY if BOTH models agree. Maximizes specificity (Great for Clavicles).
# CXAS_ONLY: Strictly use CXAS. Ignore XRV entirely (Used for specific ribs, sternum).
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

# 3. The Comprehensive, Typed Recipe Book mapped to available_classes.md
ANATOMY_RECIPES: dict[str, AnatomyRecipe] = {
    # --- LUNGS & PLEURA (Exposure / Inspiration) ---
    "Lung_Right": AnatomyRecipe("class_135", "Right_Lung", Strategy.UNION, "blue"),
    "Lung_Left": AnatomyRecipe("class_136", "Left_Lung", Strategy.UNION, "blue"),
    
    # --- CARDIAC & VESSELS (Exposure) ---
    "Heart": AnatomyRecipe("class_121", "Heart", Strategy.UNION, "red"),
    # "Aorta": AnatomyRecipe("class_127", "Aorta", Strategy.UNION, "crimson"),
    # "Mediastinum": AnatomyRecipe("class_115", "Mediastinum", Strategy.UNION, "purple"),
    
    # --- BONES: ANCHORS & ROTATION ---
    "Spine": AnatomyRecipe("class_000", "Spine", Strategy.CXAS_PRIORITY, "yellow"),
    "Sternum": AnatomyRecipe("class_029", None, Strategy.CXAS_ONLY, "cyan"),
    "Clavicle_Right": AnatomyRecipe("class_032", "Right_Clavicle", Strategy.UNION, "magenta"),
    "Clavicle_Left": AnatomyRecipe("class_031", "Left_Clavicle", Strategy.UNION, "magenta"),
    # "Scapula_Right": AnatomyRecipe("class_035", "Right_Scapula", Strategy.UNION, "hotpink"),
    # "Scapula_Left": AnatomyRecipe("class_034", "Left_Scapula", Strategy.UNION, "hotpink"),
    
    # --- AIRWAYS & ABDOMEN (Projection / Inspiration) ---
    # "Trachea": AnatomyRecipe("class_154", "Weasand", Strategy.UNION, "white"),
    # "Stomach": AnatomyRecipe("class_108", None, Strategy.CXAS_ONLY, "greenyellow"), # Gastric bubble
    
    # --- DIAPHRAGM (Inspiration) ---
    "Diaphragm_Right": AnatomyRecipe("class_107", None, Strategy.CXAS_ONLY, "orange"),
    "Diaphragm_Left": AnatomyRecipe("class_106", None, Strategy.CXAS_ONLY, "orange"),
    "Diaphragm_XRV": AnatomyRecipe(None, "Facies_Diaphragmatica", Strategy.XRV_ONLY, "orange"),
    "Diaphragm_Full": AnatomyRecipe(None, None, Strategy.INTERSECTION, "darkorange"), # Derived Mask
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
# T1 corresponds to class_011, T12 to class_022
for i in range(1, 13):
    cxas_idx = 10 + i
    ANATOMY_RECIPES[f"T{i}"] = AnatomyRecipe(
        f"class_{cxas_idx:03d}", None, Strategy.CXAS_ONLY, "yellow"
    )

# --- CORE BLENDING LOGIC ---

def apply_merge_strategy(
    cxas_mask: np.ndarray | None, 
    xrv_mask: np.ndarray | None, 
    strategy: Strategy
) -> np.ndarray: 
    
    # 1. If both models failed to output an array, we have nothing to blend.
    if cxas_mask is None and xrv_mask is None: 
        raise ValueError(f"Cannot apply {strategy.value}: Both CXAS and XRV masks are missing.")
    
    # 2. Handle single-model failures explicitly
    if cxas_mask is None: 
        if strategy in [Strategy.CXAS_ONLY, Strategy.INTERSECTION]:
            raise ValueError(f"Strategy '{strategy.value}' strictly requires a CXAS mask, but it was missing.")
        assert xrv_mask is not None, "Pylance guard: xrv_mask should not be None here."
        return xrv_mask  # Guaranteed not None because of Check #1
        
    if xrv_mask is None: 
        if strategy in [Strategy.XRV_ONLY, Strategy.INTERSECTION]:
            raise ValueError(f"Strategy '{strategy.value}' strictly requires an XRV mask, but it was missing.")
        return cxas_mask # Guaranteed not None because of Check #1
        
    # 3. Apply the mathematical rules if both masks exist
    if strategy in [Strategy.CXAS_ONLY, Strategy.CXAS_PRIORITY]: return cxas_mask
    if strategy in [Strategy.XRV_ONLY, Strategy.XRV_PRIORITY]: return xrv_mask
    if strategy == Strategy.UNION: return np.logical_or(cxas_mask, xrv_mask).astype(np.uint8)
    if strategy == Strategy.INTERSECTION: return np.logical_and(cxas_mask, xrv_mask).astype(np.uint8)
    
    # Catch-all for undefined strategies
    raise ValueError(f"Unknown merging strategy encountered: {strategy}")

def blend_patient_masks(cxas_dict: dict[str, np.ndarray], xrv_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Orchestrates the blending of all requested anatomical targets using defined strategies.
    Returns a dictionary mapping the AnatomyRecipe keys to the finalized 512x512 numpy arrays.
    """
    final_masks = {}
    
    for target_name, recipe in ANATOMY_RECIPES.items():
        # Skip derived masks during the standard 1-to-1 processing loop
        if target_name == "Diaphragm_Full":
            continue
            
        c_mask = cxas_dict.get(recipe.cxas_key) if recipe.cxas_key else None
        x_mask = xrv_dict.get(recipe.xrv_key) if recipe.xrv_key else None
        
        try:
            blended = apply_merge_strategy(c_mask, x_mask, recipe.strategy)
            if blended is not None and blended.max() > 0:
                final_masks[target_name] = blended
        except ValueError as e:
            #TODO: log this to a proper logger rather than pure print
            # print(f"Warning: {target_name} - {e}")
            pass
            
    # ==========================================
    # POST-PROCESSING: DERIVED MASKS
    # ==========================================
    
    # Generate the Unified Diaphragm (Combining CXAS precision with XRV coverage)
    diaphragm_components = ["Diaphragm_Right", "Diaphragm_Left", "Diaphragm_XRV"]
    available_parts = [
        final_masks[k] for k in diaphragm_components 
        if k in final_masks and final_masks[k] is not None
    ]
    
    if available_parts:
        # Start with a boolean array of the first available part
        unified_bool = available_parts[0] > 0
        # Logical OR it against any remaining parts
        for part in available_parts[1:]:
            unified_bool = np.logical_or(unified_bool, part > 0)
            
        final_masks["Diaphragm_Full"] = unified_bool.astype(np.uint8)

    return final_masks

# --- VISUALIZATION ---
def generate_preview_overlay(base_img_array: np.ndarray, blended_dict: dict[str, np.ndarray], output_path: Path) -> None:
    """Generates an RGB visual indicator of the rationalized masks using PIL colors."""
    if base_img_array is None: 
        return
        
    # The base_img_array from preprocessing is single-channel grayscale. 
    # We must convert it to BGR so OpenCV can draw colored overlays on it.
    overlay_canvas = cv2.cvtColor(base_img_array, cv2.COLOR_GRAY2BGR)
    overlay = overlay_canvas.copy()
    
    for anatomy, mask in blended_dict.items():
        # Prevent OpenCV broadcast crashes if geometries mismatch
        if mask.shape == overlay_canvas.shape[:2]:
            color_name = ANATOMY_RECIPES[anatomy].color if anatomy in ANATOMY_RECIPES else "lime"
            
            # Fetch RGB tuple from PIL and reverse it to BGR for OpenCV
            rgb_tuple = ImageColor.getrgb(color_name)
            bgr_tuple = rgb_tuple[::-1] 
            
            # Apply the color to the mask pixels
            overlay[mask > 0] = bgr_tuple
        
    # Blend colored overlay with the original grayscale image (30% opacity)
    cv2.addWeighted(overlay, 0.3, overlay_canvas, 0.7, 0, overlay_canvas)
    cv2.imwrite(str(output_path), overlay_canvas)