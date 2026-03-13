import numpy as np
from pathlib import Path

# Import the pure, in-memory functions you already wrote!
from preprocessing import preprocess_image, KEY_CXAS_TENSOR, KEY_XRV_TENSOR
from segment import predict_cxas, predict_xrv
from ensemble import blend_patient_masks
from anatomical_filters import apply_anatomical_filters

def run_realtime_pipeline(image_path: Path, cxas_model, xrv_model, device: str) -> dict[str, np.ndarray]:
    """
    A pure in-memory pipeline for real-time clinical inference.
    Zero intermediate disk I/O.
    """
    # 1. Preprocess directly into RAM
    data = preprocess_image(image_path)
    
    # 2. Inference directly into RAM
    cxas_raw = predict_cxas(cxas_model, data[KEY_CXAS_TENSOR].to(device))
    xrv_raw = predict_xrv(xrv_model, data[KEY_XRV_TENSOR].to(device))
    
    # 3. Blend directly in RAM
    blended_dict = blend_patient_masks(cxas_raw, xrv_raw)
    
    # 4. Filter directly in RAM
    golden_dict = apply_anatomical_filters(blended_dict)
    
    # Return the final payload instantly for the RIPE QA algorithms
    return golden_dict