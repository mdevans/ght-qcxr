import os
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import (
    apply_color_lut,
    apply_modality_lut,
    apply_voi_lut,
    convert_color_space
)
from pyorthanc import Orthanc, query_orthanc, Series, Instance
from pyorthanc.util import get_pydicom
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# --- CONFIGURATION ---
ORTHANC_URL = "https://ar.dx.life/orthanc"
USERNAME = "mdevans"
PASSWORD = "pika3193"
OUTPUT_FOLDER = "data/PA_CXR_448"
TARGET_SIZE = 448
TARGET_COUNT = 100
view_position = "PA"
query = {'Modality': 'DX', 'BodyPartExamined': 'CHEST', 'SeriesDescription': 'CHEST PA'} 

console = Console()

def process_dicom_to_rgb(ds):
    """
    Robust DICOM processing logic to produce a 
    448x448 RGB image suitable for MedSigLIP.
    """
    # 1. Basic Metadata Extraction
    pixel_array = ds.pixel_array
    pi = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    
    # 2. Color Space / LUT Handling (Your Logic)
    # Apply Palette Color LUT if present
    if pi == "PALETTE COLOR" and "PaletteColorLookupTableData" in ds:
        pixel_array = apply_color_lut(pixel_array, ds)
    # Convert YBR to RGB if needed
    elif pi in ["YBR_FULL", "YBR_FULL_422"]:
        pixel_array = convert_color_space(pixel_array, pi, "RGB", per_frame=True)
    
    # 3. Grayscale Processing (Modality & VOI LUTs)
    # Most CXRs are grayscale. This applies the "Window Center/Width" correctly.
    # Create an empty destination array of the correct shape and type (uint8)
    dst = np.zeros(pixel_array.shape, dtype=np.uint8)
    if pi in ["MONOCHROME1", "MONOCHROME2"]:
        # Apply Modality LUT (Rescale Slope/Intercept)
        pixel_array = apply_modality_lut(pixel_array, ds)
        # Apply VOI LUT (Windowing) - Critical for contrast
        pixel_array = apply_voi_lut(pixel_array, ds)
        
        # 4. Normalization (Your Logic)
        # Normalize to 0-255 uint8
        pixel_array = cv2.normalize(
            pixel_array, 
            dst, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        
        # 5. Inversion (MONOCHROME1 check)
        # MONOCHROME1 means 0=White. We need 0=Black (MONOCHROME2 style).
        if pi == "MONOCHROME1":
            pixel_array = cv2.bitwise_not(pixel_array)

        # Convert Grayscale to RGB (3 channels) so it matches Color logic below
        # and satisfies model input requirements
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)

    else:
        # If it was already RGB/PALETTE, just ensure it's uint8 0-255
        pixel_array = cv2.normalize(
            pixel_array, 
            dst, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )

    # 6. Aspect-Ratio Preserving Resize ("Letterbox")
    # This replaces simple resizing to avoid geometric distortion
    h, w = pixel_array.shape[:2]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(pixel_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate Padding (to reach 448x448)
    delta_w = TARGET_SIZE - new_w
    delta_h = TARGET_SIZE - new_h
    
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Add Black Border
    final_image = cv2.copyMakeBorder(
        resized, 
        top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0] # Black padding
    )
    
    return final_image

def main():
    console.print(f"[bold cyan]🏥 Orthanc Processor -> MedSigLIP Ready ({TARGET_SIZE}px)[/bold cyan]")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Connect
    try:
        client = Orthanc(ORTHANC_URL, USERNAME, PASSWORD, timeout=180)
        console.print(client.get_system())
        console.print("[green]✅ Connected to Orthanc[/green]")
    except Exception as e:
        console.print(f"[red]Connection Error: {e}[/red]")
        return

    # 2. Query
    console.print("[bold]🔍 Searching PACS...[/bold]")
    console.print(query)

    series: list = query_orthanc(level='Series', client=client, query=query, limit=TARGET_COUNT, retrieve_all_resources=False)
    
    if not series:
        console.print("[yellow]⚠️ No series found matching the query. Check your criteria.[/yellow]")
        return

    console.print(f"   Found [bold]{len(series)}[/bold] series matching query. (Limit: {TARGET_COUNT})")
    
    # Extract all instances from all returned series:
    instances: list[Instance] = []
    for s in series:
        instances.extend(s.instances)

    # 3. Get set of instances
    existing_files = set(os.listdir(OUTPUT_FOLDER))
    existing_ids = {f.rsplit('.', 1)[0] for f in existing_files}
    target_instances = [i for i in instances if i.id_ not in existing_ids]
    
    if not target_instances:
        console.print("[green]✅ All available images are already downloaded [/green]")
        return

    console.print(f"   Download & Process [bold]{len(target_instances)}[/bold] instances.")
    
    # 3. Download & Process Loop
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        
        task = progress.add_task("Processing...", total=len(target_instances))
        
        for instance in target_instances:

            save_path = os.path.join(OUTPUT_FOLDER, f"{instance.id_}.png")
            try:
                # A: Check View Position:
                vp = instance.get_content_by_tag("ViewPosition") 
                if vp != view_position:
                    progress.update(task, advance=1, description=f"[yellow]Skipped {instance.id_} (Wrong View: {vp})[/yellow]")
                    continue

                # B. Get dicom file
                ds = get_pydicom(client, instance.id_)

                # C. Process (The Robust Pipeline)
                rgb_image = process_dicom_to_rgb(ds)
                
                # D. Save as PNG (Optimal for AI)
                # Use OpenCV to save (faster than PIL for numpy arrays)
                # cv2 saves as BGR by default, but our pipeline made RGB
                # Convert back to BGR for saving, OR just use PIL if you prefer
                # Standard cv2 expects BGR
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_image)
                
                progress.update(task, advance=1, description=f"[cyan]Saved {save_path}[/cyan]")
                
            except Exception as e:
                console.print(f"[red]Error on {instance.id_}: {e}[/red]")

    console.print(f"\n[bold green]✅ Done: Images in: {OUTPUT_FOLDER}[/bold green]")

if __name__ == "__main__":
    main()