import os
import shutil
import random
from rich.console import Console
from rich.progress import track

# --- CONFIG ---
# Update this path if your PA images are stored somewhere else!
SOURCE_DIR = "data/PA_CXR_448" 
OUTPUT_DIR = "data/rotation/keypoint_dataset/raw_PA_images"
NUM_IMAGES_TO_SAMPLE = 75

console = Console()

def main():
    console.print(f"[bold cyan]🎯 Prepping Adult PA X-rays for Keypoint Annotation[/bold cyan]")
    
    if not os.path.exists(SOURCE_DIR):
        console.print(f"[bold red]Error: Source directory '{SOURCE_DIR}' not found.[/bold red]")
        console.print("Please update the SOURCE_DIR variable in the script to point to your PA images.")
        return

    # Create fresh output directory (clears it if you run it twice)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Grab all image files
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not all_files:
        console.print(f"[bold red]No images found in {SOURCE_DIR}![/bold red]")
        return
        
    console.print(f"Found {len(all_files)} total PA images.")

    # Sample the images
    sample_size = min(NUM_IMAGES_TO_SAMPLE, len(all_files))
    
    # Using a seed so you get the same 75 images if you have to re-run it
    random.seed(42) 
    sampled_files = random.sample(all_files, sample_size)
    
    # Copy to the staging folder
    for f in track(sampled_files, description="Copying images..."):
        src = os.path.join(SOURCE_DIR, f)
        dst = os.path.join(OUTPUT_DIR, f)
        shutil.copy(src, dst)

    console.print(f"\n[bold green]✓ Successfully copied {sample_size} images to:[/bold green] {OUTPUT_DIR}")

if __name__ == "__main__":
    main()