import os
import shutil
import pandas as pd

def organize_chexpert_valid_set():
    # Define your paths based on your setup
    csv_path = "data/CheXpert-v1/valid.csv"
    source_base_dir = "data/CheXpert-v1/valid"
    
    # Define output directories
    frontal_dir = os.path.join(source_base_dir, "frontal")
    lateral_dir = os.path.join(source_base_dir, "lateral")
    
    # Create the target subdirectories if they don't exist
    os.makedirs(frontal_dir, exist_ok=True)
    os.makedirs(lateral_dir, exist_ok=True)
    
    print(f"Loading CSV from: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find the CSV file at {csv_path}")
        return

    success_count = 0
    missing_count = 0

    print("Copying images to frontal/lateral subdirectories...")
    for index, row in df.iterrows():
        # The CSV paths look like: 'CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg'
        # We need to extract just the patient/study/file part to match your local setup
        original_csv_path = row['Path']
        
        # Strip the standard CheXpert prefix to get the relative path
        relative_path = original_csv_path.split("valid/")[-1] 
        
        # Build the actual source path on your machine
        local_image_path = os.path.join(source_base_dir, relative_path)
        
        # Get the filename (e.g., 'view1_frontal.jpg')
        filename = os.path.basename(local_image_path)
        
        # Prefix the filename with the patient and study to prevent name collisions
        # because multiple patients have a 'view1_frontal.jpg'
        patient_id = relative_path.split("/")[0]
        study_id = relative_path.split("/")[1]
        new_filename = f"{patient_id}_{study_id}_{filename}"

        # Determine target directory based on the 'Frontal/Lateral' column
        view_type = str(row['Frontal/Lateral']).lower()
        
        if "frontal" in view_type:
            target_path = os.path.join(frontal_dir, new_filename)
        elif "lateral" in view_type:
            target_path = os.path.join(lateral_dir, new_filename)
        else:
            print(f"⚠️ Unknown view type for {filename}: {view_type}")
            continue
            
        # Copy the file
        if os.path.exists(local_image_path):
            shutil.copy2(local_image_path, target_path)
            success_count += 1
        else:
            print(f"❌ File missing: {local_image_path}")
            missing_count += 1

    print("\n✅ Organization Complete!")
    print(f"Successfully copied: {success_count} images.")
    if missing_count > 0:
        print(f"Could not find: {missing_count} images (check paths).")
    print(f"Frontal images are in: {frontal_dir}")
    print(f"Lateral images are in: {lateral_dir}")

if __name__ == "__main__":
    organize_chexpert_valid_set()