import os

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    print("❌ MedMNIST is not installed. Please run: uv add medmnist")
    exit(1)

def build_medmnist_other_directory(num_per_dataset=10):
    output_dir = "data/other"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Target directory: {output_dir}")

    # Datasets explicitly excluding chestmnist and pneumoniamnist
    datasets_to_sample = [
        # "bloodmnist",    # Blood cells 
        # "dermamnist",    # Skin lesions
        # "retinamnist",   # Retinal fundus images
        # "pathmnist",     # Pathology slices
        "organamnist"    # Abdominal CTs
    ]

    for data_flag in datasets_to_sample:
        print(f"\n📥 Processing {data_flag}...")
        try:
            # Dynamically grab the PyTorch Dataset class from MedMNIST
            info = INFO[data_flag]
            DataClass = getattr(medmnist, info['python_class'])
            
            # Download and load the dataset. 
            # We use size=224 for the higher quality MedMNIST+ variant.
            dataset = DataClass(split='train', download=True, size=224)
            
            # Extract first N images
            for i in range(min(num_per_dataset, len(dataset))):
                img, label = dataset[i] # dataset[i] returns a tuple of (PIL Image, Label)
                
                # Format: medmnist_bloodmnist_0.jpg
                filename = f"medmnist_{data_flag}_{i}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Convert to RGB to ensure your letterbox preprocessing doesn't fail
                img = img.convert("RGB")
                img.save(filepath)
                
            print(f"  ✅ Saved {num_per_dataset} images from {data_flag}.")
            
        except Exception as e:
            print(f"  ❌ Failed on {data_flag}: {e}")

if __name__ == "__main__":
    print(f"MedMNIST v{medmnist.__version__}")
    build_medmnist_other_directory(num_per_dataset=50)