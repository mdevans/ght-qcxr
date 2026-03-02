import pandas as pd
import re

def find_best_zip_file():
    print("Loading PadChest metadata...")
    df = pd.read_csv("data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", low_memory=False)

    frontal_views = df[df['Projection'].isin(['PA', 'AP'])]
    block_list = ['pneumonia', 'pleural effusion', 'atelectasis', 'consolidation', 'pulmonary edema']
    blocked_mask = frontal_views['Labels'].apply(lambda x: any(b in str(x).lower() for b in block_list))
    clean_enough = frontal_views[~blocked_mask]

    keywords = [r"\binspiraci[oó]n escasa\b", r"\bmala inspiraci[oó]n\b", r"\bhipoventilaci[oó]n\b", r"\bescaso volumen\b"]
    bad_insp = clean_enough[clean_enough['Report'].astype(str).str.contains('|'.join(keywords), flags=re.IGNORECASE, na=False)]

    # Group by the Zip File directory and count them!
    zip_counts = bad_insp['ImageDir'].value_counts()
    
    print(f"\n📦 Top 5 Highest Yield Zip Files:")
    for zip_num, count in zip_counts.head(5).items():
        print(f"Zip File {int(zip_num)}.zip contains --> {count} images")

if __name__ == "__main__":
    find_best_zip_file()