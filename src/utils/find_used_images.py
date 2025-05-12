import os
import pandas as pd
import random
import shutil
from pathlib import Path
import argparse

# Paths
ORIGINAL_DB_PATH = "data/10k-imagenet"
USED_IMAGES_CSV = "data/stats/images_used.csv"  # CSV containing used images
OUTPUT_DIR = "data/used_images"    # Directory to copy used images to

"""bash
example usage:
 python src/utils/find_used_images.py --num-images 30
 
"""

def find_used_images(original_db_path, used_images_csv, output_dir, num_images=30):
    """
    Find used images from the CSV file and copy them to output directory.
    
    Args:
        original_db_path: Path to the original image database
        used_images_csv: Path to CSV file containing used images
        output_dir: Directory to copy used images to
        num_images: Number of used images to find
    """
    print(f"Finding {num_images} used images...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV of used images
    try:
        if not os.path.exists(used_images_csv):
            print(f"ERROR: CSV file '{used_images_csv}' does not exist!")
            print("Please provide a valid CSV file with used images.")
            return []
        else:
            used_df = pd.read_csv(used_images_csv)
            # Extract filenames from the path column (assuming there's a column with image paths)
            # Adjust the column name if needed
            if 'imagename' in used_df.columns:
                used_filenames = set(used_df['imagename'])
                print(f"Using column 'imagename' for image filenames")
            elif 'path' in used_df.columns:
                used_filenames = set(used_df['path'].apply(os.path.basename))
            elif 'image_path' in used_df.columns:
                used_filenames = set(used_df['image_path'].apply(os.path.basename))
            elif 'Image' in used_df.columns:
                used_filenames = set(used_df['Image'].apply(os.path.basename))
            else:
                # Try to find any column that might contain file paths
                for col in used_df.columns:
                    if used_df[col].dtype == 'object' and used_df[col].str.contains('.jpg|.png|.jpeg', case=False).any():
                        used_filenames = set(used_df[col].apply(os.path.basename))
                        print(f"Using column '{col}' for image filenames")
                        break
                else:
                    print("ERROR: Could not find a column with image filenames in the CSV file!")
                    print("The CSV should have a column containing image paths or filenames.")
                    return []
        
        print(f"Found {len(used_filenames)} used images in CSV")
    except Exception as e:
        print(f"ERROR: Failed to process the CSV file: {e}")
        print("Please check that the CSV file is properly formatted.")
        return []
    
    # Get all image files from the original database
    all_images = []
    for root, _, files in os.walk(original_db_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} total images in database")
    
    # Filter to only include used images
    used_images = [img for img in all_images if os.path.basename(img) in used_filenames]
    print(f"Found {len(used_images)} matching used images in the database")
    
    # Randomly select num_images from used images
    if len(used_images) < num_images:
        print(f"Warning: Only {len(used_images)} used images available")
        selected_images = used_images
    else:
        selected_images = random.sample(used_images, num_images)
    
    # Copy selected images to output directory
    for img_path in selected_images:
        dest_path = os.path.join(output_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dest_path)
        print(f"Copied {os.path.basename(img_path)} to {output_dir}")
    
    print(f"Successfully copied {len(selected_images)} used images to {output_dir}")
    return selected_images

if __name__ == "__main__":
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description='Find used images from the CSV and copy them to output directory')
    parser.add_argument('--num-images', type=int, default=30, 
                        help='Number of used images to find (default: 30)')
    
    args = parser.parse_args()
    
    # Use command-line arguments
    used_images = find_used_images(
        ORIGINAL_DB_PATH, 
        USED_IMAGES_CSV, 
        OUTPUT_DIR, 
        args.num_images
    )
    
    # Write the list of used images to a file for reference
    if used_images:
        with open(os.path.join(OUTPUT_DIR, "used_images_list.txt"), "w") as f:
            for img in used_images:
                f.write(f"{img}\n")