import os
import zipfile
import requests
from tqdm import tqdm

def download_tinyimagenet(save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(save_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(save_dir, "tiny-imagenet-200")

    # Download
    if not os.path.exists(zip_path):
        print("Downloading TinyImageNet...")
        with requests.get(url, stream=True) as r:
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                desc="Downloading", total=total, unit="B", unit_scale=True
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        print("Already downloaded.")

    # Extract
    if not os.path.exists(extract_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)
    else:
        print("Already extracted.")

    return extract_path

def reorganize_for_imagefolder(base_dir):
    print("Reorganizing data to ImageFolder format...")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    val_images_dir = os.path.join(val_dir, "images")

    # Reorganize validation images into class folders
    with open(os.path.join(val_dir, "val_annotations.txt")) as f:
        val_annotations = f.readlines()

    val_dict = {}
    for line in val_annotations:
        parts = line.strip().split("\t")
        img, cls = parts[0], parts[1]
        val_dict[img] = cls
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        img_src = os.path.join(val_images_dir, img)
        img_dst = os.path.join(cls_dir, img)
        if os.path.exists(img_src):
            os.rename(img_src, img_dst)

    # Remove old val images folder
    if os.path.exists(val_images_dir):
        os.rmdir(val_images_dir)

    print("Done.")

# --- Run it ---
if __name__ == "__main__":
    path = download_tinyimagenet("data")
    reorganize_for_imagefolder(path)

    # Now train_dir and val_dir are:
    # data/tiny-imagenet-200/train
    # data/tiny-imagenet-200/val
    print("\nUse the following in your training pipeline:")
    print("  train_dir = 'data/tiny-imagenet-200/train'")
    print("  val_dir   = 'data/tiny-imagenet-200/val'")
