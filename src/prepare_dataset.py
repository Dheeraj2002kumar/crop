import os
import shutil
import random
import sys
from tqdm import tqdm

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

def prepare_splits():
    print(f"[*] Starting dataset preparation from: {config.DATASET_DIR}")
    
    # Check if raw dataset exists
    if not os.path.exists(config.DATASET_DIR):
        print(f"[!] Error: Raw dataset directory '{config.DATASET_DIR}' not found.")
        sys.exit(1)
        
    # Get all class directories
    classes = [d for d in os.listdir(config.DATASET_DIR) 
               if os.path.isdir(os.path.join(config.DATASET_DIR, d)) and d != "PlantVillage"]
    
    print(f"[*] Found {len(classes)} classes for classification.")
    
    # If split directory already exists, clear it to prevent mixing old splits
    if os.path.exists(config.SPLIT_DIR):
        print(f"[*] Cleaning up existing split directory at: {config.SPLIT_DIR}")
        shutil.rmtree(config.SPLIT_DIR)
        
    # Re-create directories
    for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        os.makedirs(split_dir, exist_ok=True)
        for class_name in classes:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    random.seed(config.SEED)
    
    total_copied = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    
    # Partition each class
    for class_name in tqdm(classes, desc="Processing classes"):
        class_src_dir = os.path.join(config.DATASET_DIR, class_name)
        
        # Get all image files
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        images = [f for f in os.listdir(class_src_dir) 
                  if os.path.isfile(os.path.join(class_src_dir, f)) and f.lower().endswith(valid_extensions)]
        
        # Shuffle images
        random.shuffle(images)
        
        num_images = len(images)
        n_train = int(num_images * config.TRAIN_RATIO)
        n_val = int(num_images * config.VAL_RATIO)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        splits = {
            "train": (train_imgs, config.TRAIN_DIR),
            "val": (val_imgs, config.VAL_DIR),
            "test": (test_imgs, config.TEST_DIR)
        }
        
        for split_name, (img_list, dest_split_dir) in splits.items():
            dest_class_dir = os.path.join(dest_split_dir, class_name)
            for img in img_list:
                src_path = os.path.join(class_src_dir, img)
                # Sanitize filename: replace spaces with underscores
                img_sanitized = img.replace(" ", "_")
                dest_path = os.path.join(dest_class_dir, img_sanitized)
                
                try:
                    # Attempt hard linking (fast, 0-copy)
                    os.link(src_path, dest_path)
                except OSError:
                    # Fallback to copy if hard link is not supported (e.g. across filesystems)
                    shutil.copy2(src_path, dest_path)
                
                total_copied += 1
                split_counts[split_name] += 1
                
    print("\n[+] Dataset splitting completed successfully!")
    print(f"Total processed images: {total_copied}")
    print(f" - Train: {split_counts['train']} ({config.TRAIN_RATIO * 100:.0f}%)")
    print(f" - Validation: {split_counts['val']} ({config.VAL_RATIO * 100:.0f}%)")
    print(f" - Test: {split_counts['test']} ({config.TEST_RATIO * 100:.0f}%)")

if __name__ == "__main__":
    prepare_splits()
