import os
import argparse
import random
import sys

# Add root directory to path to ensure proper imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import src.config as config
from src.prepare_dataset import prepare_splits
from src.train import train_pipeline
from src.evaluate import evaluate_model
from src.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="End-to-end Plant Disease Classification Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Split the raw dataset into train/val/test folders")
    parser.add_argument("--train", action="store_true", help="Build and train the transfer learning model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model on the testing split")
    parser.add_argument("--demo", action="store_true", help="Perform sample inference on a random test leaf image")
    parser.add_argument("--all", action="store_true", default=True, help="Run the entire end-to-end pipeline (Default)")
    
    args = parser.parse_args()
    
    # If the user specified specific flags, override the default --all behavior
    has_specific_flags = any([args.prepare, args.train, args.evaluate, args.demo])
    if has_specific_flags:
        args.all = False
        
    print("=============================================================")
    print("  PLANT DISEASE CLASSIFICATION DEEP LEARNING PIPELINE")
    print("=============================================================\n")
    
    # 1. Dataset Partitioning
    if args.all or args.prepare:
        print("[+] Stage 1: Partitioning raw dataset...")
        prepare_splits()
        print("\n-------------------------------------------------------------\n")
        
    # 2. Model Training
    if args.all or args.train:
        print("[+] Stage 2: Training Transfer Learning Model...")
        train_pipeline(fine_tune=True)
        print("\n-------------------------------------------------------------\n")
        
    # 3. Model Evaluation
    if args.all or args.evaluate:
        print("[+] Stage 3: Evaluating Model on Test Dataset...")
        evaluate_model()
        print("\n-------------------------------------------------------------\n")
        
    # 4. Demo Single-Image Inference
    if args.all or args.demo:
        print("[+] Stage 4: Running demo single-image inference...")
        run_demo_inference()
        
    print("\n=============================================================")
    print("  PIPELINE PROCESSING COMPLETE")
    print("=============================================================")

def run_demo_inference():
    """
    Selects a random image from the test split and runs prediction.
    """
    if not os.path.exists(config.TEST_DIR):
        print("[!] Error: Test directory not found. Prepare the dataset first.")
        return
        
    # Find all class folders
    class_folders = [d for d in os.listdir(config.TEST_DIR) if os.path.isdir(os.path.join(config.TEST_DIR, d))]
    if not class_folders:
        print("[!] Error: No class directories found in the test split.")
        return
        
    # Select a random class and a random image from that class
    selected_class = random.choice(class_folders)
    class_path = os.path.join(config.TEST_DIR, selected_class)
    
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"[!] Error: No images found in test class folder '{selected_class}'.")
        return
        
    random_image = random.choice(images)
    image_path = os.path.join(class_path, random_image)
    
    print(f"[*] Selected random test leaf image from class '{selected_class}':")
    print(f"    Path: {image_path}")
    
    # Run prediction
    try:
        run_inference(image_path)
    except Exception as e:
        print(f"[!] Demo inference failed: {e}")

if __name__ == "__main__":
    main()
