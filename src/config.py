import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = "/Users/dheeraj_kumar/Downloads/crop/archive/PlantVillage/PlantVillage"
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")

# Subdirectories for splitted data
TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DIR, "test")

# Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4  # Good rate for stable transfer learning
SEED = 42

# Dataset Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model choice: 'MobileNetV2' or 'ResNet50'
BASE_MODEL_NAME = "MobileNetV2"

# Output paths
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")

# Final Saved Model Paths
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
MODEL_SAVE_H5_PATH = os.path.join(MODELS_DIR, "plant_disease_model.h5")
TFLITE_SAVE_PATH = os.path.join(MODELS_DIR, "plant_disease_model.tflite")

# Class labels list (to be filled dynamically or stored)
# Since they are loaded dynamically, we will save them in metadata.
