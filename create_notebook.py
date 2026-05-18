import json
import os

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Plant Disease Classification Deep Learning Model\n",
                    "### Developed with Transfer Learning, Data Augmentation, and TFLite Export\n",
                    "\n",
                    "This interactive notebook demonstrates the complete process of building, training, and deploying a state-of-the-art Convolutional Neural Network (CNN) to classify plant diseases from leaf images using the Kaggle **PlantVillage** dataset.\n",
                    "\n",
                    "## Notebook Structure\n",
                    "1. **Environment Setup & Verification**: Verifies GPU acceleration (Apple Metal API on Mac).\n",
                    "2. **Dataset Split (70/15/15)**: Splits the raw directories into clean Train, Val, and Test folders.\n",
                    "3. **Data Loading & Augmentation**: Loads high-performance `tf.data.Dataset` pipelines with caching, prefetching, and random augmentations.\n",
                    "4. **Transfer Learning Model**: Builds a CNN with **MobileNetV2** as the base and custom classification/dropout heads.\n",
                    "5. **Training (2 Phases)**: Warms up classification head, then fine-tunes base-model layers with early stopping.\n",
                    "6. **Model Evaluation**: Generates classification reports, confusion matrices, and training performance charts.\n",
                    "7. **TFLite Export**: Converts the trained model into optimized FP16 TensorFlow Lite format for web/mobile deployment."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment Setup & GPU Verification\n",
                    "First, let's verify that TensorFlow is installed and successfully accesses your **Apple Silicon (M1/M2/M3) GPU** using the Apple Metal API."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import tensorflow as tf\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import os\n",
                    "\n",
                    "print(\"TensorFlow Version:\", tf.__version__)\n",
                    "print(\"Eager execution:\", tf.executing_eagerly())\n",
                    "\n",
                    "# Check for GPU acceleration\n",
                    "gpus = tf.config.list_physical_devices('GPU')\n",
                    "if gpus:\n",
                    "    print(\"\\n[+] GPU(s) available for training:\")\n",
                    "    for gpu in gpus:\n",
                    "        print(f\"  - Name: {gpu.name}, Type: {gpu.device_type}\")\n",
                    "else:\n",
                    "    print(\"\\n[-] No GPU found. Training will run on CPU (this will be slower).\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Dataset Split (70% Train, 15% Val, 15% Test)\n",
                    "Let's split the raw image folders in `/Users/dheeraj_kumar/Downloads/crop/archive/PlantVillage/PlantVillage` into partitioned dataset subdirectories."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.prepare_dataset import prepare_splits\n",
                    "\n",
                    "# Run dataset partitioner\n",
                    "prepare_splits()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. High-Performance Dataset Loading & Augmentation\n",
                    "Now, let's load our data using `tf.data.Dataset` pipelines. This handles automatic resizing to `224x224`, batching, and GPU data augmentation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.dataset import load_datasets\n",
                    "\n",
                    "# Load prefetching train/val/test splits\n",
                    "train_ds, val_ds, test_ds, class_names = load_datasets()\n",
                    "print(f\"\\n[+] Number of disease classes loaded: {len(class_names)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Visualize Sample Images with Augmentation\n",
                    "Let's see some sample leaf images with random rotations, flips, and zooms applied to them."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.dataset import get_augmentation_layers\n",
                    "\n",
                    "# Fetch a single batch of images\n",
                    "image_batch, label_batch = next(iter(train_ds))\n",
                    "augmentation_layers = get_augmentation_layers()\n",
                    "\n",
                    "plt.figure(figsize=(10, 10))\n",
                    "first_image = image_batch[0]\n",
                    "for i in range(9):\n",
                    "    ax = plt.subplot(3, 3, i + 1)\n",
                    "    # Augment the first image\n",
                    "    augmented_image = augmentation_layers(tf.expand_dims(first_image, 0), training=True)\n",
                    "    plt.imshow(augmented_image[0].numpy().astype(\"uint8\"))\n",
                    "    plt.title(class_names[np.argmax(label_batch[0])], fontsize=8)\n",
                    "    plt.axis(\"off\")\n",
                    "plt.suptitle(\"Sample Augmented Images (GPU-Accelerated On-The-Fly)\", fontsize=14, fontweight='bold')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Transfer Learning Model Setup\n",
                    "We'll build a Transfer Learning model utilizing **MobileNetV2** (highly optimized for low-latency, real-world edge deployment). The Keras model contains the preprocessing and data augmentation layers inside it for seamless inference deployment."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.model import build_transfer_learning_model\n",
                    "\n",
                    "# Build model\n",
                    "model, base_model = build_transfer_learning_model(num_classes=len(class_names))\n",
                    "model.summary()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Model Training (Warmup + Fine-Tuning)\n",
                    "Let's execute the complete training pipeline. It runs in two steps:\n",
                    "1. **Warmup**: Trains only the custom classification head while freezing MobileNetV2 weights (prevents distortion of pre-trained ImageNet weights).\n",
                    "2. **Fine-Tuning**: Unfreezes the top layers of the MobileNetV2 base model and trains with a lower learning rate (`1e-5`) for maximum accuracy."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.train import train_pipeline\n",
                    "\n",
                    "# Train the model (this will automatically save models in Keras & H5 format)\n",
                    "train_pipeline(fine_tune=True)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Model Evaluation on Testing Split\n",
                    "Let's evaluate the model on the test split. We'll generate a classification report (precision, recall, f1-score per class) and save the confusion matrix and loss/accuracy curves."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.evaluate import evaluate_model\n",
                    "\n",
                    "# Run evaluations\n",
                    "evaluate_model()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Display training curves and confusion matrix\n",
                    "Let's view the generated evaluation figures."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from PIL import Image\n",
                    "\n",
                    "# Load training curves plot\n",
                    "curves_path = \"outputs/graphs/training_curves.png\"\n",
                    "if os.path.exists(curves_path):\n",
                    "    display(Image.open(curves_path))\n",
                    "else:\n",
                    "    print(\"Training curves plot not found. Run training/eval first.\")\n",
                    "\n",
                    "# Load Confusion Matrix plot\n",
                    "cm_path = \"outputs/graphs/confusion_matrix.png\"\n",
                    "if os.path.exists(cm_path):\n",
                    "    plt.figure(figsize=(12, 10))\n",
                    "    plt.imshow(Image.open(cm_path))\n",
                    "    plt.axis('off')\n",
                    "else:\n",
                    "    print(\"Confusion Matrix plot not found. Run training/eval first.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Sample Single-Image Inference CLI Demo\n",
                    "Let's select a random image from the test set and run inference to see how it classifies leaf images!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from main import run_demo_inference\n",
                    "\n",
                    "# Run sample leaf classification\n",
                    "run_demo_inference()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.12 (venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Plant_Disease_Classification.ipynb")
    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=4)
    print(f"[+] Programmatically generated interactive notebook at: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
