import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

def load_image(image_path):
    """
    Loads and resizes an RGB leaf image.
    """
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at '{image_path}'")
        sys.exit(1)
        
    try:
        img = Image.open(image_path).convert('RGB')
        # Resize to model input shape
        img_resized = img.resize(config.IMAGE_SIZE)
        
        # Convert to numpy array and add batch dimension
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 224, 224, 3)
        return img_array, img
    except Exception as e:
        print(f"[!] Error reading image '{image_path}': {e}")
        sys.exit(1)

def run_inference(image_path, model_path=config.MODEL_SAVE_PATH):
    # 1. Load Metadata (to get class names)
    metadata_path = os.path.join(config.OUTPUTS_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"[!] Error: Metadata file '{metadata_path}' not found. Run training first.")
        sys.exit(1)
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    class_names = metadata["classes"]
    
    # 2. Check if Keras model exists
    if not os.path.exists(model_path):
        print(f"[!] Error: Trained model not found at '{model_path}'")
        sys.exit(1)
        
    # 3. Load Model
    print(f"[*] Loading deep learning model from '{model_path}'...")
    import tensorflow as tf
    # For backward compatibility with older models that used custom Lambda layers
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    
    custom_objects = {
        'preprocess_input': mobilenet_preprocess if config.BASE_MODEL_NAME == "MobileNetV2" else resnet_preprocess,
        'mobilenet_preprocess': mobilenet_preprocess,
        'resnet_preprocess': resnet_preprocess
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # 4. Load & Preprocess single image
    print(f"[*] Processing image: '{image_path}'")
    img_array, original_img = load_image(image_path)
    
    # 5. Run Prediction
    # Note: Model includes preprocessing inside it, so we pass raw [0-255] float array
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top predicted index and confidence
    top_idx = np.argmax(predictions)
    top_label = class_names[top_idx]
    top_conf = predictions[top_idx] * 100
    
    print("\n=======================================================")
    print("  INFERENCE RESULTS")
    print("=======================================================")
    print(f"  Predicted Disease: {top_label}")
    print(f"  Confidence:        {top_conf:.2f}%")
    print("=======================================================\n")
    
    # Display top-3 predictions
    print("[*] Top-3 Predictions:")
    top_3_indices = np.argsort(predictions)[::-1][:3]
    for i, idx in enumerate(top_3_indices):
        label = class_names[idx]
        conf = predictions[idx] * 100
        print(f"  {i+1}. {label:<45} ({conf:.2f}%)")
        
    return top_label, top_conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plant disease from leaf image.")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image.")
    parser.add_argument("--model", type=str, default=config.MODEL_SAVE_PATH, help="Path to trained Keras model.")
    args = parser.parse_args()
    
    run_inference(args.image, args.model)
