import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model

# Import config & dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
import src.dataset as dataset

def build_transfer_learning_model(num_classes, apply_augmentation=True):
    """
    Builds a Transfer Learning model using MobileNetV2 or ResNet50 base models.
    Includes data preprocessing and data augmentation layers directly in the model.
    """
    print(f"[*] Building Transfer Learning model with base: {config.BASE_MODEL_NAME}")
    
    # Input layer
    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    
    x = inputs
    
    # 1. Apply Data Augmentation (only during training)
    if apply_augmentation:
        augmentation_layers = dataset.get_augmentation_layers()
        x = augmentation_layers(x)
        
    # 2. Apply Model-Specific Preprocessing (rescale/normalize)
    if config.BASE_MODEL_NAME == "MobileNetV2":
        # MobileNetV2 expects values in [-1, 1]
        x = layers.Rescaling(scale=1.0/127.5, offset=-1.0, name="mobilenet_v2_preprocess")(x)
        
        # Load pre-trained Base Model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
    elif config.BASE_MODEL_NAME == "ResNet50":
        # ResNet50 expects zero-centering and BGR channel format
        # 1. RGB to BGR conversion
        x = layers.Lambda(
            lambda img: tf.reverse(img, axis=[-1]),
            name="rgb_to_bgr"
        )(x)
        # 2. Subtract ImageNet BGR means
        imagenet_means = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x = layers.Lambda(
            lambda img: img - imagenet_means,
            name="bgr_mean_subtraction"
        )(x)
        
        # Load pre-trained Base Model
        base_model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError(f"[!] Unknown base model: {config.BASE_MODEL_NAME}. Use 'MobileNetV2' or 'ResNet50'")

    # Freeze base model weights
    base_model.trainable = False
    print(f"[*] Base model ({config.BASE_MODEL_NAME}) loaded and frozen.")
    
    # 3. Connect Base Model
    x = base_model(x, training=False)  # Ensure batchnorm layers run in inference mode
    
    # 4. Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.5, name="dropout_regularization")(x)
    
    # 5. Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_predictions")(x)
    
    # Build complete model
    model = Model(inputs=inputs, outputs=outputs, name=f"PlantDisease_{config.BASE_MODEL_NAME}")
    
    return model, base_model

def setup_fine_tuning(model, base_model, fine_tune_at=100):
    """
    Unfreezes the base model from the specified layer index ('fine_tune_at') onwards
    to perform precise model adaptation.
    """
    print(f"[*] Setting up fine-tuning: unfreezing layers from index {fine_tune_at} onwards...")
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Re-freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    print(f"[+] Fine-tuning configured. Unfrozen layers: {len(base_model.layers) - fine_tune_at} / {len(base_model.layers)}")

if __name__ == "__main__":
    # Test building the model
    model, base_model = build_transfer_learning_model(num_classes=15)
    model.summary()
