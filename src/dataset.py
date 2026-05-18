import os
import sys
import tensorflow as tf

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

def get_augmentation_layers():
    """
    Returns data augmentation layers to be included inside the Keras model.
    This runs on the GPU and is automatically skipped during evaluation/inference.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=config.SEED),
        tf.keras.layers.RandomRotation(0.2, seed=config.SEED),
        tf.keras.layers.RandomZoom(0.2, seed=config.SEED),
    ], name="data_augmentation")
    return data_augmentation

def load_datasets():
    """
    Loads train, validation, and test datasets from dataset_split/.
    """
    print("[*] Loading datasets from directory...")
    
    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        label_mode='categorical',
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED
    )
    
    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        label_mode='categorical',
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED
    )
    
    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        label_mode='categorical',
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED
    )
    
    # Extract and store class names
    class_names = train_ds.class_names
    print(f"[+] Loaded {len(class_names)} classes: {class_names}")
    
    # Configure datasets for high-performance (cache & prefetch)
    # Using autotune to let TensorFlow dynamically adjust prefetch sizes
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().shuffle(1000, seed=config.SEED).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

if __name__ == "__main__":
    train_ds, val_ds, test_ds, classes = load_datasets()
    print("Train element spec:", train_ds.element_spec)
