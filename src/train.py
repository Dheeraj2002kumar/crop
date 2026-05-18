import os
import sys
import json
import tensorflow as tf

# Import config, dataset, and model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
import src.dataset as dataset
import src.model as model_builder

def train_pipeline(fine_tune=True):
    # Ensure outputs directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.GRAPHS_DIR, exist_ok=True)
    
    print("[*] Starting model training pipeline...")
    
    # 1. Load Data
    train_ds, val_ds, test_ds, class_names = dataset.load_datasets()
    num_classes = len(class_names)
    
    # Save class names as metadata JSON for deployment
    metadata = {
        "classes": class_names,
        "base_model": config.BASE_MODEL_NAME,
        "input_shape": config.IMAGE_SIZE + (3,)
    }
    metadata_path = os.path.join(config.OUTPUTS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Class names and metadata saved to: {metadata_path}")
    
    # 2. Build Model
    model, base_model = model_builder.build_transfer_learning_model(
        num_classes=num_classes,
        apply_augmentation=True
    )
    
    # 3. Phase 1: Train Classification Head (Base Model Frozen)
    print("\n=======================================================")
    print("  PHASE 1: Training Classification Head (Warmup)")
    print("=======================================================")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Run warmup training
    warmup_epochs = max(1, config.EPOCHS // 2)
    print(f"[*] Training for {warmup_epochs} warmup epochs...")
    
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks
    )
    
    history_dict = {
        "loss": history_warmup.history["loss"],
        "accuracy": history_warmup.history["accuracy"],
        "precision": history_warmup.history["precision"],
        "recall": history_warmup.history["recall"],
        "val_loss": history_warmup.history["val_loss"],
        "val_accuracy": history_warmup.history["val_accuracy"],
        "val_precision": history_warmup.history["val_precision"],
        "val_recall": history_warmup.history["val_recall"]
    }
    
    # 4. Phase 2: Fine-Tuning (Unfreeze Top Layers)
    if fine_tune:
        print("\n=======================================================")
        print("  PHASE 2: Fine-Tuning Unfrozen Base Model Layers")
        print("=======================================================")
        
        # Unfreeze from layer 100 onwards (MobileNetV2 has 154 layers total)
        fine_tune_at = 100 if config.BASE_MODEL_NAME == "MobileNetV2" else 120
        model_builder.setup_fine_tuning(model, base_model, fine_tune_at=fine_tune_at)
        
        # Recompile with a very low learning rate to prevent massive weight changes
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE * 0.1),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )
        
        # Fine-tune training
        fine_tune_epochs = config.EPOCHS - warmup_epochs
        print(f"[*] Fine-tuning for {fine_tune_epochs} epochs...")
        
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS,
            initial_epoch=warmup_epochs,
            callbacks=callbacks
        )
        
        # Merge histories
        for metric in history_dict.keys():
            history_dict[metric].extend(history_finetune.history[metric])
            
    # Save combined history to JSON
    history_file = os.path.join(config.REPORTS_DIR, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"[+] Training history saved to: {history_file}")
    
    # Save the final best model in both native Keras (.keras) and legacy H5 (.h5) formats
    print(f"[*] Saving final trained model in native Keras format...")
    try:
        model.save(config.MODEL_SAVE_PATH)
        print(f"[+] Model saved in native Keras format at: {config.MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"[!] Warning: Could not save in native Keras format directly: {e}")
        
    print(f"[*] Saving model copy in legacy .h5 format...")
    try:
        model.save(config.MODEL_SAVE_H5_PATH)
        print(f"[+] Model saved in legacy .h5 format at: {config.MODEL_SAVE_H5_PATH}")
    except Exception as e:
        print(f"[!] Warning: Could not save in legacy .h5 format directly: {e}")
        
    # 5. Export to TFLite for Edge Deployment (React/Mobile)
    print("\n=======================================================")
    print("  PHASE 3: Exporting Optimized TensorFlow Lite Model")
    print("=======================================================")
    try:
        print("[*] Converting Keras model to TFLite format with Float16 quantization...")
        # Convert the trained model directly
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        with open(config.TFLITE_SAVE_PATH, "wb") as f:
            f.write(tflite_model)
        print(f"[+] TFLite model successfully exported to: {config.TFLITE_SAVE_PATH}")
        print(f"[+] Model file sizes:")
        print(f"    - Keras: {os.path.getsize(config.MODEL_SAVE_PATH) / (1024*1024):.2f} MB")
        print(f"    - TFLite (Float16): {os.path.getsize(config.TFLITE_SAVE_PATH) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[!] Error exporting to TFLite: {e}")
        
    print("\n[+] Model training and saving complete!")

if __name__ == "__main__":
    train_pipeline()
