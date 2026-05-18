import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Import config & dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
import src.dataset as dataset

def evaluate_model():
    print("[*] Starting model evaluation pipeline...")
    
    # 1. Check if model exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[!] Error: Trained model not found at: {config.MODEL_SAVE_PATH}")
        sys.exit(1)
        
    # 2. Load dataset and class names
    _, _, test_ds, class_names = dataset.load_datasets()
    
    # 3. Load Trained Model
    print(f"[*] Loading model from: {config.MODEL_SAVE_PATH}")
    model = tf_load_model()
    
    # 4. Standard Metrics Evaluation
    print("[*] Evaluating model on the test dataset...")
    test_metrics = model.evaluate(test_ds, verbose=1)
    
    # Print metrics
    metric_names = model.metrics_names
    print("\n=======================================================")
    print("  TEST SPLIT PERFORMANCE")
    print("=======================================================")
    for name, val in zip(metric_names, test_metrics):
        print(f" - Test {name.capitalize()}: {val:.4f}")
    print("=======================================================\n")
    
    # 5. Class-wise Predictions
    print("[*] Collecting model predictions for detailed metrics...")
    y_true = []
    y_pred = []
    
    # Predict batches to save memory
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 6. Classification Report
    print("[*] Generating Classification Report...")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    report_path = os.path.join(config.REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[+] Saved text classification report to: {report_path}")
    
    # 7. Confusion Matrix
    print("[*] Computing Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)
    
    # 8. Training Curves
    print("[*] Generating Training Progression Curves...")
    plot_training_history()
    
    print("[+] Evaluation complete! All outputs saved in outputs/ folder.")

def tf_load_model():
    """Import tensorflow locally to avoid overhead on imports"""
    import tensorflow as tf
    # For backward compatibility with older models that used custom Lambda layers
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    
    custom_objects = {
        'preprocess_input': mobilenet_preprocess if config.BASE_MODEL_NAME == "MobileNetV2" else resnet_preprocess,
        'mobilenet_preprocess': mobilenet_preprocess,
        'resnet_preprocess': resnet_preprocess
    }
    return tf.keras.models.load_model(config.MODEL_SAVE_PATH, custom_objects=custom_objects)

def plot_confusion_matrix(cm, class_names):
    """
    Plots a highly polished, detailed multi-class Confusion Matrix using Matplotlib.
    """
    plt.figure(figsize=(14, 12))
    
    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    plt.colorbar(label='Accuracy Scale')
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)
    
    # Add labels in the matrix blocks
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Display absolute count and percentage
            text_str = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            plt.text(j, i, text_str,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black",
                     fontsize=7,
                     fontweight='semibold')
            
    plt.tight_layout()
    plt.ylabel('Ground Truth Label', fontsize=12, fontweight='bold')
    plt.xlabel('Model Predicted Label', fontsize=12, fontweight='bold')
    
    cm_path = os.path.join(config.GRAPHS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Confusion Matrix visualization to: {cm_path}")

def plot_training_history():
    """
    Plots training vs validation accuracy & loss graphs from the saved JSON history.
    """
    history_file = os.path.join(config.REPORTS_DIR, "training_history.json")
    if not os.path.exists(history_file):
        print(f"[!] Warning: Training history file '{history_file}' not found. Skipping graphs.")
        return
        
    with open(history_file, "r") as f:
        history = json.load(f)
        
    epochs = range(1, len(history["loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Loss subplot
    ax1.plot(epochs, history["loss"], 'o-', label='Training Loss', color='#D32F2F', linewidth=2)
    ax1.plot(epochs, history["val_loss"], 's--', label='Validation Loss', color='#F57C00', linewidth=2)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Accuracy subplot
    ax2.plot(epochs, history["accuracy"], 'o-', label='Training Accuracy', color='#1976D2', linewidth=2)
    ax2.plot(epochs, history["val_accuracy"], 's--', label='Validation Accuracy', color='#388E3C', linewidth=2)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    graphs_path = os.path.join(config.GRAPHS_DIR, "training_curves.png")
    plt.savefig(graphs_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Training Progression curves to: {graphs_path}")

if __name__ == "__main__":
    evaluate_model()
