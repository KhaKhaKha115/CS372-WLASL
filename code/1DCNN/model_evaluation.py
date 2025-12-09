import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = '../../start_kit/processed_data' 
MODEL_PATH = 'best_1dcnn_model.h5'
TARGET_GLOSSES = ['book', 'friend', 'go', 'yes', 'no']

def load_test_data(base_path):    
    test_data = np.load(f'{base_path}_test.npz', allow_pickle=True)
    X_test = test_data['X']
    Y_test = test_data['Y']
    
    train_data = np.load(f'{base_path}_train.npz', allow_pickle=True)
    gloss_to_label = train_data['gloss_to_label'].item()
    
    print(f"Test Data Loaded: Samples={len(X_test)}")
    return X_test, Y_test, gloss_to_label

def evaluate_and_plot(X_test, Y_test, gloss_to_label):    
    if not os.path.exists(MODEL_PATH):
        print(f"🛑 ERROR: Can't find model at {MODEL_PATH}")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
    
    loss, accuracy = model.evaluate(
        X_test, 
        tf.keras.utils.to_categorical(Y_test, num_classes=len(TARGET_GLOSSES)), 
        verbose=2
    )
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = Y_test
    
    cm = tf.math.confusion_matrix(y_true_classes, y_pred_classes).numpy()
    
    labels = [k for k,v in sorted(gloss_to_label.items(), key=lambda x: x[1])]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels, 
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    precision = tf.keras.metrics.Precision()(y_true_classes, y_pred_classes).numpy()
    recall = tf.keras.metrics.Recall()(y_true_classes, y_pred_classes).numpy()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    
def plot_training_curves(history):
    loss = history_dict.get('loss')
    val_loss = history_dict.get('val_loss')
    acc = history_dict.get('accuracy')
    val_acc = history_dict.get('val_accuracy')
    
    # --- LOSS ---
    plt.subplot(1, 2, 1)
    if loss is not None:
        plt.plot(loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # --- ACCURACY ---
    plt.subplot(1, 2, 2)
    if acc is not None:
        plt.plot(acc, label='Training Accuracy')
    if val_acc is not None:
        plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == '__main__':
    X_test, Y_test, gloss_to_label = load_test_data(DATA_PATH)
    
    history_path = 'saved_history/history.npy'
    if os.path.exists(history_path):
        history_dict = np.load(history_path, allow_pickle=True).item()
        plot_training_curves(history_dict)
    evaluate_and_plot(X_test, Y_test, gloss_to_label)
