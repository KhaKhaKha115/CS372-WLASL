import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from TGCN.tgcn_model import GCN_muti_att

DATA_PATH_BASE = '../start_kit/processed_data_fold'
NUM_FOLD = 5

CNN_MODEL_PATH_FORMAT = '1DCNN/best_1dcnn_fold{i}_model.h5'
TGCN_MODEL_PATH_FORMAT = 'TGCN/fine_tuned_tgcn_fold{i}.pth'
HISTORY_PATH_FORMAT = '1DCNN/saved_history/history_fold{i}.npy'

# Fixed paras of TGCN (based on TGCN/train_tgcn_finetune.py)
TGCN_HIDDEN_FEATURE = 256
TGCN_DROPOUT = 0.5
TGCN_NUM_STAGE = 2

SAVE_DIR = 'evaluation_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)


def load_fold_test_data(base_path, fold_id):
    test_file = f'{base_path}{fold_id}.npz'
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Missing K-Fold data file: {npz_file}")
        
    fold_data = np.load(test_file, allow_pickle=True)
    
    X_test = fold_data['X_test']
    Y_test = fold_data['Y_test']
    gloss_to_label = fold_data['gloss_to_label'].item()
    NUM_CLASSES = len(gloss_to_label)

    print(f"Loaded Test Set (Fold {fold_id}): {len(X_test)} samples")
    return X_test, Y_test, gloss_to_label, NUM_CLASSES

def plot_confusion_matrix(cm, labels, model_name):
    plt.figure(figsize=(7, 5))
    label_names = [label for label in labels]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    save_path = f"{SAVE_DIR}/confusion_matrix_{model_name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")
   
    
def compute_metrics(y_true, y_pred):
    precision = tf.keras.metrics.Precision()(y_true, y_pred).numpy()
    recall = tf.keras.metrics.Recall()(y_true, y_pred).numpy()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1


def evaluate_cnn(X_test, Y_test, fold_id, NUM_CLASSES):
    cnn_model_path = CNN_MODEL_PATH_FORMAT.format(i=fold_id)
    
    if not os.path.exists(cnn_model_path):
        print("CNN model for Fold {fold_id} not found at {cnn_model_path}")
        return None

    model = tf.keras.models.load_model(cnn_model_path)
    print(f"Loaded 1D-CNN model for Fold {fold_id}")

    # LATENCY test
    start = time.time()
    preds = model.predict(X_test, verbose=0)
    latency = (time.time() - start) * 1000

    y_pred_classes = preds.argmax(axis=1)
    cm = tf.math.confusion_matrix(Y_test, y_pred_classes).numpy()

    precision, recall, f1 = compute_metrics(Y_test, y_pred_classes)
    loss, accuracy = model.evaluate(
        X_test,
        tf.keras.utils.to_categorical(Y_test, num_classes=NUM_CLASSES),
        verbose=0
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": latency,
        "cm": cm
    }


def evaluate_tgcn(X_test, Y_test,fold_id, NUM_CLASSES):
    tgcn_model_path = TGCN_MODEL_PATH_FORMAT.format(i=fold_id)
    if not os.path.exists(tgcn_model_path):
        print(f"TGCN model for Fold {fold_id} not found at {tgcn_model_path}")
        return None

    # Load PyTorch model
    num_nodes_actual = X_test.shape[1]
    num_features_actual = X_test.shape[2]
    
    model = GCN_muti_att(
        input_feature=num_features_actual, 
        hidden_feature=TGCN_HIDDEN_FEATURE, 
        num_class=NUM_CLASSES,
        num_nodes=num_nodes_actual,
        p_dropout=TGCN_DROPOUT,
        num_stage=TGCN_NUM_STAGE).to('cpu')
    model.load_state_dict(torch.load(tgcn_model_path, map_location="cpu"))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=1)

    preds = []
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    
    start = time.time()

    with torch.no_grad():
        for i, (x,) in enumerate(loader):
            x=x.to("cpu")
            out = model(x)
            # Loss
            y_true_tensor = torch.tensor([Y_test[i]], dtype=torch.long)
            loss = criterion(out, y_true_tensor)
            total_loss += loss.item()
            
            preds.append(torch.argmax(out, dim=1).item())

    latency = (time.time() - start) * 1000
    preds = np.array(preds)

    avg_loss = total_loss / len(loader)
    cm = tf.math.confusion_matrix(Y_test, preds).numpy()
    precision, recall, f1 = compute_metrics(Y_test, preds)
    accuracy = (preds == Y_test).mean()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": latency,
        "cm": cm
    }
    
def plot_training_curves(fold_id):
    history_path = HISTORY_PATH_FORMAT.format(i=fold_id)
    if not os.path.exists(history_path):
        print("No training history found for fold {fold_id} at {history_path}")
        return

    history = np.load(history_path, allow_pickle=True).item()

    plt.figure(figsize=(12, 5))

    # LOSS
    plt.subplot(1, 2, 1)
    plt.plot(history.get("loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Validation Loss")
    plt.title("Loss over Epochs (Fold {fold_id})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ACCURACY
    plt.plot(history.get("accuracy", []), label="Train Accuracy")
    plt.plot(history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title(f"Accuracy over Epochs (Fold {fold_id})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    save_path = f"{SAVE_DIR}/training_curves_cnn_fold{fold_id}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved 1DCNN training curves: {save_path}")
    
def plot_comparison(results):
    models = list(results.keys())
    accs = [results[m]["accuracy"] for m in models]
    losses = [results[m]["loss"] for m in models]

    plt.figure(figsize=(12, 5))
    # ACCURACY COMPARISON
    plt.subplot(1, 2, 1)
    plt.bar(models, accs, color=['skyblue', 'salmon'])
    plt.ylabel("Accuracy (Average)")
    plt.title("Accuracy Comparison")
    
    # LOSS COMPARISON
    plt.subplot(1, 2, 2)
    plt.bar(models, losses, color=['lightgreen', 'darkorange'])
    plt.ylabel("Loss (Average)")
    plt.title("Loss Comparison")
    
    save_path = f"{SAVE_DIR}/comparison_summary.png"
    plt.savefig(save_path)
    plt.close()

    print("✅ Saved accuracy comparison graph.")
    

if __name__ == "__main__":
    all_cnn_results = []
    all_tgcn_results = []
    
    for fold in range(NUM_FOLD):
        print(f"\n======== FOLD {fold} / {NUM_FOLD} ========")
        try:
            X_test, Y_test, gloss_to_label, NUM_CLASSES = load_fold_test_data(DATA_PATH_BASE, fold)
        except FileNotFoundError as e:
            print(e)
            continue
            
        labels = [k for k,v in sorted(gloss_to_label.items(), key=lambda x: x[1])]
        
        # ---- CNN ----
        print("\nEvaluating 1D-CNN Fold {fold}...")
        cnn_res = evaluate_cnn(X_test, Y_test, fold, NUM_CLASSES)
        if cnn_res:
            all_cnn_results.append(cnn_res)
            plot_confusion_matrix(cnn_res["cm"], labels, f"CNN_Fold{fold}")
            plot_training_curves(fold)
        
        # ---- TGCN ----
        print("\nEvaluating TGCN Fold {fold}...")
        tgcn_res = evaluate_tgcn(X_test, Y_test, fold, NUM_CLASSES)
        if tgcn_res:
            all_tgcn_results.append(tgcn_res)
            plot_confusion_matrix(tgcn_res["cm"], labels, f"TGCN_Fold{fold}")

    if all_cnn_results and all_tgcn_results:
        # average results for folds
        avg_results = {
            "CNN": {"accuracy": np.mean([r["accuracy"] for r in all_cnn_results]),
                    "f1": np.mean([r["f1"] for r in all_cnn_results]),
                    "latency_ms": np.mean([r["latency_ms"] for r in all_cnn_results]),
                    "loss": np.mean([r["loss"] for r in all_cnn_results])},
            
            "TGCN": {"accuracy": np.mean([r["accuracy"] for r in all_tgcn_results]),
                     "f1": np.mean([r["f1"] for r in all_tgcn_results]),
                     "latency_ms": np.mean([r["latency_ms"] for r in all_tgcn_results]),
                     "loss": np.mean([r["loss"] for r in all_tgcn_results])}
        }
        print("\n================ FINAL SUMMARY ================")
        print("--- 1D CNN ---")
        print(f"Average Accuracy: {avg_results['CNN']['accuracy']:.4f}")
        print(f"Average F1-Score: {avg_results['CNN']['f1']:.4f}")
        print(f"Average Loss: {avg_results['CNN']['loss']:.4f}")
        print(f"Average Latency: {avg_results['CNN']['latency_ms']:.2f} ms/sample")
        
        print("\n--- TGCN ---")
        print(f"Average Accuracy: {avg_results['TGCN']['accuracy']:.4f}")
        print(f"Average F1-Score: {avg_results['TGCN']['f1']:.4f}")
        print(f"Average Loss: {avg_results['TGCN']['loss']:.4f}")
        print(f"Average Latency: {avg_results['TGCN']['latency_ms']:.2f} ms/sample")
        # ---- Comparison ----
        plot_comparison(avg_results)
    else:
        print("\n Don't have enough data to compare")