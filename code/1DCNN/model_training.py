import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, AdamW
import os

gpu = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpu)

DATA_PATH_BASE = '../../start_kit/processed_data_fold'
NUM_FOLD = 5 

def load_fold_data(base_path, fold_id): 
    npz_file = f'{base_path}{fold_id}.npz'
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Missing K-Fold data file: {npz_file}")
        
    fold_data = np.load(npz_file, allow_pickle=True)
    
    # Load 2 set: Train và Val
    X_train, Y_train = fold_data['X_train'], fold_data['Y_train']
    X_val, Y_val = fold_data['X_val'], fold_data['Y_val']
    
    # CHangable Para
    MAX_FRAMES = X_train.shape[1]
    NUM_FEATURES = X_train.shape[2]
    NUM_CLASSES = len(fold_data['gloss_to_label'].item())
    
    # Change labels (Y) to One-Hot Encoding
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=NUM_CLASSES)
    Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=NUM_CLASSES)
    
    print(f"Fold {fold_id} loaded: Train={len(X_train)}, Val={len(X_val)}")
    return X_train, Y_train, X_val, Y_val, MAX_FRAMES, NUM_FEATURES, NUM_CLASSES

def augment_keypoints(X):
    noise = np.random.normal(0,0.02, X.shape)
    X_aug = X+ noise
    return X_aug

def build_cnn1d_model(MAX_FRAMES, NUM_FEATURES, NUM_CLASSES):
    """Build model 1D Convolutional Neural Network (1D CNN)."""
    model = Sequential([
        Input(shape=(MAX_FRAMES, NUM_FEATURES)),
        # 1. 1st Convolution Layer 1D
        # Use small kernel_size to make sample move shortly (3 continuous frames)
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # 2. 2nd Convolution Layer 1D
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        
        GlobalAveragePooling1D(), # Tối ưu hóa bằng cách lấy trung bình thay vì làm phẳng (Flatten)
        
        # 3. Output Dense
        Dense(128, activation='relu'),
        Dropout(0.5), # Regularization mạnh hơn ở lớp cuối
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    return model

def train_model(X_train, Y_train, X_val, Y_val, fold_id, MAX_FRAMES, NUM_FEATURES, NUM_CLASSES):
    # Early Stopping: Stop training if val_loss not better after 10 epochs
    es_callback = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        verbose=1, 
        restore_best_weights=True
    )
    
    lr_callback = ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.5,  #Decrease LR to 50% when plateau
        patience = 5,  #after 5 epochs don't improve
        verbose = 1
    )
    
    checkpoint_filepath = f'best_1dcnn_fold{fold_id}_model.h5'
    mc_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    optimizers = {
        'Adam': Adam(learning_rate=0.001),
        'AdamW': AdamW(learning_rate=0.001),
        'SGD': SGD(learning_rate = 0.01, momentum=0.9)
    }
    best_val_acc = 0
    best_history = None
    best_optimizer_name = None
    X_train_aug = augment_keypoints(X_train)
    
    for opt_name, opt in optimizers.items():
        current_model = build_cnn1d_model(MAX_FRAMES, NUM_FEATURES, NUM_CLASSES)
        print(f"\n--- Training with optimizer: {opt_name} ---")
        current_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history = current_model.fit(
            X_train_aug, 
            Y_train,
            validation_data=(X_val, Y_val),
            epochs=100, 
            batch_size=32,
            callbacks=[es_callback, mc_callback, lr_callback],
            verbose=2
        )
        val_acc = max(history.history['val_accuracy'])
        print(f"Best val_accuracy with {opt_name}: {val_acc:.4f}")
        
        if val_acc>best_val_acc:
            best_val_acc = val_acc
            best_history = history
            best_optimizer_name = opt_name
            
    print(f"\nBest optimizer: {best_optimizer_name} with val_accuracy={best_val_acc:.4f}")
    return best_history, checkpoint_filepath

if __name__ == '__main__':
    all_best_val_accs = []
    for fold in range(NUM_FOLD):
        print(f"\n=================== STARTING FOLD {fold} / {NUM_FOLD} ===================")
        try:
            X_train, Y_train, X_val, Y_val, MAX_FRAMES, NUM_FEATURES, NUM_CLASSES = load_fold_data(DATA_PATH_BASE, fold)
        except FileNotFoundError as e:
            print(e)
            continue
            
        history, best_model_path = train_model(
            X_train, Y_train, X_val, Y_val, fold, MAX_FRAMES, NUM_FEATURES, NUM_CLASSES
        )
        
        print(f"Fold {fold} Training Complete. Best model saved to: {best_model_path}")
        
        # 🟢 SAVE HISTORY ACCORDING TO FOLD ID
        os.makedirs('saved_history', exist_ok=True)
        np.save(f'saved_history/history_fold{fold}.npy', history.history)
        
        all_best_val_accs.append(max(history.history['val_accuracy']))
        
    print(f"\nAverage Val Accuracy over {NUM_FOLD} folds: {np.mean(all_best_val_accs):.4f}")