import re
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- CONFIGURATION ---
KEYPOINTS_FOLDER = 'keypoints_data'
JSON_FILE_PATH = 'WLASL_v0.3.json'
TARGET_GLOSSES = ['book', 'friend', 'go', 'yes', 'no']
OUTPUT_FILE_BASE = 'processed_data' # Create: processed_data_train.npz, ...

K = 5   # number of folds

def extract_youtube_hash(url):
    """extract sequence Hash 11 characters from URL YouTube."""
    if 'youtube' in url or 'youtu.be' in url:
        # Regex find sequence Hash 11 characters
        match = re.search(r'(?:v=|youtu\.be/|embed/)([^"&?\/\s]{11})', url)
        return match.group(1) if match else None
    return None

def load_and_prepare_data():
    # Load keypoints data, label, and group label keypoints
    
    # 1. Mapping ID video to Gloss
    video_to_gloss = {}
    with open(JSON_FILE_PATH, 'r') as f:
        full_data = json.load(f)
        
    for entry in full_data:
        if entry['gloss'] in TARGET_GLOSSES:
            gloss = entry['gloss']
            
            for inst in entry['instances']:
                wlasl_video_id = str(inst['video_id'])
                url = inst['url']
                
                # Case 1: Map WLASL ID
                video_to_gloss[wlasl_video_id] = gloss
                
                # Case 2: Map Youtube Hash
                youtube_hash = extract_youtube_hash(url)
                if youtube_hash:
                    video_to_gloss[youtube_hash] = gloss
                
    # 2. Load keypoints and label
    all_X = [] #store keypoint (time sequence)
    all_Y = [] #store label
    gloss_to_label = {gloss: i for i, gloss in enumerate(TARGET_GLOSSES)}
    
    print(f'Loading data from {KEYPOINTS_FOLDER}...')
    
    #Loop through all file npy in Keypoints_data
    for filename in tqdm(os.listdir(KEYPOINTS_FOLDER)):
        if filename.endswith('.npy'):
            video_id = filename.split('.')[0]
            
            if video_id in video_to_gloss:
                gloss = video_to_gloss[video_id]
                
                keypoints = np.load(os.path.join(KEYPOINTS_FOLDER, filename))
                
                all_X.append(keypoints)
                all_Y.append(gloss_to_label[gloss])
    
    print(f'Loaded {len(all_X)} valid samples')
    return all_X, np.array(all_Y), gloss_to_label

def normalize_data(X):
    # Use StandardScaler to normalize keypoint
    # Keypoint is time sequence data --> need to temporarily flatten to calculate mean/std
    
    max_len = max(len(seq) for seq in X)
    
    padded_X = []
    for seq in X:
        # padding sequences <= 0
        pad_width = ((0, max_len-len(seq)), (0,0))
        padded_seq = np.pad(seq, pad_width=pad_width, mode='constant', constant_values = 0)
        padded_X.append(padded_seq)
    X_padded = np.stack(padded_X) # Shape: [N, Max_Len, Features]
    
    # Nomalize
    N,T,F = X_padded.shape
    X_flat = X_padded.reshape(-1,F)
    
    scaler = StandardScaler()
    scaler.fit(X_flat)
    X_normalized_flat = scaler.transform(X_flat)
    X_normalized = X_normalized_flat.reshape(N,T,F)
    
    print('Data normalized and padded')
    return X_normalized, max_len

def create_kfold_splits(X, Y, gloss_to_label, max_len):
    """Create 5-fold cross validation datasets and save .npz for each fold."""

    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    fold_id = 0
    for train_val_idx, test_idx in kf.split(X):
        X_temp, Y_temp = X[train_val_idx], Y[train_val_idx]
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_temp, Y_temp, test_size=0.10, random_state=42, stratify=Y_temp
        )
        
        X_test, Y_test = X[test_idx], Y[test_idx]

        out_path = f"{OUTPUT_FILE_BASE}_fold{fold_id}.npz"

        np.savez_compressed(
            out_path,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            max_len=max_len,
            gloss_to_label=gloss_to_label
        )
        print(f"✅ Saved fold {fold_id}: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test → {out_path}")
        fold_id += 1
        
    print("\n================ K-FOLD CREATION COMPLETE ================")
    print(f"Created {K} folds. Each fold contains:")
    print(" - Train Set (90%)")
    print(" - Validation Set (10%% of Train+Val set)")
    print(" - Unique Test Set (20%% of total data)")
    print("===========================================================")

def calculate_velocity_features(X_padded):
    """
    Calculate velocity keypoints (delta keypoints). [N, T, F] -> [N, T, F]
    V_t = X_t - X_{t-1}. Frame V_0 sẽ is padding = 0.
    """
    X_padded = X_padded.astype(np.float32)
    N, T, F = X_padded.shape
    
    if T <= 1:
        # Nếu T <= 1, trả về khung hình 0 để tránh lỗi shape
        return np.zeros((N, T, F), dtype=np.float32) 

    #  V_t = X_t - X_(t-1). X_delta: [N, T-1, F]
    X_delta = X_padded[:, 1:, :] - X_padded[:, :-1, :]      

    # Pad thêm một khung hình 0 ở đầu để độ dài T khớp với X_norm
    X_velocity = np.pad(X_delta, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0) # [N, T, F]

    print(f"Velocity Features Calculated. Dim: {X_velocity.shape[2]}")
    return X_velocity.astype(np.float32)

def calculate_frame_stacking_features(X_padded, stack_depth=4):
    """
    Apply Frame Stacking (X_t, X_{t-1}, ..., X_{t-D+1})
    X_padded with shape (N, T, F).
    Output will have shape (N, T - (D-1), F * D)
    """
    N, T, F = X_padded.shape

    if T < stack_depth:
        min_T = max_len 
        print(f"⚠️ Warning: T={T} too short (need {stack_depth}).")
        
    X_list = []

    for i in range(stack_depth - 1, T):
        # Stack (X_i, X_{i-1}, X_{i-2}, X_{i-3})
        stack = X_padded[:, i - stack_depth + 1 : i + 1, :].reshape(N, 1, -1)
        X_list.append(stack)

    # X_combined has shape (N, T', F*D)
    X_combined = np.concatenate(X_list, axis=1)

    print(f"Frame Stacking Complete: New Feature Dim: {X_combined.shape[2]} ({F} * {stack_depth})")
    return X_combined.astype(np.float32)

if __name__ == '__main__':
    X, Y, gloss_to_label = load_and_prepare_data()

    if len(X) > 0:
        X_norm, max_len = normalize_data(X)

        X_vel = calculate_velocity_features(X_norm) # [N, T, 63]
        X_stacked_vel = np.concatenate([X_norm, X_vel], axis=2)
        # FEATURE ENGINEERING
        X_processed = calculate_frame_stacking_features(X_norm)

        # Max_len mới = T - 1
        new_max_len = X_processed.shape[1]

        # Create K-FOLD from database with FEATURE ENGINEERING
        create_kfold_splits(X_processed, Y, gloss_to_label, new_max_len)

    else:
        print("ERROR: Cannot find valid samples.")