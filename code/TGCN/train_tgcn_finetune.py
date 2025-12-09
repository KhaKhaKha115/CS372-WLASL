import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from configs import Config
from tgcn_model import GCN_muti_att
from sign_dataset import Sign_Dataset
from train_utils import train, validation
from sklearn.model_selection import KFold, train_test_split

device = torch.device('cpu') 
print(f"Using device: {device}")

PRETRAINED_MODEL = 'TGCN_pretrained_models/asl1000/ckpt.pth'  # link topretrained model
SAVE_MODEL_TO = 'fine_tuned_tgcn.pth'
DATA_PATH_BASE = '../../start_kit/processed_data_fold'
EPOCHS = 10 
LR = 1e-4  
BATCH_SIZE = 16

NUM_FOLDS = 5

def load_fold_data(fold_id):
    npz_file = f'{DATA_PATH_BASE}{fold_id}.npz'
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Missing K-Fold data file: {npz_file}")
        
    data = np.load(npz_file, allow_pickle=True)
    
    # Load 2 sets: Train và Val
    X_train, Y_train = data['X_train'].astype(np.float32), data['Y_train'].astype(np.int64)
    X_val, Y_val = data['X_val'].astype(np.float32), data['Y_val'].astype(np.int64)
    
     # ✅ Debug: xác nhận dữ liệu
    print(f"DEBUG: Fold {fold_id} shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    
    # para for model
    num_features = X_train.shape[2]
    num_classes = len(data['gloss_to_label'].item())
    
    print(f"Fold {fold_id} loaded: Train={len(X_train)}, Val={len(X_val)}")
    return train_dataset, val_dataset, num_features, num_classes

def load_compatible_weights(model, pretrained_path, device, fold_id, target_input_dim):
    """
    Load compatible pretrained weights, EXCLUDING the input layer ('gc1.weight', 'gc1.att',...) 
    and output layer ('fc_out').
    """
    if not os.path.exists(pretrained_path):
        print(f"❌ Warning: Pretrained model not found at {pretrained_path}")
        return model

    pretrained_dict = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()

    skip_prefixes = ('gc1', 'bn1', 'fc_out') 
    compatible_dict = {}
    skipped_keys = []
    
    for k, v in pretrained_dict.items():
        is_skipped_prefix = any(k.startswith(prefix) for prefix in skip_prefixes)
        
        if k in model_dict:
            if v.shape == model_dict[k].shape and not is_skipped_prefix:
                compatible_dict[k] = v
            else:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)

    model.load_state_dict(compatible_dict, strict=False) 
        
    print(f"✅ Loaded compatible pretrained weights for FOLD {fold_id}.")
    print(f"   - Compatible Layers: {len(compatible_dict)}")
    print(f"   - Skipped Layers (Count: {len(skipped_keys)}): {skipped_keys[:5]}...") 
    
    return model


# --- K-FOLD (FINETUNE LOOP) ---
all_best_val_accs = []
for fold in range(NUM_FOLDS):
    print(f"\n=================== STARTING FOLD {fold} / {NUM_FOLDS} ===================")
    try:
        train_dataset, val_dataset, NUM_FEATURES_ACTUAL, NUM_CLASSES = load_fold_data(fold)
    except FileNotFoundError as e:
        print(e)
        continue
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_nodes = train_dataset.tensors[0].shape[1]
    actual_num_features = train_dataset.tensors[0].shape[2]
    print(f"DEBUG: Data shape features: {actual_num_features}")
    HIDDEN_FEATURE_SIZE = 256
    # --- INITIALIZE NEW MODEL FOR EACH FOLD ---
    model = GCN_muti_att(
        input_feature=train_dataset.tensors[0].shape[2],
        hidden_feature=HIDDEN_FEATURE_SIZE,
        num_class=NUM_CLASSES,
        num_nodes=num_nodes,     
        p_dropout=0.5,
        num_stage=2
    ).to(device)
    
    # --- LOAD PRETRAINED ---
    model = load_compatible_weights(model, PRETRAINED_MODEL, device, fold, actual_num_features)

    # --- OPTIMIZER ---
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_acc = 0
    SAVE_MODEL_TO_FOLD = f'fine_tuned_tgcn_fold{fold}.pth'
    
    for epoch in range(EPOCHS):
        # train
        train_losses, train_scores, train_gts, train_preds = train(
            log_interval=1,
            model = model,              # model
            train_loader=train_loader,       # train_loader
            optimizer=optimizer,          # optimizer
            epoch=epoch,              # epoch
        )
        # validation
        val_loss, val_score, val_gts, val_preds, _ = validation(
            model=model,                 # model
            test_loader=val_loader,     # loader
            epoch=epoch,                 # epoch
            save_to=None
        )
        print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Top1 Acc={val_score[0]:.4f}")
        if val_score[0] > best_val_acc:
            best_val_acc = val_score[0]
            torch.save(model.state_dict(), SAVE_MODEL_TO_FOLD)
            print(f"✅ Saved best finetuned model for FOLD {fold} at epoch {epoch} with acc={best_val_acc:.4f}")
            
    all_best_val_accs.append(best_val_acc)

print("✅ Finetuning complete!")
print(f"Average Val Accuracy over {NUM_FOLDS} folds: {np.mean(all_best_val_accs):.4f}")