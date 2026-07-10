# ==============================================================================
# 2-Layer LSTM Sweep on Sequential MNIST (sMNIST) Downsampled to 400 Steps (20x20)
# Target Params = 100k. Strict parameter matching applied dynamically at startup.
# ==============================================================================

import os
import time
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision

# --- 1. CONFIGURATION ---
MODES = ["plain_lstm", "resnet_lstm", "fen_lstm", "fen_rotational_lstm"]
SEEDS = [100, 200]
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_LAYERS = 2
SEQ_LEN = 400  # 20x20 downsampled pixels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | 2-Layer LSTM sMNIST Sweep | Layers: {NUM_LAYERS} | Seq Len: {SEQ_LEN}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# --- 2. DATA DOWNLOAD & PREPARATION ---
def make_loaders():
    import kagglehub
    import pandas as pd
    
    print("Downloading MNIST via kagglehub...")
    path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
    print(f"Dataset path: {path}")
    
    train_file = os.path.join(path, "mnist_train.csv")
    test_file = os.path.join(path, "mnist_test.csv")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    y_train = train_df['label'].values
    x_train_raw = train_df.drop(columns=['label']).values.astype(np.float32) / 255.0
    
    y_test = test_df['label'].values
    x_test_raw = test_df.drop(columns=['label']).values.astype(np.float32) / 255.0
    
    x_train_t = torch.tensor(x_train_raw).view(-1, 1, 28, 28)
    x_test_t = torch.tensor(x_test_raw).view(-1, 1, 28, 28)
    
    # Downsample from 28x28 to 20x20 using bilinear interpolation
    print("Downsampling MNIST images from 28x28 to 20x20...")
    x_train_resized = F.interpolate(x_train_t, size=(20, 20), mode="bilinear", align_corners=False) # [60000, 1, 20, 20]
    x_test_resized = F.interpolate(x_test_t, size=(20, 20), mode="bilinear", align_corners=False)   # [10000, 1, 20, 20]
    
    # Flatten to sequence [Batch, 400, 1]
    x_train_seq = x_train_resized.squeeze(1).view(-1, 400, 1).numpy()
    x_test_seq = x_test_resized.squeeze(1).view(-1, 400, 1).numpy()
    
    # Subset to keep training fast (15,000 train, 2,000 test)
    # Stratified split using numpy
    np.random.seed(42)
    train_indices = []
    test_indices = []
    
    for digit in range(10):
        digit_train_idx = np.where(y_train == digit)[0]
        digit_test_idx = np.where(y_test == digit)[0]
        
        train_indices.extend(np.random.choice(digit_train_idx, size=1500, replace=False))
        test_indices.extend(np.random.choice(digit_test_idx, size=200, replace=False))
        
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    x_tr = x_train_seq[train_indices]
    y_tr = y_train[train_indices]
    x_te = x_test_seq[test_indices]
    y_te = y_test[test_indices]
    
    # Normalize features (standardization)
    mean = x_tr.mean()
    std = x_tr.std()
    x_tr = (x_tr - mean) / (std + 1e-8)
    x_te = (x_te - mean) / (std + 1e-8)
    
    train_ds = TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(x_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, test_loader

# --- 3. MODEL ARCHITECTURES ---
class DeepLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, mode):
        super().__init__()
        self.mode = mode
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Use native PyTorch nn.LSTM layers for high speed cuDNN execution
        self.cells = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True) for _ in range(num_layers)
        ])
        
        self.has_escrow = mode in ["fen_lstm", "fen_rotational_lstm"]
        
        if mode == "fen_lstm":
            self.gate = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.escrow_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim * 2, 10)
        elif mode == "fen_rotational_lstm":
            self.gate = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.escrow_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.roll_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim * 2, 10)
        else:
            self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # x shape: [B, T, in_dim]
        h = self.input_proj(x)
        
        if self.has_escrow:
            E = torch.zeros_like(h)
            
        for l in range(self.num_layers):
            # Run the entire sequence through the native LSTM layer
            out, _ = self.cells[l](h) # out shape: [B, T, hidden_dim]
            
            if self.mode == "plain_lstm":
                h = out
            elif self.mode == "resnet_lstm":
                h = out + h
            elif self.mode == "fen_lstm":
                f_raw = out + h
                g = torch.sigmoid(self.gate[l](f_raw))
                D = g * f_raw
                h = f_raw - D
                E = E + self.escrow_proj[l](D)
            elif self.mode == "fen_rotational_lstm":
                f_raw = out + h
                g = torch.sigmoid(self.gate[l](f_raw))
                D = g * f_raw
                h = f_raw - D
                gamma = torch.sigmoid(self.roll_gate[l](f_raw))
                E = (1 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + self.escrow_proj[l](D)
                
        # Take the final sequence step's representation for classification
        if self.has_escrow:
            final_rep = torch.cat([h[:, -1, :], E[:, -1, :]], dim=-1)
        else:
            final_rep = h[:, -1, :]
            
        return self.head(final_rep)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def choose_hidden(mode, target_params):
    best_h = 8
    best_diff = float('inf')
    for h in range(8, 256, 4):
        model = DeepLSTM(1, h, NUM_LAYERS, mode)
        params = count_params(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_h = h
            best_diff = diff
    return best_h

TARGET_PARAMS = 100000

def save_history_to_csv(history, filepath="experiments_h100/lstm_smnist_history.csv"):
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    file_exists = os.path.exists(filepath)
    keys = history[0].keys()
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(history)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.shape[0]
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += bs
    return {"loss": loss_sum / total, "acc": (correct / total) * 100}

def train_one(seed, mode, train_loader, val_loader):
    seed_everything(seed)
    h_dim = choose_hidden(mode, TARGET_PARAMS)
    model = DeepLSTM(1, h_dim, NUM_LAYERS, mode).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80, flush=True)
    print(f"LSTM SMNIST | MODE={mode.upper()} | SEED={seed} | Width={h_dim} | Params={params:,}", flush=True)
    print("=" * 80, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start = time.time()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(xb)
                loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
        val = evaluate(model, val_loader, criterion)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            
        print(f"  Ep {epoch:02d} | Val Loss: {val['loss']:.4f} | Val Acc: {val['acc']:.2f}% | Best Acc: {best_acc:.2f}%", flush=True)

        history.append({
            "seed": seed,
            "mode": mode,
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_acc": val["acc"]
        })

    print(f"  Finished {mode.upper()} | Best Val Acc: {best_acc:.2f}% | Time: {time.time()-start:.1f}s")
    save_history_to_csv(history)
    return best_acc

if __name__ == "__main__":
    # Print parameter count matrix at the absolute start and flush immediately
    print("\n" + "=" * 80, flush=True)
    print("PARAMETER COUNT SUMMARY (Target: 100k):", flush=True)
    for mode in MODES:
        h_dim = choose_hidden(mode, TARGET_PARAMS)
        model_test = DeepLSTM(1, h_dim, NUM_LAYERS, mode)
        print(f"  {mode.upper():<20} (Width={h_dim:<3}) : {count_params(model_test):,} parameters", flush=True)
    print("=" * 80 + "\n", flush=True)

    train_loader, val_loader = make_loaders()
    
    print("\n" + "#" * 80, flush=True)
    print(f"STARTING LSTM SWEEP | TARGET_PARAMS = {TARGET_PARAMS:,} | EPOCHS = {EPOCHS}", flush=True)
    print("#" * 80, flush=True)
    
    for seed in SEEDS:
        for mode in MODES:
            train_one(seed, mode, train_loader, val_loader)
