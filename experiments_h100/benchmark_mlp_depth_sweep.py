# ==============================================================================
# MLP Depth Sweep: ResNet vs. FEN MLP over depths [20, 50, 80]
# Strict parameter matching applied dynamically at each depth.
# ==============================================================================

import os
import time
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
DEPTHS = [20, 50, 80]
MODES = ["resnet_mlp", "fen_mlp"]
SEEDS = [100, 200]
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | MLP Depth Sweep | Depths: {DEPTHS}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# --- 2. DATA GENERATION ---
def get_dataset(seed=42):
    X, y = make_classification(
        n_samples=50000,
        n_features=128,
        n_informative=96,
        n_redundant=32,
        n_classes=10,
        random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=seed)
    
    # Scale
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, val_loader

# --- 3. MODEL ARCHITECTURES ---
class MLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.linear(x))) + x

class FENMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.SiLU(),
            nn.Linear(dim // 8, dim)
        )

    def forward(self, h, E):
        f_raw = self.act(self.norm(self.linear(h))) + h
        g = torch.sigmoid(self.gate(f_raw))
        D = g * f_raw
        h_next = f_raw - D
        E_next = E + D
        return h_next, E_next

class DeepMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, mode):
        super().__init__()
        self.mode = mode
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        if mode == "resnet_mlp":
            self.layers = nn.ModuleList([MLPBlock(hidden_dim) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim, num_classes)
        elif mode == "fen_mlp":
            self.layers = nn.ModuleList([FENMLPBlock(hidden_dim) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim * 2, num_classes)
        else:
            raise ValueError(mode)

    def forward(self, x):
        h = self.input_proj(x)
        if self.mode == "fen_mlp":
            E = torch.zeros_like(h)
            for layer in self.layers:
                h, E = layer(h, E)
            combined = torch.cat([h, E], dim=-1)
            logits = self.head(combined)
        else:
            for layer in self.layers:
                h = layer(h)
            logits = self.head(h)
        return logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def choose_hidden(mode, num_layers, target_params):
    best_h = 16
    best_diff = float('inf')
    for h in range(16, 512):
        model = DeepMLP(128, h, 10, num_layers, mode)
        params = count_params(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_h = h
            best_diff = diff
    return best_h

def save_history_to_csv(history, filepath="experiments_h100/mlp_depth_sweep_history.csv"):
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

def train_one(seed, mode, num_layers, target_params, train_loader, val_loader):
    seed_everything(seed)
    h_dim = choose_hidden(mode, num_layers, target_params)
    model = DeepMLP(128, h_dim, 10, num_layers, mode).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"DEPTH SWEEP | MODE={mode.upper()} | LAYERS={num_layers} | SEED={seed} | Width={h_dim} | Params={params:,}")
    print("=" * 80)

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
            opt.step()
            
        val = evaluate(model, val_loader, criterion)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            
        print(f"  Ep {epoch:02d} | Val Loss: {val['loss']:.4f} | Val Acc: {val['acc']:.2f}% | Best Acc: {best_acc:.2f}%")

        history.append({
            "seed": seed,
            "mode": mode,
            "depth": num_layers,
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_acc": val["acc"]
        })

    print(f"  Finished {mode.upper()} Layers={num_layers} | Best Val Acc: {best_acc:.2f}% | Time: {time.time()-start:.1f}s")
    save_history_to_csv(history)
    return best_acc

if __name__ == "__main__":
    # Base parameter targets for depth scaling (scaling parameters proportionally to keep relative representation capacity)
    # At L=20, base target is 500k parameters
    # At L=50, base target is 1.1M parameters
    # At L=80, base target is 1.7M parameters
    PARAM_TARGETS = {
        20: 500000,
        50: 1100000,
        80: 1700000
    }

    print("\n" + "=" * 80)
    print("DEPTH SWEEP PARAMETER MATRIX:")
    for l in DEPTHS:
        target = PARAM_TARGETS[l]
        print(f"  Depth = {l} (Target Params: {target:,}):")
        for mode in MODES:
            h_dim = choose_hidden(mode, l, target)
            model_test = DeepMLP(128, h_dim, 10, l, mode)
            print(f"    {mode.upper():<16} (Width={h_dim:<3}) : {count_params(model_test):,} parameters")
    print("=" * 80 + "\n")

    train_loader, val_loader = get_dataset(seed=42)
    
    for l in DEPTHS:
        target = PARAM_TARGETS[l]
        print(f"\n" + "#" * 80)
        print(f"STARTING SWEEP FOR DEPTH L={l} | TARGET PARAMS = {target:,}")
        print("#" * 80)
        
        for seed in SEEDS:
            print(f"\n  Seed: {seed}")
            for mode in MODES:
                train_one(seed, mode, l, target, train_loader, val_loader)
