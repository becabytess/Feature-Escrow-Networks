# ==============================================================================
# 30-Layer Deep MLP Representation Propagation Sweep
# Task: High-dimensional non-linear classification (128 features, 10 classes)
# Sweep: Plain MLP, Residual MLP, and FEN-MLP
# Setup: 30 layers, width 128, 30 epochs, synthetic dataset, BF16 AMP on H100.
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
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION ---
MODES = ["plain_mlp", "resnet_mlp", "fen_mlp"]
SEEDS = [100, 200]
EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 128
NUM_LAYERS = 30
POLARITY_COEFF = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Deep MLP Sweep | Layers: {NUM_LAYERS} | Width: {HIDDEN_DIM}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True

# --- 2. DATA GENERATOR ---
def get_dataset(seed):
    print("Generating complex synthetic classification dataset...")
    X_raw, y_raw = make_classification(
        n_samples=60000, 
        n_features=128, 
        n_informative=64, 
        n_redundant=32,
        n_classes=10, 
        n_clusters_per_class=2,
        random_state=seed
    )
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=10000, random_state=seed)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.int64)
    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.int64)
    
    train_set = TensorDataset(train_x, train_y)
    val_set = TensorDataset(val_x, val_y)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, val_loader

# --- 3. MODEL BLOCKS & ARCHITECTURES ---

class MLPBlock(nn.Module):
    def __init__(self, dim, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
    def forward(self, x):
        out = self.act(self.norm(self.linear(x)))
        if self.use_residual:
            return out + x
        return out

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
        return h_next, E_next, g.mean()

class DeepMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, mode):
        super().__init__()
        self.mode = mode
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        if mode == "plain_mlp":
            self.layers = nn.ModuleList([MLPBlock(hidden_dim, use_residual=False) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim, out_dim)
        elif mode == "resnet_mlp":
            self.layers = nn.ModuleList([MLPBlock(hidden_dim, use_residual=True) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim, out_dim)
        elif mode == "fen_mlp":
            self.layers = nn.ModuleList([FENMLPBlock(hidden_dim) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim * 2, out_dim)
        else:
            raise ValueError(mode)

    def forward(self, x, return_stats=False):
        h = self.input_proj(x)
        
        if self.mode == "fen_mlp":
            E = torch.zeros_like(h)
            g_sum = 0.0
            for layer in self.layers:
                h, E, g = layer(h, E)
                g_sum += g
            combined = torch.cat([h, E], dim=-1)
            logits = self.head(combined)
            if return_stats:
                return logits, torch.stack([g_sum / self.num_layers, h.norm(dim=-1).mean()])
        else:
            for layer in self.layers:
                h = layer(h)
            logits = self.head(h)
            if return_stats:
                return logits, torch.stack([torch.tensor(0.0, device=x.device), h.norm(dim=-1).mean()])
                
        return logits

# --- 4. SWEEP RUNNER ---
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_history_to_csv(history, filepath="experiments_h100/deep_mlp_history.csv"):
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
    gate_sum, pipe_sum, stat_n = 0.0, 0.0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, stats = model(xb, return_stats=True)
        loss = criterion(logits, yb)
        
        bs = xb.shape[0]
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += bs
        
        if stats is not None:
            gate_sum += stats[0].item()
            pipe_sum += stats[1].item()
            stat_n += 1
            
    out = {"loss": loss_sum / total, "acc": (correct / total) * 100}
    if stat_n:
        out["gate"] = gate_sum / stat_n
        out["pipe"] = pipe_sum / stat_n
    return out

def choose_hidden(mode, target_params):
    best_h = 16
    best_diff = float('inf')
    for h in range(16, 512):
        model = DeepMLP(128, h, 10, NUM_LAYERS, mode)
        params = count_params(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_h = h
            best_diff = diff
    return best_h

TARGET_PARAMS = 650000

def train_one(seed, mode, train_loader, val_loader):
    seed_everything(seed)
    h_dim = choose_hidden(mode, TARGET_PARAMS)
    model = DeepMLP(128, h_dim, 10, NUM_LAYERS, mode).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"DEEP MLP | MODE={mode.upper()} | SEED={seed} | Width={h_dim} | Params={params:,}")
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
            
        gate_str = f" gate={val['gate']:.3f} pipe={val['pipe']:.2f}" if "gate" in val else ""
        print(f"  Ep {epoch:02d} | Val Loss: {val['loss']:.4f} | Val Acc: {val['acc']:.2f}% | Best Acc: {best_acc:.2f}%{gate_str}")

        history.append({
            "seed": seed,
            "mode": mode,
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "gate": val.get("gate", 0.0),
            "pipe": val.get("pipe", 0.0)
        })

    print(f"  Finished {mode.upper()} | Best Val Acc: {best_acc:.2f}% | Time: {time.time()-start:.1f}s")
    save_history_to_csv(history)
    return {"mode": mode, "best_acc": best_acc}

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PARAMETER COUNT SUMMARY:")
    for mode in MODES:
        h_dim = choose_hidden(mode, TARGET_PARAMS)
        model_test = DeepMLP(128, h_dim, 10, NUM_LAYERS, mode)
        print(f"  {mode.upper():<16} (Width={h_dim:<3}) : {count_params(model_test):,} parameters")
    print("=" * 80 + "\n")

    train_loader, val_loader = get_dataset(seed=42)
    results = {mode: [] for mode in MODES}
    
    print("\n" + "#" * 80)
    print(f"STARTING DEEP MLP SWEEP | TARGET_PARAMS = {TARGET_PARAMS:,} | EPOCHS = {EPOCHS}")
    print("#" * 80)
    
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n================================================================================")
        print(f"SEED BATCH {s_idx+1}/{len(SEEDS)}: SEED = {seed}")
        print(f"================================================================================")
        
        for mode in MODES:
            res = train_one(seed, mode, train_loader, val_loader)
            results[mode].append(res)
