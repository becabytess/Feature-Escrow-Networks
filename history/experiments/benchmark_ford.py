# ==============================================================================
# FEN vs Baselines Benchmarking on FordA Time-Series Classification (UCR Benchmark)
# Task: Univariate sequence classification (500 steps)
# Sweep: 10-Seed Alternating Benchmark of FEN Subtractive vs. Rotational vs. LSTM
# ==============================================================================

import os
import time
import random
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

# --- 1. CONFIGURATION ---
SEED = 2026  # Random seed to verify robustness and eliminate "luck"
TARGET_PARAMS = 100000
AUTO_MATCH_PARAMS = True
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 50  # Must be 50 to capture standard FEN's late-gestation phase transition
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the seed index to resume from (e.g. set to 3 to skip the first 3 seeds we already ran)
START_SEED_INDEX = 0

# Polarity Regularization Coefficient to sharpen gating decisions
POLARITY_COEFF = 0.05

print(f"Using device: {DEVICE}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 2. DATASET DOWNLOADING & PREPROCESSING ---
def download_dataset():
    os.makedirs('data', exist_ok=True)
    train_path = 'data/FordA_TRAIN.tsv'
    test_path = 'data/FordA_TEST.tsv'
    
    # Direct GitHub raw download links
    train_url = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv'
    test_url = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv'
    
    if not os.path.exists(train_path):
        print("Downloading FordA training set from GitHub...")
        urllib.request.urlretrieve(train_url, train_path)
        print("Download complete.")
        
    if not os.path.exists(test_path):
        print("Downloading FordA testing set from GitHub...")
        urllib.request.urlretrieve(test_url, test_path)
        print("Download complete.")
        
    return train_path, test_path

class FordADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # Shape: [N, seq_len, 1]
        self.y = torch.tensor(y, dtype=torch.long) # Shape: [N]
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]

def prepare_data(seed):
    train_path, test_path = download_dataset()
    
    # Load files using sep=None to dynamically handle space or tab delimiters
    train_df = pd.read_csv(train_path, sep=None, engine='python', header=None)
    test_df = pd.read_csv(test_path, sep=None, engine='python', header=None)
    
    y_train = train_df[0].values
    y_train = np.where(y_train == -1, 0, 1)
    X_train = train_df.iloc[:, 1:].values
    
    y_test = test_df[0].values
    y_test = np.where(y_test == -1, 0, 1)
    X_test = test_df.iloc[:, 1:].values
    
    # Create validation split from training set (15%) using specified SEED
    n = len(X_train)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    val_size = int(n * 0.15)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    
    print(f"Dataset Loaded Successfully!")
    print(f"Sequences -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Sequence Length: {X_train.shape[1]}")
    
    train_dataset = FordADataset(X_train, y_train)
    val_dataset = FordADataset(X_val, y_val)
    test_dataset = FordADataset(X_test, y_test)
    
    # Ensure dataloader uses the SEED for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

# --- 3. MODEL DEFINITIONS ---

class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Deep head to match FEN's readout complexity
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, return_stats=False):
        out, (h_n, c_n) = self.lstm(x)
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        if return_stats:
            return logits, {"active_norm": last_out.norm(dim=-1).mean().item()}
        return logits

class FeatureEscrow1DVariants(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes=2, mode="fen_rotational"):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_dim
        
        self.x_proj = nn.Linear(input_size, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        
        if mode == "fen_rotational":
            self.roll_gate = nn.Linear(hidden_dim, 1)
            
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        E = torch.zeros(B, self.hidden_size, device=x.device)
        
        # Pre-project sequence outside loop for speed
        x_proj_all = self.x_proj(x)
        
        gate_list = []
        gamma_list = []
        
        for t in range(seq_len):
            xt = x_proj_all[:, t, :]
            z = h + xt
            f_raw = torch.tanh(self.core(z) + z)
            
            g = torch.sigmoid(self.gate(f_raw))
            D = g * f_raw
            
            # Subtractive depletion
            h = f_raw - D
            
            if self.mode == "fen_rotational":
                gamma = torch.sigmoid(self.roll_gate(f_raw))
                E = (1 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + self.escrow_proj(D)
                gamma_list.append(gamma)
            elif self.mode == "fen_subtractive":
                E = E + self.escrow_proj(D)
                
            gate_list.append(g)
            
        combined = torch.cat([h, E], dim=-1)
        logits = self.head(combined)
        
        stats = {
            "gate": torch.stack(gate_list, dim=1),
            "active_norm": h.norm(dim=-1).mean().item()
        }
        if gamma_list:
            stats["gamma"] = torch.stack(gamma_list, dim=1)
            
        if return_stats:
            return logits, stats
        return logits

# --- 4. AUTO PARAMETER MATCHING ---

def build_model(mode, input_size, hidden_dim):
    if mode == "lstm":
        return LSTMBaseline(input_size=input_size, hidden_size=hidden_dim, num_classes=2, num_layers=2)
    else:
        return FeatureEscrow1DVariants(input_size=input_size, hidden_dim=hidden_dim, mode=mode)

def choose_hidden_dim(mode, input_size):
    if not AUTO_MATCH_PARAMS:
        return 64
        
    best_h = 8
    best_diff = float('inf')
    
    for h in range(8, 512):
        model = build_model(mode, input_size, h)
        params = count_params(model)
        
        diff = abs(params - TARGET_PARAMS)
        if diff < best_diff:
            best_h = h
            best_diff = diff
            
    return best_h

# --- 5. TRAINING & EVALUATION ---
def train_and_evaluate(mode, train_loader, val_loader, test_loader, input_size, seed):
    set_seed(seed)
    
    hidden_dim = choose_hidden_dim(mode, input_size)
    model = build_model(mode, input_size, hidden_dim).to(DEVICE)
    params_count = count_params(model)
    

    
    print(f"\n" + "-" * 80)
    print(f"SEED={seed} | Model={mode.upper()} | Hidden={hidden_dim} | Params={params_count:,}")
    print("-" * 80)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = -1.0
    best_model_path = f"best_{mode}_seed{seed}_ford.pth"
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            logits, stats = model(x, return_stats=True)
            loss = criterion(logits, y)
            
            # Polarity regularization for FEN models
            polarity_loss = 0.0
            if "gate" in stats:
                g_t = stats["gate"]
                polarity_loss = polarity_loss + torch.mean(g_t * (1.0 - g_t))
            
            if "gamma" in stats:
                gam_t = stats["gamma"]
                polarity_loss = polarity_loss + torch.mean(gam_t * (1.0 - gam_t))
                
            total_loss = loss + POLARITY_COEFF * polarity_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        active_norm_sum = 0.0
        batches_val = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs, stats = model(x, return_stats=True)
                
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())
                active_norm_sum += stats["active_norm"]
                batches_val += 1
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_active_norm = active_norm_sum / batches_val if batches_val > 0 else 0.0
        
        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds, average='macro') * 100
        
        if epoch % 5 == 0 or epoch == NUM_EPOCHS or val_f1 > best_val_f1:
            print(f"  Ep {epoch:02d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | Active Norm: {avg_active_norm:.2f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            
    training_time = time.time() - start_time
    print(f"  Finished {mode.upper()} | Best Val F1: {best_val_f1:.2f}% | Time: {training_time:.1f}s")
    
    # Test Evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(y.cpu().numpy())
            
    test_acc = accuracy_score(test_targets, test_preds) * 100
    test_f1 = f1_score(test_targets, test_preds, average='macro') * 100
    
    print(f"  Test Eval: Accuracy: {test_acc:.2f}% | Macro F1: {test_f1:.2f}%")
    
    # Cleanup saved model file to save space
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        
    return {
        "mode": mode,
        "best_val_f1": best_val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "time": training_time
    }

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        MODES = ["fen_subtractive", "fen_rotational", "lstm"]
        
        # Slice the seeds to resume from the user-specified start index
        active_seeds = SEEDS[START_SEED_INDEX:]
        
        results = {mode: [] for mode in MODES}
        
        print("=" * 90)
        print(f"STARTING 10-SEED ALTERNATING BENCHMARK | TARGET PARAMS = {TARGET_PARAMS}")
        print("=" * 90)
        
        for s_idx, seed in enumerate(active_seeds):
            abs_batch_idx = s_idx + START_SEED_INDEX + 1
            print(f"\n================================================================================")
            print(f"SEED BATCH {abs_batch_idx}/{len(SEEDS)}: SEED = {seed}")
            print(f"================================================================================")
            
            train_loader, val_loader, test_loader = prepare_data(seed)
            
            for mode in MODES:
                res = train_and_evaluate(mode, train_loader, val_loader, test_loader, input_size=1, seed=seed)
                results[mode].append(res)
                
            # Print running summary after every seed completes to preserve data if interrupted
            print(f"\n--- RUNNING RESULTS SUMMARY (Completed Seeds: {abs_batch_idx}/{len(SEEDS)}) ---")
            print(f"{'Model':<16} | {'Test Acc (Mean)':<16} | {'Test F1 (Mean)':<16} | {'Completed Runs'}")
            print("-" * 70)
            for mode in MODES:
                accs = [r["test_acc"] for r in results[mode]]
                f1s = [r["test_f1"] for r in results[mode]]
                print(f"{mode.upper():<16} | {np.mean(accs):.2f}% (±{np.std(accs):.2f}) | {np.mean(f1s):.2f}% (±{np.std(f1s):.2f}) | {len(accs)}/10")
            print("-" * 70)
            
        print("\n\n" + "#" * 90)
        print("FINAL 10-SEED BENCHMARK COMPLETE SUMMARY")
        print("#" * 90)
        for mode in MODES:
            accs = [r["test_acc"] for r in results[mode]]
            f1s = [r["test_f1"] for r in results[mode]]
            print(f"MODE: {mode.upper():<16} | Test Acc: {np.mean(accs):.2f}% (±{np.std(accs):.2f}) | Test F1: {np.mean(f1s):.2f}% (±{np.std(f1s):.2f})")
        print("#" * 90)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
