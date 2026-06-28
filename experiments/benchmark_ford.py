# ==============================================================================
# FEN vs Baselines Benchmarking on FordA Time-Series Classification (UCR Benchmark)
# Publicly accessible dataset, univariate sequence classification (500 steps)
# Full evaluation of 6 configurations: Vanilla, Residual, Copy-Only, and Subtractive
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
NUM_EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Initialize seed immediately
set_seed(SEED)

# --- 2. DATASET DOWNLOADING & PREPROCESSING ---
def download_dataset():
    os.makedirs('data', exist_ok=True)
    train_path = 'data/FordA_TRAIN.tsv'
    test_path = 'data/FordA_TEST.tsv'
    
    # Direct GitHub raw download links (no signup, no API keys needed)
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

def prepare_data():
    train_path, test_path = download_dataset()
    
    # Load TSV files
    train_df = pd.read_csv(train_path, sep='\t', header=None)
    test_df = pd.read_csv(test_path, sep='\t', header=None)
    
    # First column is the class label: -1 or 1
    # Map class labels to 0 and 1
    y_train = train_df[0].values
    y_train = np.where(y_train == -1, 0, 1)
    X_train = train_df.iloc[:, 1:].values
    
    y_test = test_df[0].values
    y_test = np.where(y_test == -1, 0, 1)
    X_test = test_df.iloc[:, 1:].values
    
    # Create validation split from training set (15%) using specified SEED
    n = len(X_train)
    indices = np.arange(n)
    np.random.seed(SEED)
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
    g.manual_seed(SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

# --- 3. MODEL DEFINITIONS ---

# 3.1 Vanilla RNN Baseline (Standard PyTorch stacked RNN)
class RNNBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, return_stats=False):
        out, h_n = self.rnn(x)
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        if return_stats:
            return logits, {"active_norm": last_out.norm(dim=-1).mean().item()}
        return logits

# 3.2 Residual RNN Baseline (2-Layer Stacked with Temporal Residuals)
class ResidualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.RNNCell(input_size if l == 0 else hidden_size, hidden_size) 
            for l in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            xt = x[:, t, :]
            h_next = []
            
            # Layer 0
            h0_n = self.cells[0](xt, h[0])
            h0 = h0_n + h[0]  # Temporal residual
            h_next.append(h0)
            
            # Deeper layers
            for l in range(1, self.num_layers):
                hl_n = self.cells[l](h_next[l-1], h[l])
                hl = hl_n + h[l]  # Temporal residual
                h_next.append(hl)
                
            h = h_next
            
        last_out = h[-1] # Sequence-to-one readout
        logits = self.fc(last_out)
        if return_stats:
            return logits, {"active_norm": last_out.norm(dim=-1).mean().item()}
        return logits

# 3.3 Vanilla LSTM Baseline (Standard PyTorch stacked LSTM)
class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, return_stats=False):
        out, (h_n, c_n) = self.lstm(x)
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        if return_stats:
            return logits, {"active_norm": last_out.norm(dim=-1).mean().item()}
        return logits

# 3.4 Residual LSTM Baseline (2-Layer Stacked with Temporal Residuals)
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.LSTMCell(input_size if l == 0 else hidden_size, hidden_size) 
            for l in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            xt = x[:, t, :]
            h_next, c_next = [], []
            
            # Layer 0
            h0_n, c0_n = self.cells[0](xt, (h[0], c[0]))
            h0 = h0_n + h[0]  # Temporal residual
            h_next.append(h0)
            c_next.append(c0_n)
            
            # Deeper layers
            for l in range(1, self.num_layers):
                hl_n, cl_n = self.cells[l](h_next[l-1], (h[l], c[l]))
                hl = hl_n + h[l]  # Temporal residual
                h_next.append(hl)
                c_next.append(cl_n)
                
            h = h_next
            c = c_next
            
        last_out = h[-1] # Sequence-to-one readout
        logits = self.fc(last_out)
        if return_stats:
            return logits, {"active_norm": last_out.norm(dim=-1).mean().item()}
        return logits

# 3.5 Feature-Escrow Network (FEN-RNN: supports both subtractive and copy-only)
class FeatureEscrowRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, subtractive=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.subtractive = subtractive
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.core = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.escrow_proj = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        E = torch.zeros(B, self.hidden_size, device=x.device)
        
        # Speed Optimization: Pre-project the entire sequence outside the loop
        x_proj_all = self.x_proj(x) # Shape: [B, seq_len, hidden_size]
        
        gate_means = []
        for t in range(seq_len):
            xt = x_proj_all[:, t, :] # Slice pre-projected input
            z = h + xt
            f_raw = torch.tanh(self.core(z) + z)
            
            g = torch.sigmoid(self.gate(f_raw))
            D = g * f_raw
            
            # Active Depletion Routing
            if self.subtractive:
                h = f_raw - D # Subtractive routing (Active State Depletion)
            else:
                h = f_raw     # Copy-only (No subtraction)
                
            E = E + self.escrow_proj(D)
            
            if return_stats:
                gate_means.append(g.mean().item())
                
        combined = torch.cat([h, E], dim=-1)
        logits = self.fc(combined)
        
        if return_stats:
            stats = {
                "active_norm": h.norm(dim=-1).mean().item(),
                "gate_mean": sum(gate_means)/len(gate_means) if gate_means else 0.5
            }
            return logits, stats
        return logits

# --- 4. AUTO PARAMETER MATCHING ---
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(mode, input_size, hidden_dim):
    if mode == "rnn_vanilla":
        return RNNBaseline(input_size=input_size, hidden_size=hidden_dim, num_classes=2, num_layers=2)
    elif mode == "rnn_residual":
        return ResidualRNN(input_size=input_size, hidden_size=hidden_dim, num_classes=2, num_layers=2)
    elif mode == "lstm_vanilla":
        return LSTMBaseline(input_size=input_size, hidden_size=hidden_dim, num_classes=2, num_layers=2)
    elif mode == "lstm_residual":
        return ResidualLSTM(input_size=input_size, hidden_size=hidden_dim, num_classes=2, num_layers=2)
    elif mode == "fen_copy":
        return FeatureEscrowRNN(input_size=input_size, hidden_size=hidden_dim, subtractive=False)
    elif mode == "fen_subtractive":
        return FeatureEscrowRNN(input_size=input_size, hidden_size=hidden_dim, subtractive=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def choose_hidden_dim(mode, input_size):
    if not AUTO_MATCH_PARAMS:
        return 64
        
    best_h = 8
    best_diff = float('inf')
    
    for h in range(8, 512):
        model = build_model(mode, input_size, h)
        params = count_params(model)
        
        # If it is a FEN variant, ensure its parameter count is strictly under the 
        # minimum baseline parameter count (which is 99,182 for ResidualLSTM)
        if mode.startswith("fen"):
            if params <= 99182:
                diff = 99182 - params
                if diff < best_diff:
                    best_h = h
                    best_diff = diff
        else:
            diff = abs(params - TARGET_PARAMS)
            if diff < best_diff:
                best_h = h
                best_diff = diff
            
    return best_h

# --- 5. TRAINING & EVALUATION ---
def train_and_evaluate(mode, train_loader, val_loader, test_loader, input_size):
    # Seed training variables for absolute robustness check
    set_seed(SEED)
    
    hidden_dim = choose_hidden_dim(mode, input_size)
    model = build_model(mode, input_size, hidden_dim).to(DEVICE)
    params_count = count_params(model)
    
    print(f"\n================================================================================")
    print(f"START TRAINING: Model={mode.upper()} | Hidden={hidden_dim} | Params={params_count:,}")
    print(f"================================================================================")
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = -1.0
    best_model_path = f"best_{mode}_ford.pth"
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
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
                
                # Check for return stats support
                if mode in ["rnn_residual", "lstm_residual", "fen_copy", "fen_subtractive"]:
                    outputs, stats = model(x, return_stats=True)
                    active_norm_sum += stats["active_norm"]
                else:
                    # For vanilla models, we can also extract active norm from standard forward
                    outputs, stats = model(x, return_stats=True)
                    active_norm_sum += stats["active_norm"]
                    
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())
                batches_val += 1
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_active_norm = active_norm_sum / batches_val if batches_val > 0 else 0.0
        
        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds, average='macro') * 100
        
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | Active Norm: {avg_active_norm:.2f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            
    training_time = time.time() - start_time
    print(f"Finished {mode} | Best Val Macro F1: {best_val_f1:.2f}% | Time: {training_time:.1f}s")
    
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
    
    print(f"Test Evaluation for {mode.upper()}:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Macro F1-Score: {test_f1:.2f}%")
    
    return {
        "mode": mode,
        "params": params_count,
        "best_val_f1": best_val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "active_norm": avg_active_norm,
        "time": training_time
    }

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = prepare_data()
        
        # Sweep all 6 models
        modes = [
            "rnn_vanilla", 
            "rnn_residual", 
            "lstm_vanilla", 
            "lstm_residual", 
            "fen_copy", 
            "fen_subtractive"
        ]
        
        results = {}
        for mode in modes:
            results[mode] = train_and_evaluate(mode, train_loader, val_loader, test_loader, input_size=1)
            
        # Print Final Summary Table
        print("\n" + "#" * 90)
        print("FINAL BENCHMARK SUMMARY (FORD A TIME-SERIES CLASSIFICATION)")
        print("#" * 90)
        print(f"{'Model':<16} | {'Params':<8} | {'Best Val F1':<12} | {'Test F1':<12} | {'Test Acc':<10} | {'Active Norm':<11} | {'Time':<6}")
        print("-" * 98)
        for mode in modes:
            res = results[mode]
            val_f1_str = f"{res['best_val_f1']:.2f}%"
            test_f1_str = f"{res['test_f1']:.2f}%"
            test_acc_str = f"{res['test_acc']:.2f}%"
            active_norm_str = f"{res['active_norm']:.2f}"
            time_str = f"{res['time']:.1f}s"
            
            print(f"{mode.upper():<16} | {res['params']:<8,} | {val_f1_str:<12} | {test_f1_str:<12} | {test_acc_str:<10} | {active_norm_str:<11} | {time_str:<6}")
        print("#" * 90)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
