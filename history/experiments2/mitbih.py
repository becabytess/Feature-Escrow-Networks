# ==============================================================================
# ECG Heartbeat Arrhythmia Classification (MIT-BIH) Benchmark
# Task: 187-step 1D sequence classification (Topology / Order-Sensitive Waveform Test)
# Sweep: 5-Seed Alternating Benchmark of Subtractive FEN vs. Rotational FEN vs. LSTM
# Speedups: Direct GPU preloading, torch.compile(mode="reduce-overhead"), cached choose_hidden
# ==============================================================================

import sys
# Boost recursion limit so PyTorch Inductor can trace the 187-step loop without crashing
sys.setrecursionlimit(50000)

import os
import time
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# --- 1. CONFIGURATION ---
SEEDS = [100, 200, 300, 400, 500]
START_SEED_INDEX = 0
EPOCHS = 15
BATCH_SIZE = 1000
LR = 2e-3  
WEIGHT_DECAY = 1e-4

# Equalize all models to a 75k parameter budget
TARGET_PARAMS = 75000
AUTO_MATCH_PARAMS = True
POLARITY_COEFF = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 2. FAST ZERO-HANG DOWNLOADER & GPU LOADER ---
def download_mitbih():
    train_url = "https://github.com/csuustc/ECG-Heartbeat-Classification/raw/master/mitbih_train.csv.zip"
    test_url = "https://github.com/csuustc/ECG-Heartbeat-Classification/raw/master/mitbih_test.csv.zip"
    
    os.makedirs("./data", exist_ok=True)
    
    if not os.path.exists("./data/mitbih_train.csv"):
        print("Downloading mitbih_train.csv.zip (15MB)...")
        urllib.request.urlretrieve(train_url, "./data/mitbih_train.csv.zip")
        print("Extracting train set...")
        with zipfile.ZipFile("./data/mitbih_train.csv.zip", 'r') as zip_ref:
            zip_ref.extractall("./data")
        os.remove("./data/mitbih_train.csv.zip")
        
    if not os.path.exists("./data/mitbih_test.csv"):
        print("Downloading mitbih_test.csv.zip (4MB)...")
        urllib.request.urlretrieve(test_url, "./data/mitbih_test.csv.zip")
        print("Extracting test set...")
        with zipfile.ZipFile("./data/mitbih_test.csv.zip", 'r') as zip_ref:
            zip_ref.extractall("./data")
        os.remove("./data/mitbih_test.csv.zip")
        print("Download complete.")

def load_mitbih_gpu():
    download_mitbih()
    print("Parsing CSV files into memory (takes a few seconds)...")
    train_df = pd.read_csv("./data/mitbih_train.csv", header=None)
    test_df = pd.read_csv("./data/mitbih_test.csv", header=None)
    
    # Split features and labels (last column is the label, 0 to 4)
    X_train_raw = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train_raw = train_df.iloc[:, -1].values.astype(np.int64)
    
    X_test_raw = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test_raw = test_df.iloc[:, -1].values.astype(np.int64)
    
    # Reshape to [N, 187, 1] for 1D sequence models and push to CUDA
    X_train = torch.tensor(X_train_raw).unsqueeze(-1).to(device)
    y_train = torch.tensor(y_train_raw).to(device)
    X_test = torch.tensor(X_test_raw).unsqueeze(-1).to(device)
    y_test = torch.tensor(y_test_raw).to(device)
    
    print(f"Dataset fully loaded to GPU. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def get_batches(X, y, batch_size, shuffle=True):
    N = X.size(0)
    if shuffle:
        indices = torch.randperm(N, device=X.device)
        X = X[indices]
        y = y[indices]
    
    for i in range(0, N, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# --- 3. MODEL DEFINITIONS ---

class LSTMBaseline(nn.Module):
    def __init__(self, in_dim=1, hidden_size=64, out_dim=5, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits, {"pipe_norm": last_out.norm(dim=-1).mean()}

@torch.jit.script
def subtractive_loop(X_projected: torch.Tensor, h0: torch.Tensor, E0: torch.Tensor,
                     core_w: torch.Tensor, core_b: torch.Tensor,
                     gate_w: torch.Tensor, gate_b: torch.Tensor,
                     escrow_w: torch.Tensor, escrow_b: torch.Tensor):
    h = h0
    E = E0
    seq_len = X_projected.size(1)
    gate_mean_acc = torch.tensor(0.0, device=X_projected.device)
    pol_acc = torch.tensor(0.0, device=X_projected.device)
    
    for t in range(seq_len):
        xt = X_projected[:, t, :]
        z = h + xt
        f_raw = torch.tanh(torch.nn.functional.linear(z, core_w, core_b) + z)
        
        g = torch.sigmoid(torch.nn.functional.linear(f_raw, gate_w, gate_b))
        D = g * f_raw
        h = f_raw - D
        E = E + torch.nn.functional.linear(D, escrow_w, escrow_b)
        
        gate_mean_acc = gate_mean_acc + g.mean()
        pol_acc = pol_acc + torch.mean(g * (1.0 - g))
        
    return h, E, gate_mean_acc, pol_acc

@torch.jit.script
def rotational_loop(X_projected: torch.Tensor, h0: torch.Tensor, E0: torch.Tensor,
                    core_w: torch.Tensor, core_b: torch.Tensor,
                    gate_w: torch.Tensor, gate_b: torch.Tensor,
                    escrow_w: torch.Tensor, escrow_b: torch.Tensor,
                    roll_gate_w: torch.Tensor, roll_gate_b: torch.Tensor):
    h = h0
    E = E0
    seq_len = X_projected.size(1)
    gate_mean_acc = torch.tensor(0.0, device=X_projected.device)
    pol_acc = torch.tensor(0.0, device=X_projected.device)
    
    for t in range(seq_len):
        xt = X_projected[:, t, :]
        z = h + xt
        f_raw = torch.tanh(torch.nn.functional.linear(z, core_w, core_b) + z)
        
        g = torch.sigmoid(torch.nn.functional.linear(f_raw, gate_w, gate_b))
        D = g * f_raw
        h = f_raw - D
        gamma = torch.sigmoid(torch.nn.functional.linear(f_raw, roll_gate_w, roll_gate_b))
        E = (1 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + torch.nn.functional.linear(D, escrow_w, escrow_b)
        
        gate_mean_acc = gate_mean_acc + g.mean()
        pol_acc = pol_acc + torch.mean(g * (1.0 - g)) + torch.mean(gamma * (1.0 - gamma))
        
    return h, E, gate_mean_acc, pol_acc

class FeatureEscrow1DVariants(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mode):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        self.x_proj = nn.Linear(in_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        
        if mode == "fen_rotational":
            self.roll_gate = nn.Linear(hidden_dim, 1)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        E = torch.zeros(B, self.hidden_dim, device=x.device)
        
        # Precompute projection outside the loop
        X_projected = self.x_proj(x)
        
        core_w, core_b = self.core.weight, self.core.bias
        gate_w, gate_b = self.gate.weight, self.gate.bias
        escrow_w, escrow_b = self.escrow_proj.weight, self.escrow_proj.bias
        
        if self.mode == "fen_rotational":
            roll_gate_w, roll_gate_b = self.roll_gate.weight, self.roll_gate.bias
            h, E, gate_mean_acc, pol_acc = rotational_loop(
                X_projected, h, E, core_w, core_b, gate_w, gate_b, escrow_w, escrow_b, roll_gate_w, roll_gate_b
            )
        elif self.mode == "fen_subtractive":
            h, E, gate_mean_acc, pol_acc = subtractive_loop(
                X_projected, h, E, core_w, core_b, gate_w, gate_b, escrow_w, escrow_b
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        combined = torch.cat([h, E], dim=-1)
        logits = self.head(combined)
        
        stats = {
            "gate": gate_mean_acc / seq_len,
            "pipe_norm": h.norm(dim=-1).mean(),
            "escrow_norm": E.norm(dim=-1).mean(),
            "polarity_loss": pol_acc / seq_len
        }
        return logits, stats

# --- 4. PARAMETER MATCHING WITH GLOBAL CACHE ---

HIDDEN_SIZES_CACHE = {}

def build_model(mode, hidden_dim):
    if mode == "lstm":
        return LSTMBaseline(in_dim=1, hidden_size=hidden_dim, out_dim=5, num_layers=2)
    else:
        return FeatureEscrow1DVariants(in_dim=1, hidden_dim=hidden_dim, out_dim=5, mode=mode)

def choose_hidden(mode):
    if mode in HIDDEN_SIZES_CACHE:
        return HIDDEN_SIZES_CACHE[mode]
        
    if not AUTO_MATCH_PARAMS: 
        return 64
        
    best_h, best_diff = 16, float('inf')
    for h in range(16, 256):
        model = build_model(mode, h)
        diff = abs(count_params(model) - TARGET_PARAMS)
        if diff < best_diff:
            best_h, best_diff = h, diff
            
    HIDDEN_SIZES_CACHE[mode] = best_h
    return best_h

def save_history_to_csv(history, filepath="mitbih_history.csv"):
    import csv
    file_exists = os.path.exists(filepath)
    keys = history[0].keys()
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(history)

# --- 5. TRAINING & EVALUATION ---

@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    preds, targets = [], []
    gate_sum, pipe_sum = 0.0, 0.0
    gate_n, pipe_n = 0, 0
    
    for xb, yb in get_batches(X, y, BATCH_SIZE, shuffle=False):
        logits, stats = model(xb)
        
        pred = logits.argmax(dim=-1).cpu().numpy()
        preds.extend(pred)
        targets.extend(yb.cpu().numpy())
        
        if stats:
            if "gate" in stats:
                gate_sum += stats["gate"].item()
                gate_n += 1
            if "pipe_norm" in stats:
                pipe_sum += stats["pipe_norm"].item()
                pipe_n += 1
            
    acc = accuracy_score(targets, preds)
    out = {"acc": acc * 100}
    if gate_n:
        out["gate"] = gate_sum / gate_n
    if pipe_n:
        out["pipe"] = pipe_sum / pipe_n
    return out

def train_one(seed, mode, X_train, y_train, X_test, y_test):
    seed_everything(seed)
    
    hidden = choose_hidden(mode)
    model = build_model(mode, hidden).to(device)
    params = count_params(model)
    
    print(f"\n" + "-" * 80)
    print(f"SEED={seed} | Model={mode.upper()} | Hidden={hidden} | Params={params:,}")
    print("-" * 80)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_ep = 0
    best_model_path = f"best_{mode}_seed{seed}_mitbih.pth"
    
    start = time.time()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in get_batches(X_train, y_train, BATCH_SIZE, shuffle=True):
            opt.zero_grad(set_to_none=True)
            logits, stats = model(xb)
            loss = criterion(logits, yb)
            
            polarity_loss = stats.get("polarity_loss", torch.tensor(0.0, device=device))
            total_loss = loss + POLARITY_COEFF * polarity_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
        val = evaluate(model, X_test, y_test)
        
        if val["acc"] > best_acc: 
            best_acc = val["acc"]
            best_ep = epoch
            torch.save(model.state_dict(), best_model_path)
            
        gate_str = f" gate={val['gate']:.3f} pipe={val['pipe']:.2f}" if "gate" in val else f" pipe={val['pipe']:.2f}"
        print(f"  Ep {epoch:02d} | Val Acc: {val['acc']:.2f}% | Best Acc: {best_acc:.2f}%@{best_ep}{gate_str}")

        history.append({
            "seed": seed,
            "mode": mode,
            "epoch": epoch,
            "val_acc": val["acc"],
            "gate": val.get("gate", 0.0),
            "pipe": val.get("pipe", 0.0)
        })

    training_time = time.time() - start
    print(f"  Finished {mode.upper()} | Best Val Acc: {best_acc:.2f}% | Time: {training_time:.1f}s")
    save_history_to_csv(history)
    
    model.load_state_dict(torch.load(best_model_path))
    final_metrics = evaluate(model, X_test, y_test)
    
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        
    return {
        "mode": mode,
        "params": params,
        "best_acc": final_metrics["acc"],
        "time": training_time
    }

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mitbih_gpu()
    
    MODES = ["fen_subtractive", "fen_rotational", "lstm"]
    active_seeds = SEEDS[START_SEED_INDEX:]
    
    results = {mode: [] for mode in MODES}
    
    print("=" * 90)
    print(f"STARTING 5-SEED ALTERNATING BENCHMARK (MIT-BIH) | TARGET PARAMS = {TARGET_PARAMS}")
    print("=" * 90)
    
    for s_idx, seed in enumerate(active_seeds):
        abs_batch_idx = s_idx + START_SEED_INDEX + 1
        print(f"\n================================================================================")
        print(f"SEED BATCH {abs_batch_idx}/{len(SEEDS)}: SEED = {seed}")
        print(f"================================================================================")
        
        for mode in MODES:
            res = train_one(seed, mode, X_train, y_train, X_test, y_test)
            results[mode].append(res)
            
        print(f"\n--- RUNNING RESULTS SUMMARY (Completed Seeds: {abs_batch_idx}/{len(SEEDS)}) ---")
        print(f"{'Model':<16} | {'Val Acc (Mean)':<16} | {'Completed Runs'}")
        print("-" * 55)
        for mode in MODES:
            accs = [r["best_acc"] for r in results[mode]]
            print(f"{mode.upper():<16} | {np.mean(accs):.2f}% (±{np.std(accs):.2f}) | {len(accs)}/5")
        print("-" * 55)
        
    print("\n\n" + "#" * 90)
    print("FINAL 5-SEED BENCHMARK COMPLETE SUMMARY (MIT-BIH)")
    print("#" * 90)
    for mode in MODES:
        accs = [r["best_acc"] for r in results[mode]]
        print(f"MODE: {mode.upper():<16} | Val Acc: {np.mean(accs):.2f}% (±{np.std(accs):.2f})")
    print("#" * 90)
