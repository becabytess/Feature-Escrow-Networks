# ============================================================
# 1D Feature-Escrow Network (FEN)
# Real-World Verification: UCI Human Activity Recognition
# ============================================================

import os, time, urllib.request, zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 1. EDIT CONFIG
# ============================================================

RUN_MODE = "all_quick"
# Options:
#   "lstm"
#   "fen_copy_only"
#   "fen_full"

SEEDS = [1]
EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

TARGET_PARAMS = 12500  # Strict matching to the original 12.5k claim
AUTO_MATCH_PARAMS = True

# ============================================================
# 2. SETUP & DATA DOWNLOADER
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def download_and_load_uci_har():
    data_dir = "./UCI_HAR_Dataset"
    zip_path = "./UCI_HAR.zip"
    
    if not os.path.exists(data_dir):
        print("Downloading UCI HAR Dataset...")
        url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            
        # The inner zip might need extraction
        inner_zip = "./UCI HAR Dataset.zip"
        if os.path.exists(inner_zip):
            with zipfile.ZipFile(inner_zip, 'r') as zip_ref:
                zip_ref.extractall(".")
        
        os.remove(zip_path)
        if os.path.exists(inner_zip): os.remove(inner_zip)

    # Standard UCI HAR paths
    train_x_path = "./UCI HAR Dataset/train/Inertial Signals/"
    test_x_path = "./UCI HAR Dataset/test/Inertial Signals/"
    
    signals = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    
    def load_signals(subset):
        path = f"./UCI HAR Dataset/{subset}/Inertial Signals/"
        loaded = []
        for sig in signals:
            filename = f"{path}{sig}_{subset}.txt"
            # loadtxt is safe here, shapes are fixed 128 columns
            loaded.append(np.loadtxt(filename, dtype=np.float32))
        # Stack into [samples, 128, 9]
        return np.dstack(loaded)
    
    def load_labels(subset):
        filename = f"./UCI HAR Dataset/{subset}/y_{subset}.txt"
        # UCI HAR labels are 1-6, shift to 0-5
        return np.loadtxt(filename, dtype=np.int64) - 1

    print("Loading text files into memory (takes a few seconds)...")
    X_train = load_signals("train")
    y_train = load_labels("train")
    X_test = load_signals("test")
    y_test = load_labels("test")
    
    print(f"Data loaded! Train: {X_train.shape}, Test: {X_test.shape}")
    return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test)

# ============================================================
# 3. MODELS
# ============================================================

class LSTMBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_stats=False):
        out, (h_n, c_n) = self.lstm(x)
        h = h_n[-1] # Take the final hidden state
        logits = self.head(h)
        if return_stats:
            return logits, {"pipe_norm": h.norm(dim=-1).mean().item()}
        return logits

class FeatureEscrow1D(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mode):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        self.x_proj = nn.Linear(in_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Classifier reads final Active Stream + final Escrow
        self.head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        E = torch.zeros(B, self.hidden_dim, device=x.device)
        
        gate_means = []
        
        no_sub = (self.mode == "fen_copy_only")
        
        for t in range(seq_len):
            xt = self.x_proj(x[:, t, :])
            z = h + xt
            
            # Active Transformation
            f_raw = torch.tanh(self.core(z) + z)
            
            # Escrow Gate
            g = torch.sigmoid(self.gate(f_raw))
            D = g * f_raw
            
            # Subtractive Routing
            if no_sub:
                h = f_raw
            else:
                h = f_raw - D
                
            # Secure Archiving
            E = E + self.escrow_proj(D)
            
            if return_stats:
                gate_means.append(g.mean().detach())
                
        combined = torch.cat([h, E], dim=-1)
        logits = self.head(combined)
        
        if return_stats:
            return logits, {
                "gate_mean": sum(gate_means)/len(gate_means),
                "pipe_norm": h.norm(dim=-1).mean().item(),
                "escrow_norm": E.norm(dim=-1).mean().item()
            }
        return logits

def build_model(mode, hidden_dim):
    if mode == "lstm":
        return LSTMBaseline(9, hidden_dim, 6)
    else:
        return FeatureEscrow1D(9, hidden_dim, 6, mode)

def choose_hidden(mode):
    if not AUTO_MATCH_PARAMS: return 32
    best_h, best_diff = 16, float('inf')
    for h in range(16, 128):
        model = build_model(mode, h)
        diff = abs(count_params(model) - TARGET_PARAMS)
        if diff < best_diff:
            best_h, best_diff = h, diff
    return best_h

# ============================================================
# 4. TRAINING
# ============================================================

@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    correct, total = 0, 0
    gate_sum, pipe_sum, stat_n = 0.0, 0.0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, stats = model(xb, return_stats=True)
        
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += xb.shape[0]
        
        if stats:
            gate_sum += stats.get("gate_mean", 0)
            pipe_sum += stats.get("pipe_norm", 0)
            stat_n += 1
            
    out = {"acc": correct / total}
    if stat_n:
        out["gate"] = gate_sum / stat_n
        out["pipe"] = pipe_sum / stat_n
    return out

def train_one(seed, mode, X_train, y_train, X_test, y_test):
    seed_everything(seed)
    
    train_ds = TensorDataset(X_train, y_train)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    
    hidden = choose_hidden(mode)
    model = build_model(mode, hidden).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"MODE={mode} | SEED={seed} | hidden={hidden} | params={params:,}")
    print("=" * 80)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_ep = 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
        val = evaluate(model, X_test, y_test)
        
        if val["acc"] > best_acc: 
            best_acc = val["acc"]
            best_ep = epoch
            
        gate_str = f" gate={val['gate']:.3f} pipe={val['pipe']:.2f}" if "gate" in val else f" pipe={val['pipe']:.2f}"
        print(f"ep {epoch:02d} | val_acc={val['acc']:.4f} | best={best_acc:.4f}@{best_ep} {gate_str}")

    print(f"FINAL {mode}: Best Acc {best_acc:.4f} | Time: {time.time()-start:.1f}s")
    return {"mode": mode, "params": params, "best_acc": best_acc, "best_epoch": best_ep}

# ============================================================
# 5. EXECUTION
# ============================================================

X_train, y_train, X_test, y_test = download_and_load_uci_har()

if RUN_MODE == "all_quick":
    modes = ["lstm", "fen_copy_only", "fen_full"]
else:
    modes = [RUN_MODE]

results = []
for mode in modes:
    results.append(train_one(SEEDS[0], mode, X_train, y_train, X_test, y_test))

print("\n\n" + "#" * 80)
print("SUMMARY")
print("#" * 80)
for r in results:
    print(f"MODE: {r['mode']:<15} | Params: {r['params']} | Best Acc: {r['best_acc']:.4f} @ Ep {r['best_epoch']}")
