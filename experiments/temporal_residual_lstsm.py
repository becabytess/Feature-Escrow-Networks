# ============================================================
# 1D Feature-Escrow Network (FEN)
# Verification on Synthetic "Distracted Counting" Task (Patched)
# ============================================================

import os, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 1. EDIT CONFIG
# ============================================================

RUN_MODE = "all_quick"
# Options:
#   "all_quick"       -> Runs all 4 models sequentially
#   "lstm"
#   "lstm_residual"   -> The true temporal residual LSTM
#   "fen_rnn"         -> Original FEN (Raw RNN active stream)
#   "fen_lstm"        -> FEN wrapping an LSTM active stream

SEEDS = [1]
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0

TRAIN_N = 8000
TEST_N = 2000

TARGET_PARAMS = 15000  # Strict matching
AUTO_MATCH_PARAMS = True

# ============================================================
# 2. SETUP & DATASET GENERATOR
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_distracted_dataset(n, seed):
    """
    1D Sequence Reasoning Task:
      - Step 0 contains a static ID (1 of 10) that must be preserved.
      - Remaining 95 steps contain random active counting pulses (+/-) and noise.
      - Target = ID * 3 + Count_Bin (30 total classes)
    """
    rng = np.random.default_rng(seed)

    n_id = 10
    n_bins = 3
    op_dims = 4
    noise_dims = 16
    input_dim = n_id + op_dims + noise_dims

    X = rng.normal(0.0, 0.45, size=(n, 96, input_dim)).astype(np.float32)

    # Clean up symbolic channels relative to noise
    X[:, :, :n_id + op_dims] *= 0.10

    y = np.zeros((n,), dtype=np.int64)

    plus_dim = n_id
    minus_dim = n_id + 1
    distract_a = n_id + 2
    distract_b = n_id + 3

    for i in range(n):
        static_id = rng.integers(0, n_id)
        count_bin = rng.integers(0, n_bins)

        # Place ID strictly at the beginning (t=0)
        X[i, 0, static_id] += 2.0

        possible = np.array([t for t in range(96) if t != 0])
        n_events = rng.integers(18, 31)
        positions = rng.choice(possible, size=n_events, replace=False)

        if count_bin == 0:
            p_plus = 0.25
        elif count_bin == 1:
            p_plus = 0.50
        else:
            p_plus = 0.75

        n_plus = int(round(n_events * p_plus))
        plus_positions = positions[:n_plus]
        minus_positions = positions[n_plus:]

        X[i, plus_positions, plus_dim] += 1.5
        X[i, minus_positions, minus_dim] += 1.5

        # Distractors
        n_distract = rng.integers(10, 25)
        dpos = rng.choice(possible, size=n_distract, replace=False)
        half = n_distract // 2
        X[i, dpos[:half], distract_a] += 1.25
        X[i, dpos[half:], distract_b] += 1.25

        y[i] = static_id * n_bins + count_bin

    meta = {
        "input_dim": input_dim,
        "output_dim": n_id * n_bins,
        "task_type": "single",
        "n_id": n_id,
        "n_bins": n_bins,
    }

    return torch.tensor(X), torch.tensor(y), meta

# ============================================================
# 3. MODELS
# ============================================================

class LSTMBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_stats=False):
        out, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        logits = self.head(h)
        if return_stats: return logits, {"pipe_norm": h.norm(dim=-1).mean().item()}
        return logits

class ResidualLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        for t in range(seq_len):
            h_next, c = self.lstm_cell(x[:, t, :], (h, c))
            h = h_next + h  # The temporal residual
        logits = self.head(h)
        if return_stats: return logits, {"pipe_norm": h.norm(dim=-1).mean().item()}
        return logits

class FeatureEscrowRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_proj = nn.Linear(in_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        E = torch.zeros(B, self.hidden_dim, device=x.device)
        gate_means = []
        for t in range(seq_len):
            xt = self.x_proj(x[:, t, :])
            z = h + xt
            f_raw = torch.tanh(self.core(z) + z)  # RNN active stream
            
            g = torch.sigmoid(self.gate(f_raw))
            D = g * f_raw
            h = f_raw - D  # Depletion
            E = E + self.escrow_proj(D)
            if return_stats: gate_means.append(g.mean().detach())
                
        logits = self.head(torch.cat([h, E], dim=-1))
        if return_stats:
            return logits, {"gate_mean": sum(gate_means)/len(gate_means), "pipe_norm": h.norm(dim=-1).mean().item()}
        return logits

class FeatureEscrowLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(in_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        E = torch.zeros(B, self.hidden_dim, device=x.device)
        gate_means = []
        for t in range(seq_len):
            h_next, c = self.lstm_cell(x[:, t, :], (h, c)) # LSTM active stream
            
            g = torch.sigmoid(self.gate(h_next))
            D = g * h_next
            h = h_next - D
            E = E + self.escrow_proj(D)
            if return_stats: gate_means.append(g.mean().detach())
                
        logits = self.head(torch.cat([h, E], dim=-1))
        if return_stats:
            return logits, {"gate_mean": sum(gate_means)/len(gate_means), "pipe_norm": h.norm(dim=-1).mean().item()}
        return logits

def build_model(mode, hidden_dim):
    if mode == "lstm": return LSTMBaseline(30, hidden_dim, 30)
    if mode == "lstm_residual": return ResidualLSTM(30, hidden_dim, 30)
    if mode == "fen_rnn": return FeatureEscrowRNN(30, hidden_dim, 30)
    if mode == "fen_lstm": return FeatureEscrowLSTM(30, hidden_dim, 30)
    raise ValueError(mode)

def choose_hidden(mode):
    if not AUTO_MATCH_PARAMS: return 32
    best_h, best_diff = 4, float('inf')
    for h in range(4, 128):
        model = build_model(mode, h)
        diff = abs(count_params(model) - TARGET_PARAMS)
        if diff < best_diff:
            best_h, best_diff = h, diff
    return best_h

# ============================================================
# 4. TRAINING & EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    correct, total = 0, 0
    total_loss = 0.0
    gate_sum, pipe_sum, stat_n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, stats = model(xb, return_stats=True)
        loss = criterion(logits, yb)
        
        bs = xb.shape[0]
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += bs
        if stats:
            gate_sum += stats.get("gate_mean", 0)
            pipe_sum += stats.get("pipe_norm", 0)
            stat_n += 1
            
    out = {"acc": correct / total, "loss": total_loss / total}
    if stat_n:
        out["gate"] = gate_sum / stat_n
        out["pipe"] = pipe_sum / stat_n
    return out

def train_one(seed, mode, X_train, y_train, X_test, y_test):
    seed_everything(seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    
    hidden = choose_hidden(mode)
    model = build_model(mode, hidden).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"MODE={mode} | SEED={seed} | hidden={hidden} | params={params:,}")
    print("=" * 80)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_ep = 0.0, 0
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
            best_acc, best_ep = val["acc"], epoch
        gate_str = f" gate={val['gate']:.3f} pipe={val['pipe']:.2f}" if "gate" in val else f" pipe={val['pipe']:.2f}"
        print(f"ep {epoch:02d} | val_loss={val['loss']:.4f} | val_acc={val['acc']:.4f} | best={best_acc:.4f}@{best_ep} {gate_str}")

    print(f"FINAL {mode}: Best Acc {best_acc:.4f} | Time: {time.time()-start:.1f}s")
    return {"mode": mode, "params": params, "best_acc": best_acc, "best_epoch": best_ep}

# ============================================================
# 5. EXECUTION
# ============================================================

X_train, y_train, meta = make_distracted_dataset(TRAIN_N, seed=1000)
X_test, y_test, _ = make_distracted_dataset(TEST_N, seed=2000)

modes = ["lstm", "lstm_residual", "fen_rnn", "fen_lstm"] if RUN_MODE == "all_quick" else [RUN_MODE]
results = [train_one(SEEDS[0], mode, X_train, y_train, X_test, y_test) for mode in modes]

print("\n\n" + "#" * 80)
print("SUMMARY")
print("#" * 80)
for r in results:
    print(f"MODE: {r['mode']:<15} | Params: {r['params']} | Best Acc: {r['best_acc']:.4f} @ Ep {r['best_epoch']}")
