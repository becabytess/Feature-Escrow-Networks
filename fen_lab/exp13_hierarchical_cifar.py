# ==============================================================================
# FEN / Hierarchical FEN-Sandwich RNN on CIFAR-100
# ==============================================================================
# Core Concept:
#   A hierarchical "divide and conquer" FEN-Sandwich model on sequential CIFAR-100:
#     - Processes pixel-by-pixel (T=1024, C=3) by dividing the image into 32 rows.
#     - Runs a local FEN-Sandwich over each row (length 32) in parallel.
#     - Runs a global FEN-Sandwich over the row summaries (length 32).
#     - Auto-matches size to target exactly ~100k parameters.
# ==============================================================================

import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ CONFIG ----------------------------------------
FAST_MODE = True
INPUT_MODE = "pixel"
PATCH_SIZE = 1

if FAST_MODE:
    SEEDS = [1]
    EPOCHS = 15
    PRINT_EVERY = 1
    TARGET_PARAMS = 100000
else:
    SEEDS = [1, 2, 3]
    EPOCHS = 30
    PRINT_EVERY = 1
    TARGET_PARAMS = 100000

BATCH_SIZE = 256
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128
MIN_H, MAX_H = 16, 256
AUTO_MATCH_PARAMS = True

TRAIN_PER_CLASS = 150
TEST_PER_CLASS = 20
NUM_CLASSES = 100

USE_CUDA_GRAPHS = True
CUDA_GRAPH_WARMUP_STEPS = 3

MODEL_ORDER = [
    "standard_hrnn",
    "standard_hrnn_residual",
    "fen_roll_hierarchical",
    "fen_sandwich_hierarchical",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    USE_CUDA_GRAPHS = False

print(
    f"Device: {DEVICE} | CIFAR-100 seq | INPUT_MODE={INPUT_MODE} | "
    f"FAST_MODE={FAST_MODE} | EPOCHS={EPOCHS} | BATCH={BATCH_SIZE} | "
    f"CUDA_GRAPHS={USE_CUDA_GRAPHS}"
)
print("Models to compare:", MODEL_ORDER)


# ------------------------------ UTILS -----------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------ DATA ------------------------------------------
def unpickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def find_cifar100_pickles():
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        for root, _dirs, files in os.walk(kaggle_root):
            if "train" in files and "test" in files:
                train_p = os.path.join(root, "train")
                test_p = os.path.join(root, "test")
                if os.path.isfile(train_p) and os.path.isfile(test_p):
                    return train_p, test_p

    try:
        import kagglehub
        path = kagglehub.dataset_download("fedesoriano/cifar100")
        for root, _dirs, files in os.walk(path):
            if "train" in files and "test" in files:
                train_p = os.path.join(root, "train")
                test_p = os.path.join(root, "test")
                if os.path.isfile(train_p) and os.path.isfile(test_p):
                    return train_p, test_p
    except Exception as e:
        pass

    local_candidates = ["./cifar-100-python", "./data/cifar-100-python"]
    for c in local_candidates:
        if os.path.isdir(c):
            train_p = os.path.join(c, "train")
            test_p = os.path.join(c, "test")
            if os.path.isfile(train_p) and os.path.isfile(test_p):
                return train_p, test_p

    # download via torchvision if all fails
    from torchvision import datasets
    datasets.CIFAR100(root="./data", train=True, download=True)
    datasets.CIFAR100(root="./data", train=False, download=True)
    c = "./data/cifar-100-python"
    return os.path.join(c, "train"), os.path.join(c, "test")


def _images_to_sequence(imgs, input_mode, patch_size):
    N = imgs.shape[0]
    P = patch_size
    T = (32 // P) * (32 // P)
    C = 3 * P * P
    patches = []
    for r in range(32 // P):
        for c in range(32 // P):
            patch = imgs[:, :, r*P:(r+1)*P, c*P:(c+1)*P]
            patches.append(patch.transpose(0, 2, 3, 1).reshape(N, 1, C))
    return np.concatenate(patches, axis=1)


def load_cifar100_sequence(input_mode="patch", patch_size=4, train_per_class=150, test_per_class=20):
    train_path, test_path = find_cifar100_pickles()
    tr_dict = unpickle(train_path)
    te_dict = unpickle(test_path)

    x_tr_raw = tr_dict[b"data"].astype(np.float32) / 255.0
    y_tr_raw = np.array(tr_dict[b"fine_labels"], dtype=np.int64)
    x_te_raw = te_dict[b"data"].astype(np.float32) / 255.0
    y_te_raw = np.array(te_dict[b"fine_labels"], dtype=np.int64)

    imgs_tr = x_tr_raw.reshape(-1, 3, 32, 32)
    imgs_te = x_te_raw.reshape(-1, 3, 32, 32)

    rng = np.random.default_rng(42)
    tr_idx, te_idx = [], []
    for c in range(NUM_CLASSES):
        tr_c = np.where(y_tr_raw == c)[0]
        te_c = np.where(y_te_raw == c)[0]
        tr_idx.extend(rng.choice(tr_c, train_per_class, replace=False))
        te_idx.extend(rng.choice(te_c, test_per_class, replace=False))
    tr_idx = np.array(tr_idx)
    te_idx = np.array(te_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)

    imgs_tr, y_tr = imgs_tr[tr_idx], y_tr_raw[tr_idx]
    imgs_te, y_te = imgs_te[te_idx], y_te_raw[te_idx]

    x_tr = _images_to_sequence(imgs_tr, input_mode, patch_size)
    x_te = _images_to_sequence(imgs_te, input_mode, patch_size)

    flat = x_tr.reshape(-1, x_tr.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    x_tr = (x_tr - mean) / (std + 1e-8)
    x_te = (x_te - mean) / (std + 1e-8)

    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    x_te = torch.tensor(x_te, dtype=torch.float32, device=DEVICE)
    y_te = torch.tensor(y_te, dtype=torch.long, device=DEVICE)

    meta = {
        "name": "cifar100",
        "input_mode": input_mode,
        "input_dim": int(x_tr.shape[-1]),
        "num_classes": NUM_CLASSES,
        "seq_len": int(x_tr.shape[1]),
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "patch_size": patch_size if input_mode == "patch" else None,
    }
    print(
        f"CIFAR-100 seq ready on {DEVICE}: train={tuple(x_tr.shape)} "
        f"test={tuple(x_te.shape)} mode={input_mode} "
        f"T={meta['seq_len']} C={meta['input_dim']} classes={NUM_CLASSES}"
    )
    return x_tr, y_tr, x_te, y_te, meta


def iterate_batches(x, y, batch_size, shuffle):
    n = x.shape[0]
    n_batches = n // batch_size
    if n_batches == 0:
        return
    if shuffle:
        perm = torch.randperm(n, device=x.device)
    else:
        perm = torch.arange(n, device=x.device)
    for i in range(n_batches):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        yield x[idx], y[idx]


# ------------------------------ MODELS ----------------------------------------
def _mlp_head(in_dim, out_dim, width=HEAD_WIDTH):
    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.ReLU(),
        nn.Linear(width, out_dim),
    )


class ResidualRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.hdim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.head = _mlp_head(hidden_dim, num_classes)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        xp = self.x_proj(x)
        for t in range(T):
            z = h + xp[:, t]
            h = torch.tanh(self.core(z) + z)
        logits = self.head(h)
        if return_stats:
            return logits, {
                "pipe_norm": h.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": x.new_tensor(float("nan")),
            }
        return logits


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = _mlp_head(hidden_dim, num_classes)

    def forward(self, x, return_stats=False):
        _out, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        logits = self.head(h)
        if return_stats:
            return logits, {
                "pipe_norm": h.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": x.new_tensor(float("nan")),
            }
        return logits


class FENRoll(nn.Module):
    """Standard FEN Roll (Blind gates)."""
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.hdim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.roll_gate = nn.Linear(hidden_dim, 1)
        self.head = _mlp_head(hidden_dim * 2, num_classes)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        xp = self.x_proj(x)
        g_acc = x.new_zeros(())

        for t in range(T):
            z = h + xp[:, t]
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            v = self.v_proj(D)
            h = f - D
            gamma = torch.sigmoid(self.roll_gate(f))
            E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v
            g_acc = g_acc + g.detach().mean()

        logits = self.head(torch.cat([h, E], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h.detach().norm(dim=-1).mean(),
                "gate": g_acc / max(T, 1),
                "escrow_norm": E.detach().norm(dim=-1).mean(),
            }
        return logits


class FENHierarchicalSandwich(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chunks=32):
        super().__init__()
        self.hdim = hidden_dim
        self.num_chunks = num_chunks
        
        # --- Local Sandwich (processes each row) ---
        self.local_gru1 = nn.RNN(input_dim, hidden_dim, batch_first=True)
        # FEN Roll compression params for local pass
        self.local_gate = nn.Linear(hidden_dim, hidden_dim)
        self.local_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.local_roll_gate = nn.Linear(hidden_dim, 1)
        self.local_gru2 = nn.RNN(input_dim, hidden_dim, batch_first=True)
        
        # --- Global Sandwich (processes the combined row summaries and escrows) ---
        # input_dim for global RNNs is 2 * hidden_dim
        self.global_gru1 = nn.RNN(hidden_dim * 2, hidden_dim, batch_first=True)
        # FEN Roll compression params for global pass
        self.global_gate = nn.Linear(hidden_dim, hidden_dim)
        self.global_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_roll_gate = nn.Linear(hidden_dim, 1)
        self.global_gru2 = nn.RNN(hidden_dim * 2, hidden_dim, batch_first=True)
        
        self.head = _mlp_head(hidden_dim * 2, num_classes)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        chunk_len = T // self.num_chunks
        
        # 1. Chunk and flatten
        x_chunked = x.view(B, self.num_chunks, chunk_len, C)
        x_flat = x_chunked.view(B * self.num_chunks, chunk_len, C)
        
        # 2. Local Pass 1
        H1_local, _ = self.local_gru1(x_flat)
        
        # 3. Local FEN Roll Compression
        E_local = x.new_zeros(B * self.num_chunks, self.hdim)
        for t in range(chunk_len):
            ht = H1_local[:, t]
            g = torch.sigmoid(self.local_gate(ht))
            D = g * ht
            v = self.local_v_proj(D)
            gamma = torch.sigmoid(self.local_roll_gate(ht))
            E_local = (1.0 - gamma) * E_local + gamma * torch.roll(E_local, shifts=1, dims=-1) + v
            
        # 4. Local Pass 2 (initialized with E_local)
        h0_local = E_local.unsqueeze(0)
        _, hn_local2 = self.local_gru2(x_flat, h0_local)
        chunk_summaries = hn_local2[-1] # (B * num_chunks, hdim)
        
        # 5. Assemble sequence representations: concatenate active state and escrow state
        E_local_seq = E_local.view(B, self.num_chunks, self.hdim)
        chunk_seq_raw = chunk_summaries.view(B, self.num_chunks, self.hdim)
        chunk_seq = torch.cat([chunk_seq_raw, E_local_seq], dim=-1) # [B, num_chunks, 2 * hdim]
        
        # 6. Global Pass 1
        H1_global, _ = self.global_gru1(chunk_seq)
        
        # 7. Global FEN Roll Compression
        E_global = x.new_zeros(B, self.hdim)
        for t in range(self.num_chunks):
            ht = H1_global[:, t]
            g = torch.sigmoid(self.global_gate(ht))
            D = g * ht
            v = self.global_v_proj(D)
            gamma = torch.sigmoid(self.global_roll_gate(ht))
            E_global = (1.0 - gamma) * E_global + gamma * torch.roll(E_global, shifts=1, dims=-1) + v
            
        # 8. Global Pass 2 (initialized with E_global)
        h0_global = E_global.unsqueeze(0)
        _, hn_global2 = self.global_gru2(chunk_seq, h0_global)
        h_last = hn_global2[-1] # (B, hdim)
        
        logits = self.head(torch.cat([h_last, E_global], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h_last.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": E_global.detach().norm(dim=-1).mean(),
            }
        return logits


class StandardHRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chunks=32):
        super().__init__()
        self.hdim = hidden_dim
        self.num_chunks = num_chunks
        
        self.local_rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.global_rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.head = _mlp_head(hidden_dim, num_classes)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        chunk_len = T // self.num_chunks
        
        # 1. Chunk and flatten
        x_chunked = x.view(B, self.num_chunks, chunk_len, C)
        x_flat = x_chunked.view(B * self.num_chunks, chunk_len, C)
        
        # 2. Local pass
        _, hn_local = self.local_rnn(x_flat)
        chunk_summaries = hn_local[-1] # (B * num_chunks, hdim)
        
        # 3. Assemble sequence of summaries
        chunk_seq = chunk_summaries.view(B, self.num_chunks, self.hdim)
        
        # 4. Global pass
        _, hn_global = self.global_rnn(chunk_seq)
        h_last = hn_global[-1] # (B, hdim)
        
        logits = self.head(h_last)
        if return_stats:
            return logits, {
                "pipe_norm": h_last.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": x.new_tensor(float("nan")),
            }
        return logits


class ResidualRNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hdim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim, device=x.device, dtype=x.dtype)
        xp = self.x_proj(x)
        for t in range(T):
            z = h + xp[:, t]
            h = torch.tanh(self.core(z) + z)
        return h


class StandardHRNNResidual(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chunks=32):
        super().__init__()
        self.hdim = hidden_dim
        self.num_chunks = num_chunks
        
        self.local_rnn = ResidualRNNBlock(input_dim, hidden_dim)
        self.global_rnn = ResidualRNNBlock(hidden_dim, hidden_dim)
        self.head = _mlp_head(hidden_dim, num_classes)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        chunk_len = T // self.num_chunks
        
        # 1. Chunk and flatten
        x_chunked = x.view(B, self.num_chunks, chunk_len, C)
        x_flat = x_chunked.view(B * self.num_chunks, chunk_len, C)
        
        # 2. Local pass
        chunk_summaries = self.local_rnn(x_flat) # (B * num_chunks, hdim)
        
        # 3. Assemble sequence of summaries
        chunk_seq = chunk_summaries.view(B, self.num_chunks, self.hdim)
        
        # 4. Global pass
        h_last = self.global_rnn(chunk_seq) # (B, hdim)
        
        logits = self.head(h_last)
        if return_stats:
            return logits, {
                "pipe_norm": h_last.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": x.new_tensor(float("nan")),
            }
        return logits


class FENRollBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hdim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.roll_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim, device=x.device, dtype=x.dtype)
        E = x.new_zeros(B, self.hdim, device=x.device, dtype=x.dtype)
        xp = self.x_proj(x)
        for t in range(T):
            z = h + xp[:, t]
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            v = self.v_proj(D)
            h = f - D
            gamma = torch.sigmoid(self.roll_gate(f))
            E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v
        return h, E


class FENRollHierarchical(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chunks=32):
        super().__init__()
        self.hdim = hidden_dim
        self.num_chunks = num_chunks
        
        self.local_fen = FENRollBlock(input_dim, hidden_dim)
        self.global_fen = FENRollBlock(hidden_dim * 2, hidden_dim)
        self.head = _mlp_head(hidden_dim * 2, num_classes)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        chunk_len = T // self.num_chunks
        
        # 1. Chunk and flatten
        x_chunked = x.view(B, self.num_chunks, chunk_len, C)
        x_flat = x_chunked.view(B * self.num_chunks, chunk_len, C)
        
        # 2. Local pass
        h_local, E_local = self.local_fen(x_flat)
        
        # 3. Concatenate and assemble
        chunk_combined = torch.cat([h_local, E_local], dim=-1) # [B * num_chunks, 2 * hdim]
        chunk_seq = chunk_combined.view(B, self.num_chunks, self.hdim * 2)
        
        # 4. Global pass
        h_global, E_global = self.global_fen(chunk_seq)
        
        logits = self.head(torch.cat([h_global, E_global], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h_global.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": E_global.detach().norm(dim=-1).mean(),
            }
        return logits


# ------------------------------ BUILD & PARAMS MATCH --------------------------
MODEL_SPECS = {
    "standard_hrnn": dict(kind="standard_hrnn"),
    "standard_hrnn_residual": dict(kind="standard_hrnn_residual"),
    "fen_roll_hierarchical": dict(kind="fen_roll_hierarchical"),
    "fen_sandwich_hierarchical": dict(kind="fen_sandwich_hierarchical"),
}


def build(name, input_dim, num_classes, hidden_dim):
    spec = MODEL_SPECS[name]
    if spec["kind"] == "standard_hrnn":
        return StandardHRNN(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "standard_hrnn_residual":
        return StandardHRNNResidual(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "fen_roll_hierarchical":
        return FENRollHierarchical(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "fen_sandwich_hierarchical":
        return FENHierarchicalSandwich(input_dim, hidden_dim, num_classes)
    raise ValueError(name)


_HIDDEN_CACHE = {}


def choose_hidden(name, input_dim, num_classes):
    key = (name, input_dim, num_classes, TARGET_PARAMS)
    if key in _HIDDEN_CACHE:
        return _HIDDEN_CACHE[key]
    if not AUTO_MATCH_PARAMS:
        _HIDDEN_CACHE[key] = 64
        return 64
    lo, hi = MIN_H, MAX_H
    best_h, best_diff = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(build(name, input_dim, num_classes, mid))
        d = abs(n - TARGET_PARAMS)
        if d < best_diff:
            best_h, best_diff = mid, d
        if n < TARGET_PARAMS:
            lo = mid + 1
        elif n > TARGET_PARAMS:
            hi = mid - 1
        else:
            break
    for h in range(max(MIN_H, best_h - 4), min(MAX_H, best_h + 4) + 1):
        n = count_params(build(name, input_dim, num_classes, h))
        d = abs(n - TARGET_PARAMS)
        if d < best_diff:
            best_h, best_diff = h, d
    _HIDDEN_CACHE[key] = best_h
    return best_h


# ------------------------------ CUDA GRAPH ------------------------------------
def _snapshot_state(model, opt):
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt_state = {}
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if st:
                opt_state[p] = {
                    k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()
                }
    return params, opt_state


def _restore_state(model, opt, params, opt_state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(params[n])
        for group in opt.param_groups:
            for p in group["params"]:
                if p in opt_state:
                    st = opt.state[p]
                    for k, v in opt_state[p].items():
                        if torch.is_tensor(st.get(k)):
                            st[k].copy_(v)
                        else:
                            st[k] = v


def _try_build_cuda_graph(model, opt, criterion, static_x, static_y):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(CUDA_GRAPH_WARMUP_STEPS):
            opt.zero_grad(set_to_none=True)
            out = model(static_x)
            loss = criterion(out, static_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    opt.zero_grad(set_to_none=False)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        static_out = model(static_x)
        static_loss = criterion(static_out, static_y)
        static_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    return g


# ------------------------------ TRAIN -----------------------------------------
@torch.no_grad()
def evaluate(model, X, y, batch_size):
    model.eval()
    total_correct = 0
    total_n = 0
    pipe_sum = gate_sum = esc_sum = 0.0
    n_batches = 0
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits, st = model(xb, return_stats=True)
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
        total_n += yb.numel()
        pipe_sum += float(st["pipe_norm"].item())
        g, e = st["gate"], st["escrow_norm"]
        if torch.isfinite(g):
            gate_sum += float(g.item())
        if torch.isfinite(e):
            esc_sum += float(e.item())
        n_batches += 1
    return {
        "acc": total_correct / max(total_n, 1),
        "pipe": pipe_sum / max(n_batches, 1),
        "gate": gate_sum / max(n_batches, 1) if n_batches else float("nan"),
        "escrow": esc_sum / max(n_batches, 1) if n_batches else float("nan"),
    }


def train_one(name, X_train, y_train, X_test, y_test, meta, seed, epochs, batch_size):
    seed_everything(seed)
    in_dim = meta["input_dim"]
    n_cls = meta["num_classes"]
    T = meta["seq_len"]
    h = choose_hidden(name, in_dim, n_cls)
    model = build(name, in_dim, n_cls, h).to(DEVICE)
    n_params = count_params(model)

    print(f"\n--- Model: {name} ---")
    print(
        f"  hidden={h}  params={n_params}  epochs={epochs}  seed={seed}  "
        f"T={T}  batch={batch_size}"
    )

    capturable = DEVICE.type == "cuda"
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, capturable=capturable
    )
    criterion = nn.CrossEntropyLoss()

    graph = None
    static_x = static_y = None
    if USE_CUDA_GRAPHS and DEVICE.type == "cuda":
        try:
            static_x = torch.zeros(
                batch_size, T, in_dim, device=DEVICE, dtype=X_train.dtype
            )
            static_y = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
            clean_p, clean_o = _snapshot_state(model, opt)
            graph = _try_build_cuda_graph(model, opt, criterion, static_x, static_y)
            _restore_state(model, opt, clean_p, clean_o)
            print("  [CUDA graph capture OK]")
        except Exception as e:
            print(f"  [CUDA graph failed ({type(e).__name__}: {e}); eager]")
            graph = None
            seed_everything(seed)
            model = build(name, in_dim, n_cls, h).to(DEVICE)
            opt = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )

    best_acc, best_ep, best_snap = -1.0, 0, None
    history = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        for xb, yb in iterate_batches(X_train, y_train, batch_size, shuffle=True):
            if graph is not None:
                static_x.copy_(xb)
                static_y.copy_(yb)
                graph.replay()
            else:
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

        val = evaluate(model, X_test, y_test, batch_size)
        history.append(val["acc"])
        if val["acc"] > best_acc:
            best_acc, best_ep, best_snap = val["acc"], ep, dict(val)

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            print(
                f"    ep {ep:02d}/{epochs}  acc={val['acc']:.3f}  "
                f"pipe={val['pipe']:.2f}  [{(time.time() - ep_t0):.1f}s]"
            )

    elapsed = time.time() - t0
    first_good_ep = epochs
    target_target = 0.9 * best_acc
    for idx, acc in enumerate(history, 1):
        if acc >= target_target:
            first_good_ep = idx
            break

    best_snap["time_s"] = elapsed
    best_snap["to_best_acc_ep"] = best_ep
    best_snap["to_90_pct_ep"] = first_good_ep
    best_snap["ep1"] = history[0]
    best_snap["ep2"] = history[1] if len(history) > 1 else history[0]
    best_snap["last"] = history[-1]
    best_snap["params"] = n_params
    best_snap["hidden"] = h
    return best_snap


def main():
    seed_everything(42)
    X_train, y_train, X_test, y_test, meta = load_cifar100_sequence(
        input_mode=INPUT_MODE,
        patch_size=PATCH_SIZE,
        train_per_class=TRAIN_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
    )

    summary = defaultdict(list)
    for model_name in MODEL_ORDER:
        for seed in SEEDS:
            snap = train_one(
                model_name,
                X_train,
                y_train,
                X_test,
                y_test,
                meta,
                seed,
                EPOCHS,
                BATCH_SIZE,
            )
            summary[model_name].append(snap)

    print("\n" + "=" * 80)
    print(f"SUMMARY TABLE: task=cifar100_patch{PATCH_SIZE} seeds={SEEDS} target_params≈{TARGET_PARAMS}")
    print("| Model | Best Acc | Epoch 1 | Epoch 2 | @Ep | to90% | Pipe Norm | Hidden | Params |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in MODEL_ORDER:
        runs = summary[name]
        accs = [r["acc"] for r in runs]
        ep1s = [r["ep1"] for r in runs]
        ep2s = [r["ep2"] for r in runs]
        best_eps = [r["to_best_acc_ep"] for r in runs]
        to90s = [r["to_90_pct_ep"] for r in runs]
        pipes = [r["pipe"] for r in runs]
        hiddens = [r["hidden"] for r in runs]
        params = [r["params"] for r in runs]

        print(
            f"| {name:<19} | "
            f"{np.mean(accs):.4f} | "
            f"{np.mean(ep1s):.4f} | "
            f"{np.mean(ep2s):.4f} | "
            f"{int(np.mean(best_eps)):3d} | "
            f"{np.mean(to90s):.1f} | "
            f"{np.mean(pipes):.2f} | "
            f"{int(np.mean(hiddens)):3d} | "
            f"{int(np.mean(params)):6d} |"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
