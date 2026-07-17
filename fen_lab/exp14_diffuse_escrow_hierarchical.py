# ==============================================================================
# FEN / Exp 14: Top-Down Escrow Diffusion Hierarchical FEN on CIFAR-100
# ==============================================================================
# Core Concept:
#   A single-pass hierarchical FEN model with Top-Down Escrow Diffusion:
#     - Local pass: processes sequence in 32 chunks (rows), yielding local hidden
#       states h_local and local escrows E_local for each chunk.
#     - Global pass: sweeps across the 32 chunk representations. As global state
#       h_global steps forward (accumulating context across chunks 1..k), it
#       diffuses top-down contextual information directly into chunk k's local escrow E_local[k].
#     - The local escrows act as a distributed RAM / communication channel, enriched
#       by global context.
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
    "standard_hgru",
    "fen_roll_gru_hierarchical",
    "fen_diffuse_gru_hierarchical",
    "fen_diffuse_rnn_hierarchical",
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
    if os.path.exists(kaggle_root):
        for root, _, files in os.walk(kaggle_root):
            if "train" in files and "test" in files:
                return os.path.join(root, "train"), os.path.join(root, "test")

    colab_cache = "/root/.cache/cifar100"
    if os.path.exists(colab_cache):
        for root, _, files in os.walk(colab_cache):
            if "train" in files and "test" in files:
                return os.path.join(root, "train"), os.path.join(root, "test")

    data_dir = "./cifar-100-python"
    if os.path.exists(data_dir):
        return os.path.join(data_dir, "train"), os.path.join(data_dir, "test")

    try:
        import torchvision
        import torchvision.datasets as datasets
        os.makedirs(colab_cache, exist_ok=True)
        print("Downloading CIFAR-100 via torchvision to cache directory...")
        datasets.CIFAR100(root=colab_cache, train=True, download=True)
        datasets.CIFAR100(root=colab_cache, train=False, download=True)
        for root, _, files in os.walk(colab_cache):
            if "train" in files and "test" in files:
                return os.path.join(root, "train"), os.path.join(root, "test")
    except Exception as e:
        print(f"Torchvision download failed: {e}")

    raise FileNotFoundError("Could not find CIFAR-100 python dataset files ('train' and 'test').")


def load_cifar100_subsample(train_per_class=150, test_per_class=20, patch_size=1):
    train_path, test_path = find_cifar100_pickles()

    train_dict = unpickle(train_path)
    X_raw_tr = train_dict[b"data"]
    y_raw_tr = np.array(train_dict[b"fine_labels"])

    test_dict = unpickle(test_path)
    X_raw_te = test_dict[b"data"]
    y_raw_te = np.array(test_dict[b"fine_labels"])

    def subsample(X_raw, y_raw, n_per_class):
        X_sub, y_sub = [], []
        for c in range(100):
            idx = np.where(y_raw == c)[0]
            chosen = idx[:n_per_class]
            X_sub.append(X_raw[chosen])
            y_sub.append(y_raw[chosen])
        return np.vstack(X_sub), np.concatenate(y_sub)

    X_tr_sub, y_tr_sub = subsample(X_raw_tr, y_raw_tr, train_per_class)
    X_te_sub, y_te_sub = subsample(X_raw_te, y_raw_te, test_per_class)

    X_tr = X_tr_sub.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    X_te = X_te_sub.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0

    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
    std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    if patch_size == 1:
        T_seq, C_in = 1024, 3
        X_tr_seq = X_tr.reshape(-1, T_seq, C_in)
        X_te_seq = X_te.reshape(-1, T_seq, C_in)
    else:
        grid = 32 // patch_size
        T_seq = grid * grid
        C_in = patch_size * patch_size * 3

        def patchify(images):
            N = images.shape[0]
            out = np.zeros((N, grid, grid, C_in), dtype=np.float32)
            for r in range(grid):
                for c in range(grid):
                    patch = images[:, r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size, :]
                    out[:, r, c, :] = patch.reshape(N, -1)
            return out.reshape(N, T_seq, C_in)

        X_tr_seq = patchify(X_tr)
        X_te_seq = patchify(X_te)

    return (
        torch.tensor(X_tr_seq, dtype=torch.float32),
        torch.tensor(y_tr_sub, dtype=torch.long),
        torch.tensor(X_te_seq, dtype=torch.float32),
        torch.tensor(y_te_sub, dtype=torch.long),
        (T_seq, C_in),
    )


# ------------------------------ MODEL BLOCKS -----------------------------------
def _mlp_head(in_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, HEAD_WIDTH),
        nn.ReLU(),
        nn.Linear(HEAD_WIDTH, num_classes),
    )


# 1. Standard HGRU Baseline
class StandardHGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chunks=32):
        super().__init__()
        self.hdim = hidden_dim
        self.num_chunks = num_chunks

        self.local_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.global_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = _mlp_head(hidden_dim, num_classes)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        chunk_len = T // self.num_chunks

        x_chunked = x.view(B, self.num_chunks, chunk_len, C)
        x_flat = x_chunked.view(B * self.num_chunks, chunk_len, C)

        _, hn_local = self.local_gru(x_flat)
        chunk_summaries = hn_local[-1]

        chunk_seq = chunk_summaries.view(B, self.num_chunks, self.hdim)

        _, hn_global = self.global_gru(chunk_seq)
        h_last = hn_global[-1]

        logits = self.head(h_last)
        if return_stats:
            return logits, {
                "pipe_norm": h_last.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": x.new_tensor(float("nan")),
            }
        return logits


# 2. Standard Single-Pass Hierarchical FEN Roll (GRU)
class FENRollGRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hdim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.roll_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim, device=x.device, dtype=x.dtype)
        E = x.new_zeros(B, self.hdim, device=x.device, dtype=x.dtype)
        for t in range(T):
            xt = x[:, t]
            h = self.gru_cell(xt, h)
            g = torch.sigmoid(self.gate(h))
            D = g * h
            v = self.v_proj(D)
            h = h - D
            gamma = torch.sigmoid(self.roll_gate(h))
            E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v
        return h, E


class FENRollGRUHierarchical(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chunks=32):
        super().__init__()
        self.hdim = hidden_dim
        self.num_chunks = num_chunks

        self.local_fen = FENRollGRUBlock(input_dim, hidden_dim)
        self.global_fen = FENRollGRUBlock(hidden_dim * 2, hidden_dim)
        self.head = _mlp_head(hidden_dim * 2, num_classes)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        chunk_len = T // self.num_chunks

        x_chunked = x.view(B, self.num_chunks, chunk_len, C)
        x_flat = x_chunked.view(B * self.num_chunks, chunk_len, C)

        h_local, E_local = self.local_fen(x_flat)

        chunk_combined = torch.cat([h_local, E_local], dim=-1)
        chunk_seq = chunk_combined.view(B, self.num_chunks, self.hdim * 2)

        h_global, E_global = self.global_fen(chunk_seq)

        logits = self.head(torch.cat([h_global, E_global], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h_global.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": E_global.detach().norm(dim=-1).mean(),
            }
        return logits


# 3. NEW: Top-Down Escrow Diffusion Hierarchical FEN (GRU)
class FENDiffuseGRUHierarchical(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chunks=32):
        super().__init__()
        self.hdim = hidden_dim
        self.num_chunks = num_chunks

        # Local pass: FEN Roll
        self.local_fen = FENRollGRUBlock(input_dim, hidden_dim)

        # Global pass: GRU Cell taking [h_local, E_local]
        self.global_cell = nn.GRUCell(hidden_dim * 2, hidden_dim)

        # Top-down diffusion parameters into local escrows
        self.diffuse_gate = nn.Linear(hidden_dim, hidden_dim)
        self.diffuse_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.diffuse_roll_gate = nn.Linear(hidden_dim, 1)

        self.head = _mlp_head(hidden_dim * 2, num_classes)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        chunk_len = T // self.num_chunks

        # 1. Chunk & flatten
        x_chunked = x.view(B, self.num_chunks, chunk_len, C)
        x_flat = x_chunked.view(B * self.num_chunks, chunk_len, C)

        # 2. Local FEN Pass
        h_local_flat, E_local_flat = self.local_fen(x_flat)

        # Reshape into sequence of chunks: [B, K, hdim]
        h_local_seq = h_local_flat.view(B, self.num_chunks, self.hdim)
        E_local_seq = E_local_flat.view(B, self.num_chunks, self.hdim)

        # 3. Global Pass with Top-Down Diffusion into Local Escrows
        h_global = x.new_zeros(B, self.hdim)
        diffused_escrows = []

        for k in range(self.num_chunks):
            # Combined input from chunk k: [B, 2 * hdim]
            inp_k = torch.cat([h_local_seq[:, k, :], E_local_seq[:, k, :]], dim=-1)

            # Recurrent update of global state (accumulates context across chunks 0..k)
            h_global = self.global_cell(inp_k, h_global)

            # Top-down diffusion into chunk k's local escrow
            Ek = E_local_seq[:, k, :]  # [B, hdim]
            g_diff = torch.sigmoid(self.diffuse_gate(h_global))
            D_diff = g_diff * h_global
            v_diff = self.diffuse_v_proj(D_diff)
            gamma_diff = torch.sigmoid(self.diffuse_roll_gate(h_global))

            # Diffuse global context into chunk k's local escrow vault
            Ek_diffused = (1.0 - gamma_diff) * Ek + gamma_diff * torch.roll(Ek, shifts=1, dims=-1) + v_diff
            diffused_escrows.append(Ek_diffused)

        # Stack diffused local escrows: [B, K, hdim]
        E_diffused_seq = torch.stack(diffused_escrows, dim=1)

        # Pool diffused local escrows to form global escrow representation
        E_global_pooled = E_diffused_seq.mean(dim=1)

        logits = self.head(torch.cat([h_global, E_global_pooled], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h_global.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": E_global_pooled.detach().norm(dim=-1).mean(),
            }
        return logits


# 4. NEW: Top-Down Escrow Diffusion Hierarchical FEN (Vanilla RNN)
class FENRollRNNBlock(nn.Module):
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


class FENDiffuseRNNHierarchical(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chunks=32):
        super().__init__()
        self.hdim = hidden_dim
        self.num_chunks = num_chunks

        self.local_fen = FENRollRNNBlock(input_dim, hidden_dim)

        self.global_x_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.global_core = nn.Linear(hidden_dim, hidden_dim)

        self.diffuse_gate = nn.Linear(hidden_dim, hidden_dim)
        self.diffuse_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.diffuse_roll_gate = nn.Linear(hidden_dim, 1)

        self.head = _mlp_head(hidden_dim * 2, num_classes)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        chunk_len = T // self.num_chunks

        x_chunked = x.view(B, self.num_chunks, chunk_len, C)
        x_flat = x_chunked.view(B * self.num_chunks, chunk_len, C)

        h_local_flat, E_local_flat = self.local_fen(x_flat)

        h_local_seq = h_local_flat.view(B, self.num_chunks, self.hdim)
        E_local_seq = E_local_flat.view(B, self.num_chunks, self.hdim)

        h_global = x.new_zeros(B, self.hdim)
        diffused_escrows = []

        for k in range(self.num_chunks):
            inp_k = torch.cat([h_local_seq[:, k, :], E_local_seq[:, k, :]], dim=-1)
            xp_k = self.global_x_proj(inp_k)
            z = h_global + xp_k
            h_global = torch.tanh(self.global_core(z) + z)

            Ek = E_local_seq[:, k, :]
            g_diff = torch.sigmoid(self.diffuse_gate(h_global))
            D_diff = g_diff * h_global
            v_diff = self.diffuse_v_proj(D_diff)
            gamma_diff = torch.sigmoid(self.diffuse_roll_gate(h_global))

            Ek_diffused = (1.0 - gamma_diff) * Ek + gamma_diff * torch.roll(Ek, shifts=1, dims=-1) + v_diff
            diffused_escrows.append(Ek_diffused)

        E_diffused_seq = torch.stack(diffused_escrows, dim=1)
        E_global_pooled = E_diffused_seq.mean(dim=1)

        logits = self.head(torch.cat([h_global, E_global_pooled], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h_global.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": E_global_pooled.detach().norm(dim=-1).mean(),
            }
        return logits


# ------------------------------ BUILD & PARAMS MATCH --------------------------
MODEL_SPECS = {
    "standard_hgru": dict(kind="standard_hgru"),
    "fen_roll_gru_hierarchical": dict(kind="fen_roll_gru_hierarchical"),
    "fen_diffuse_gru_hierarchical": dict(kind="fen_diffuse_gru_hierarchical"),
    "fen_diffuse_rnn_hierarchical": dict(kind="fen_diffuse_rnn_hierarchical"),
}


def build(name, input_dim, num_classes, hidden_dim):
    spec = MODEL_SPECS[name]
    if spec["kind"] == "standard_hgru":
        return StandardHGRU(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "fen_roll_gru_hierarchical":
        return FENRollGRUHierarchical(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "fen_diffuse_gru_hierarchical":
        return FENDiffuseGRUHierarchical(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "fen_diffuse_rnn_hierarchical":
        return FENDiffuseRNNHierarchical(input_dim, hidden_dim, num_classes)
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
        m = build(name, input_dim, num_classes, mid)
        p = count_params(m)
        diff = abs(p - TARGET_PARAMS)
        if diff < best_diff:
            best_diff = diff
            best_h = mid
        if p < TARGET_PARAMS:
            lo = mid + 1
        else:
            hi = mid - 1
    _HIDDEN_CACHE[key] = best_h
    return best_h


# ------------------------------ TRAINING LOOP ---------------------------------
def train_one(name, X_train, y_train, X_test, y_test, meta, seed, epochs, batch_size):
    seed_everything(seed)
    T, C = meta
    hdim = choose_hidden(name, C, NUM_CLASSES)
    model = build(name, C, NUM_CLASSES, hdim).to(DEVICE)
    params_actual = count_params(model)

    print(
        f"\n--- Model: {name} ---\n"
        f"  hidden={hdim}  params={params_actual}  epochs={epochs}  "
        f"seed={seed}  T={T}  batch={batch_size}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    num_train = X_train.shape[0]
    num_batches = (num_train + batch_size - 1) // batch_size

    g_runner = None
    if USE_CUDA_GRAPHS and DEVICE.type == "cuda":
        try:
            sample_x = X_train[:batch_size].to(DEVICE)
            sample_y = y_train[:batch_size].to(DEVICE)

            s_warmup = torch.cuda.Stream()
            s_warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_warmup):
                for _ in range(CUDA_GRAPH_WARMUP_STEPS):
                    optimizer.zero_grad(set_to_none=True)
                    out = model(sample_x)
                    loss = criterion(out, sample_y)
                    loss.backward()
                    optimizer.step()
            torch.cuda.current_stream().wait_stream(s_warmup)

            static_x = sample_x.clone()
            static_y = sample_y.clone()
            g = torch.cuda.CUDAGraph()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(g):
                static_out = model(static_x)
                static_loss = criterion(static_out, static_y)
                static_loss.backward()

            class GraphRunner:
                def __init__(self, graph, sx, sy, sout, sloss, opt, model_ref):
                    self.g = graph
                    self.sx = sx
                    self.sy = sy
                    self.sout = sout
                    self.sloss = sloss
                    self.opt = opt
                    self.model = model_ref

                def step(self, bx, by):
                    if bx.shape[0] != self.sx.shape[0]:
                        self.opt.zero_grad(set_to_none=True)
                        out = self.model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                        self.opt.step()
                        return loss.item()
                    else:
                        self.sx.copy_(bx)
                        self.sy.copy_(by)
                        self.opt.zero_grad(set_to_none=True)
                        self.g.replay()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                        self.opt.step()
                        return self.sloss.item()

            g_runner = GraphRunner(g, static_x, static_y, static_out, static_loss, optimizer, model)
            print("  [CUDA graph capture OK]")
        except Exception as e:
            print(f"  [CUDA graph capture failed: {e}]")
            g_runner = None

    history = []
    pipe_norms = []
    ep1_acc, ep2_acc = None, None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(num_train)
        X_tr_shuf = X_train[perm]
        y_tr_shuf = y_train[perm]

        total_loss = 0.0
        for b in range(num_batches):
            bx = X_tr_shuf[b * batch_size : (b + 1) * batch_size].to(DEVICE)
            by = y_tr_shuf[b * batch_size : (b + 1) * batch_size].to(DEVICE)

            if g_runner is not None:
                loss_val = g_runner.step(bx, by)
            else:
                optimizer.zero_grad(set_to_none=True)
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                loss_val = loss.item()

            total_loss += loss_val

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for b in range((X_test.shape[0] + batch_size - 1) // batch_size):
                bx = X_test[b * batch_size : (b + 1) * batch_size].to(DEVICE)
                by = y_test[b * batch_size : (b + 1) * batch_size].to(DEVICE)
                logits, stats = model(bx, return_stats=True)
                preds = logits.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += by.size(0)
                if b == 0:
                    pipe_norms.append(stats["pipe_norm"].item())

        acc = correct / max(total, 1)
        dt = time.time() - t0
        history.append((ep, acc, total_loss / num_batches))

        if ep == 1:
            ep1_acc = acc
        if ep == 2:
            ep2_acc = acc

        if ep % PRINT_EVERY == 0 or ep == epochs:
            latest_pipe = pipe_norms[-1] if pipe_norms else float("nan")
            print(
                f"    ep {ep:02d}/{epochs:02d}  acc={acc:.3f}  "
                f"pipe={latest_pipe:.2f}  [{dt:.1f}s]"
            )

    best_acc = max(h[1] for h in history)
    best_ep = [h[0] for h in history if h[1] == best_acc][0]

    thresh = 0.9 * best_acc
    to_90_ep = epochs
    for ep, acc, _ in history:
        if acc >= thresh:
            to_90_ep = ep
            break

    return {
        "acc": best_acc,
        "ep1": ep1_acc if ep1_acc is not None else history[0][1],
        "ep2": ep2_acc if ep2_acc is not None else (history[1][1] if len(history) > 1 else history[0][1]),
        "to_best_acc_ep": best_ep,
        "to_90_pct_ep": to_90_ep,
        "pipe": float(np.mean(pipe_norms)) if pipe_norms else float("nan"),
        "hidden": hdim,
        "params": params_actual,
    }


# ------------------------------ MAIN ------------------------------------------
def main():
    print("Loading subsampled CIFAR-100 dataset...")
    X_train, y_train, X_test, y_test, meta = load_cifar100_subsample(
        train_per_class=TRAIN_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
        patch_size=PATCH_SIZE,
    )
    T, C = meta
    print(
        f"CIFAR-100 seq ready on {DEVICE}: train={tuple(X_train.shape)} "
        f"test={tuple(X_test.shape)} mode={INPUT_MODE} T={T} C={C} classes={NUM_CLASSES}\n"
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
            f"| {name:<28} | "
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
