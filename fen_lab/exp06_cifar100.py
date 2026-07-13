# ==============================================================================
# fen_lab / EXP06 — Sequential CIFAR-100 with frozen FEN operators
# ==============================================================================
# Hard transfer after foundation + 1D real data.
#
# Historical sequential CIFAR-100 (pixel stream T=1024, C=3): archived FEN and
# LSTM hovered ~14% in ~15 epochs — above chance (1%) but poor.
#
# Question: do fen_lab operators (deplete + bag / hard / roll + concat deliver)
# move the needle — and does *topology match* (patch tokens vs raw pixels)?
#
# INPUT modes
#   pixel  — legacy raster: (N, 1024, 3)  fair compare to old ~14% runs
#   patch  — 4×4 non-overlap patches → (N, 64, 48)  primary new bet
#
# Models (~TARGET_PARAMS)
#   residual      residual tanh RNN, no escrow
#   fen_bag       deplete + bag + head([h,E])
#   fen_hard_bag  hard pointer tape + bag
#   fen_roll      channel-roll escrow
#   fen_hybrid    bag + roll vaults (dual archive, concat both)
#   lstm          classical baseline
#
# Data: Kaggle CIFAR-100 Python pickles (no torchvision download).
#   Looks under /kaggle/input/** for train/test/meta, else local paths.
# Subsample: 150 train / 20 test images per fine class (same spirit as old runs).
#
# Speed: GPU preload, full batches, binary width search, TF32, CUDA graphs.
#
# Kaggle: paste whole file into a GPU notebook → Run.
# Deps: torch, numpy. pickle in stdlib.
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

# "patch" = primary new experiment | "pixel" = legacy T=1024 protocol
INPUT_MODE = "patch"  # "patch" | "pixel"
PATCH_SIZE = 4  # only used when INPUT_MODE == "patch" → T = (32/P)^2, C = 3*P*P

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

# pixel T=1024 needs smaller batches; patch T=64 can go larger
BATCH_SIZE = 128 if INPUT_MODE == "pixel" else 256
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128
TAPE_K = 8
EVENT_GATE_THRESH = 0.25
MIN_H, MAX_H = 16, 256
AUTO_MATCH_PARAMS = True

# Balanced subset (matches older sequential CIFAR scripts)
TRAIN_PER_CLASS = 150
TEST_PER_CLASS = 20
NUM_CLASSES = 100

USE_CUDA_GRAPHS = True
CUDA_GRAPH_WARMUP_STEPS = 3

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
print(
    "EXP06 — Frozen FEN on sequential CIFAR-100 "
    "(deplete + topology write + concat deliver; no reinject)"
)


# ------------------------------ UTILS -----------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------ DATA (Kaggle CIFAR-100) -----------------------
def unpickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def find_cifar100_pickles():
    """Locate train/test pickles under /kaggle/input or local data dirs."""
    # Kaggle: walk input tree for classic CIFAR-100 Python layout
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        print("Locating CIFAR-100 under /kaggle/input...")
        for root, _dirs, files in os.walk(kaggle_root):
            if "train" in files and "test" in files:
                train_p = os.path.join(root, "train")
                test_p = os.path.join(root, "test")
                if os.path.isfile(train_p) and os.path.isfile(test_p):
                    return train_p, test_p

    # Local fallbacks
    for d in (
        "./data/cifar-100-python",
        "./cifar-100-python",
        "./data",
        os.path.expanduser("~/cifar-100-python"),
    ):
        train_p = os.path.join(d, "train")
        test_p = os.path.join(d, "test")
        if os.path.isfile(train_p) and os.path.isfile(test_p):
            return train_p, test_p

    raise FileNotFoundError(
        "Could not find CIFAR-100 'train' and 'test' pickles. "
        "On Kaggle: add dataset fedesoriano/cifar100 (or any CIFAR-100 Python dump). "
        "Locally: place pickles under ./data/cifar-100-python/."
    )


def _to_images(raw: np.ndarray) -> np.ndarray:
    """(N, 3072) uint/float → (N, 3, 32, 32) float32 in [0,1]."""
    x = raw.astype(np.float32) / 255.0
    return x.reshape(-1, 3, 32, 32)


def _images_to_sequence(imgs: np.ndarray, mode: str, patch: int) -> np.ndarray:
    """
    imgs: (N, 3, 32, 32)
    pixel → (N, 1024, 3)
    patch → (N, (32/P)^2, 3*P*P)
    """
    if mode == "pixel":
        # CHW → sequence of RGB at each spatial location (row-major)
        return imgs.reshape(imgs.shape[0], 3, 1024).transpose(0, 2, 1).copy()

    if mode != "patch":
        raise ValueError(mode)
    if 32 % patch != 0:
        raise ValueError(f"PATCH_SIZE={patch} must divide 32")

    n, c, h, w = imgs.shape
    ph = pw = patch
    # non-overlapping patches via reshape
    # (N, 3, 32/P, P, 32/P, P) → (N, 32/P, 32/P, 3, P, P) → (N, T, 3*P*P)
    gh, gw = h // ph, w // pw
    x = imgs.reshape(n, c, gh, ph, gw, pw)
    x = x.transpose(0, 2, 4, 1, 3, 5)  # N, gh, gw, C, ph, pw
    x = x.reshape(n, gh * gw, c * ph * pw)
    return x.copy()


def make_cifar100_seq(
    input_mode: str = INPUT_MODE,
    patch_size: int = PATCH_SIZE,
    train_per_class: int = TRAIN_PER_CLASS,
    test_per_class: int = TEST_PER_CLASS,
):
    train_file, test_file = find_cifar100_pickles()
    print(f"Loading train from: {train_file}")
    print(f"Loading test from:  {test_file}")

    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)

    # keys may be bytes
    def _get(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(keys)

    x_train_raw = _get(train_dict, b"data", "data")
    y_train = np.array(_get(train_dict, b"fine_labels", "fine_labels"), dtype=np.int64)
    x_test_raw = _get(test_dict, b"data", "data")
    y_test = np.array(_get(test_dict, b"fine_labels", "fine_labels"), dtype=np.int64)

    imgs_tr = _to_images(x_train_raw)
    imgs_te = _to_images(x_test_raw)

    # balanced subset (fixed seed for reproducible split)
    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for label in range(NUM_CLASSES):
        tr = np.where(y_train == label)[0]
        te = np.where(y_test == label)[0]
        train_idx.extend(rng.choice(tr, train_per_class, replace=False).tolist())
        test_idx.extend(rng.choice(te, test_per_class, replace=False).tolist())

    imgs_tr = imgs_tr[train_idx]
    y_tr = y_train[train_idx]
    imgs_te = imgs_te[test_idx]
    y_te = y_test[test_idx]

    x_tr = _images_to_sequence(imgs_tr, input_mode, patch_size)
    x_te = _images_to_sequence(imgs_te, input_mode, patch_size)

    # per-feature normalize from train
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
        f"Data ready on {DEVICE}: train={tuple(x_tr.shape)} test={tuple(x_te.shape)} "
        f"mode={input_mode} T={meta['seq_len']} C={meta['input_dim']} "
        f"classes={NUM_CLASSES}"
    )
    print(
        f"  subset: {train_per_class}/class train, {test_per_class}/class test "
        f"→ n_train={meta['n_train']} n_test={meta['n_test']}"
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


class SeqFEN(nn.Module):
    """
    Frozen FEN for long sequences / image tokens.

    write_mode:
      bag     — additive E
      hard    — hard pointer tape + bag c
      roll    — channel-roll E
      hybrid  — bag E_b + roll E_r  (head sees both)
    Always: deplete; final concat deliver; no reinject.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, write_mode="bag", K=TAPE_K):
        super().__init__()
        assert write_mode in ("bag", "hard", "roll", "hybrid")
        self.hdim = hidden_dim
        self.write_mode = write_mode
        self.K = K
        self.has_tape = write_mode == "hard"
        self.is_hybrid = write_mode == "hybrid"

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        if write_mode in ("roll", "hybrid"):
            self.roll_gate = nn.Linear(hidden_dim, 1)
        else:
            self.roll_gate = None

        if self.has_tape:
            self.bag_proj = nn.Linear(hidden_dim, hidden_dim)
            self.tape_pool = nn.Linear(K * hidden_dim, hidden_dim)
            head_in = hidden_dim * 3
        elif self.is_hybrid:
            self.bag_proj = None
            self.tape_pool = None
            head_in = hidden_dim * 3  # h + E_bag + E_roll
        else:
            self.bag_proj = None
            self.tape_pool = None
            head_in = hidden_dim * 2

        self.head = _mlp_head(head_in, num_classes)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        E_roll = x.new_zeros(B, self.hdim) if self.is_hybrid else None
        c = x.new_zeros(B, self.hdim)
        E_tape = x.new_zeros(B, self.K, self.hdim) if self.has_tape else None
        ptr = (
            torch.zeros(B, dtype=torch.long, device=x.device) if self.has_tape else None
        )

        xp = self.x_proj(x)
        g_acc = x.new_zeros(())

        for t in range(T):
            z = h + xp[:, t]
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            v = self.v_proj(D)
            h = f - D

            if self.write_mode == "bag":
                E = E + v
            elif self.write_mode == "roll":
                gamma = torch.sigmoid(self.roll_gate(f))
                E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v
            elif self.write_mode == "hybrid":
                E = E + v
                gamma = torch.sigmoid(self.roll_gate(f))
                E_roll = (
                    (1.0 - gamma) * E_roll
                    + gamma * torch.roll(E_roll, shifts=1, dims=-1)
                    + v
                )
            else:  # hard
                one = F.one_hot(ptr, self.K).to(dtype=v.dtype)
                E_tape = E_tape + one.unsqueeze(-1) * v.unsqueeze(1)
                advance = (g.mean(dim=-1) > EVENT_GATE_THRESH).long()
                ptr = (ptr + advance) % self.K
                c = c + self.bag_proj(D)

            if return_stats:
                g_acc = g_acc + g.detach().mean()

        if self.has_tape:
            pooled = torch.tanh(self.tape_pool(E_tape.reshape(B, -1)))
            arch = torch.cat([pooled, c], dim=-1)
            esc_norm = E_tape.detach().norm(dim=-1).mean() + c.detach().norm(dim=-1).mean()
        elif self.is_hybrid:
            arch = torch.cat([E, E_roll], dim=-1)
            esc_norm = E.detach().norm(dim=-1).mean() + E_roll.detach().norm(dim=-1).mean()
        else:
            arch = E
            esc_norm = E.detach().norm(dim=-1).mean()

        logits = self.head(torch.cat([h, arch], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h.detach().norm(dim=-1).mean(),
                "gate": g_acc / max(T, 1),
                "escrow_norm": esc_norm,
            }
        return logits


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        # 1-layer: more width under param budget for long seq
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
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


MODEL_SPECS = {
    "residual": dict(kind="residual"),
    "fen_bag": dict(kind="fen", write_mode="bag"),
    "fen_hard_bag": dict(kind="fen", write_mode="hard"),
    "fen_roll": dict(kind="fen", write_mode="roll"),
    "fen_hybrid": dict(kind="fen", write_mode="hybrid"),
    "lstm": dict(kind="lstm"),
}
MODEL_ORDER = list(MODEL_SPECS.keys())


def build(name, input_dim, num_classes, hidden_dim):
    spec = MODEL_SPECS[name]
    if spec["kind"] == "residual":
        return ResidualRNN(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "lstm":
        return LSTMBaseline(input_dim, hidden_dim, num_classes)
    return SeqFEN(
        input_dim, hidden_dim, num_classes, write_mode=spec["write_mode"], K=TAPE_K
    )


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


# ------------------------------ TRAIN / EVAL ----------------------------------
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
        g = st["gate"]
        e = st["escrow_norm"]
        if torch.isfinite(g):
            gate_sum += float(g.item())
        if torch.isfinite(e):
            esc_sum += float(e.item())
        n_batches += 1
    acc = total_correct / max(total_n, 1)
    return {
        "acc": acc,
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
        f"T={T}  C={in_dim}  batch={batch_size}  mode={meta['input_mode']}"
    )

    capturable = DEVICE.type == "cuda"
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        capturable=capturable,
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
            clean_params, clean_opt = _snapshot_state(model, opt)
            graph = _try_build_cuda_graph(model, opt, criterion, static_x, static_y)
            _restore_state(model, opt, clean_params, clean_opt)
            print("  [CUDA graph capture OK]")
        except Exception as e:
            print(f"  [CUDA graph failed ({type(e).__name__}: {e}); eager path]")
            graph = None
            seed_everything(seed)
            model = build(name, in_dim, n_cls, h).to(DEVICE)
            opt = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )

    best_acc = -1.0
    best_ep = 0
    best_snap = None
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
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_ep = ep
            best_snap = dict(val)

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            dt = time.time() - ep_t0
            print(
                f"    ep {ep:02d}/{epochs}  acc={val['acc']:.3f}  "
                f"pipe={val['pipe']:.2f}  gate={val['gate']:.3f}  [{dt:.1f}s]"
            )

    elapsed = time.time() - t0
    print(
        f"  >> best acc={best_acc:.3f}  @ep{best_ep}  t={elapsed:.1f}s  "
        f"pipe={best_snap['pipe']:.2f}  graph={'yes' if graph else 'no'}"
    )
    return {
        "name": name,
        "acc": best_acc,
        "best_ep": best_ep,
        "pipe": best_snap["pipe"],
        "gate": best_snap["gate"],
        "params": n_params,
        "hidden": h,
        "time": elapsed,
        "graph": graph is not None,
    }


def main():
    print(
        f"Models: {MODEL_ORDER}\n"
        f"INPUT_MODE={INPUT_MODE} | TARGET_PARAMS≈{TARGET_PARAMS} | "
        f"EPOCHS={EPOCHS} | SEEDS={SEEDS}"
    )
    print(
        "Chance @100 classes ≈ 0.01. Historical pixel-seq FEN/LSTM ≈ 0.14 @15ep "
        "(poor, not floor)."
    )

    X_tr, y_tr, X_te, y_te, meta = make_cifar100_seq()

    # drop incomplete last batches for CUDA graphs
    bs = BATCH_SIZE
    n_tr = (X_tr.shape[0] // bs) * bs
    n_te = (X_te.shape[0] // bs) * bs
    if n_tr < X_tr.shape[0] or n_te < X_te.shape[0]:
        print(f"  truncating to full batches: train {n_tr} test {n_te} (batch={bs})")
        X_tr, y_tr = X_tr[:n_tr], y_tr[:n_tr]
        X_te, y_te = X_te[:n_te], y_te[:n_te]

    by_model = defaultdict(list)
    for seed in SEEDS:
        print(f"\n### seed={seed}")
        for name in MODEL_ORDER:
            row = train_one(
                name, X_tr, y_tr, X_te, y_te, meta, seed, EPOCHS, bs
            )
            by_model[name].append(row)

    print("\n" + "-" * 78)
    print(
        f"SUMMARY  cifar100  mode={meta['input_mode']}  T={meta['seq_len']}  "
        f"C={meta['input_dim']}  seeds={SEEDS}  target_params≈{TARGET_PARAMS}"
    )
    print("-" * 78)
    print(
        f"{'model':<14} {'acc':>7} {'±':>6} {'to_best':>8} "
        f"{'pipe':>7} {'params':>8} {'time_s':>8}"
    )
    for name in MODEL_ORDER:
        rows = by_model[name]
        accs = np.array([r["acc"] for r in rows], dtype=np.float64)
        eps = np.array([r["best_ep"] for r in rows], dtype=np.float64)
        pipes = np.array([r["pipe"] for r in rows], dtype=np.float64)
        times = np.array([r["time"] for r in rows], dtype=np.float64)
        params = rows[0]["params"]
        print(
            f"{name:<14} {accs.mean():7.3f} {accs.std():6.3f} "
            f"{eps.mean():8.1f} {pipes.mean():7.2f} {params:8d} {times.mean():8.1f}"
        )
    print("-" * 78)
    print(
        "Score: chance≈0.01; old pixel-seq ≈0.14. "
        "Does patch + roll/hybrid beat bag/residual/lstm? "
        "Watch pipe norms (residual fat vs FEN lean)."
    )
    print("DONE — paste this SUMMARY back for scoring.")
    print(
        "Tip: re-run with INPUT_MODE='pixel' for legacy T=1024 comparison; "
        "INPUT_MODE='patch' is the topology-matched default."
    )
    return by_model


if __name__ == "__main__":
    main()
