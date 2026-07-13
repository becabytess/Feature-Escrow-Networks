# ==============================================================================
# fen_lab / EXP11 — Sequential-stress / tokenization curve (regime map)
# ==============================================================================
# Step 1 after exp10 CIFAR transfer (no scaling, no new FEN variants).
#
# Question
#   How much of the FEN ranking (especially roll ≫ bag) is controlled by
#   *sequential stress* — longer T, thinner tokens — rather than CIFAR labels
#   alone?
#
#   exp10 already showed:
#     P4 (T=64,  C=48): gaps compressed (roll−bag peak ~+0.02)
#     P2 (T=256, C=12): gaps reopen   (roll−bag peak ~+0.12; bag ~floor)
#
#   This run fills a controlled curve on ONE protocol by varying only patch size.
#
# Protocol (match exp10)
#   CIFAR-100 fine, 150 train / 20 test per class, ~100k params, seed 1
#   PATCH_SIZES default: 8 → 4 → 2
#     P8: T=16,  C=192  — lowest sequential stress (short, fat)
#     P4: T=64,  C=48   — mid (exp10 main short-token run)
#     P2: T=256, C=12   — high (exp10 long-scan run)
#   Optional: add 1 to PATCH_SIZES → T=1024, C=3 (slow; max stress)
#
# Models (lean — discrimination, not invent-a-thon)
#   residual | fen_bag | fen_roll | fen_hybrid | lstm
#
# Metrics per stress level
#   peak, ep1, ep2, pipe
# Final table
#   roll−bag / roll−lstm / hybrid−roll gaps vs T (regime map)
#
# Success
#   Higher stress → larger roll−bag peak and/or ep1 gaps (roughly monotonic)
# Failure
#   Winner order scrambles randomly across patch sizes → don't overclaim defaults
#
# Kaggle/Colab: paste whole file → GPU → Run.
# Deps: torch, numpy. pickle in stdlib. Optional kagglehub.
# Data: fedesoriano/cifar100 (Python train/test pickles).
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

# Ordered low → high sequential stress (must divide 32)
# Add 1 for pixel-like T=1024 (slow). Do not add for a quick first pass.
PATCH_SIZES = [8, 4, 2]

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

LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128
TAPE_K = 8
MIN_H, MAX_H = 16, 256
AUTO_MATCH_PARAMS = True

TRAIN_PER_CLASS = 150
TEST_PER_CLASS = 20
NUM_CLASSES = 100

USE_CUDA_GRAPHS = True
CUDA_GRAPH_WARMUP_STEPS = 3

# Lean set: enough to rank write topology + residual/LSTM baselines
MODEL_ORDER = [
    "residual",
    "fen_bag",
    "fen_roll",
    "fen_hybrid",
    "lstm",
]


def batch_size_for_T(T: int) -> int:
    if T >= 512:
        return 64
    if T >= 256:
        return 128
    return 256


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    USE_CUDA_GRAPHS = False

print(
    f"Device: {DEVICE} | EXP11 stress curve | FAST_MODE={FAST_MODE} | "
    f"EPOCHS={EPOCHS} | PATCH_SIZES={PATCH_SIZES} | CUDA_GRAPHS={USE_CUDA_GRAPHS}"
)
print(
    "EXP11 — Regime map: does roll−bag gap grow with sequential stress?\n"
    "  Same CIFAR-100 subset / ~100k / seed; only patch size (T, C) changes.\n"
    f"  Chance floor = {1.0 / NUM_CLASSES:.3f} | Models: {MODEL_ORDER}"
)
print(
    "exp10 refs: P4 roll−bag peak≈+0.02 ep1≈+0.05 | "
    "P2 roll−bag peak≈+0.12 ep1≈+0.06"
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


def stress_label(patch: int) -> str:
    T = (32 // patch) ** 2
    C = 3 * patch * patch
    if patch >= 8:
        level = "low"
    elif patch >= 4:
        level = "mid"
    elif patch >= 2:
        level = "high"
    else:
        level = "max"
    return f"P{patch} T={T} C={C} ({level})"


# ------------------------------ DATA ------------------------------------------
def unpickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def find_cifar100_pickles():
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        print("Locating CIFAR-100 under /kaggle/input...")
        for root, _dirs, files in os.walk(kaggle_root):
            if "train" in files and "test" in files:
                train_p = os.path.join(root, "train")
                test_p = os.path.join(root, "test")
                if os.path.isfile(train_p) and os.path.isfile(test_p):
                    return train_p, test_p

    try:
        import kagglehub

        print("Downloading CIFAR-100 via kagglehub (fedesoriano/cifar100)...")
        path = kagglehub.dataset_download("fedesoriano/cifar100")
        for root, _dirs, files in os.walk(path):
            if "train" in files and "test" in files:
                return os.path.join(root, "train"), os.path.join(root, "test")
    except Exception as e:
        print(f"  kagglehub path failed ({type(e).__name__}: {e})")

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
        "Could not find CIFAR-100 'train' and 'test' pickles.\n"
        "  Kaggle: Add dataset fedesoriano/cifar100 (Python version).\n"
        "  Local: place pickles under ./data/cifar-100-python/."
    )


def _to_images(raw: np.ndarray) -> np.ndarray:
    x = raw.astype(np.float32) / 255.0
    return x.reshape(-1, 3, 32, 32)


def _images_to_sequence(imgs: np.ndarray, patch: int) -> np.ndarray:
    """Non-overlap patches → (N, T, C) with T=(32/P)^2, C=3*P*P."""
    if 32 % patch != 0:
        raise ValueError(f"PATCH_SIZE={patch} must divide 32")
    n, c, h, w = imgs.shape
    ph = pw = patch
    gh, gw = h // ph, w // pw
    x = imgs.reshape(n, c, gh, ph, gw, pw)
    x = x.transpose(0, 2, 4, 1, 3, 5)
    x = x.reshape(n, gh * gw, c * ph * pw)
    return x.copy()


def load_cifar100_images():
    """Load once; subset indices fixed (seed 42). Returns numpy imgs + labels."""
    train_file, test_file = find_cifar100_pickles()
    print(f"Loading train from: {train_file}")
    print(f"Loading test from:  {test_file}")

    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)

    def _get(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(keys)

    y_train = np.array(_get(train_dict, b"fine_labels", "fine_labels"), dtype=np.int64)
    y_test = np.array(_get(test_dict, b"fine_labels", "fine_labels"), dtype=np.int64)
    imgs_tr = _to_images(_get(train_dict, b"data", "data"))
    imgs_te = _to_images(_get(test_dict, b"data", "data"))

    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for label in range(NUM_CLASSES):
        tr = np.where(y_train == label)[0]
        te = np.where(y_test == label)[0]
        train_idx.extend(rng.choice(tr, TRAIN_PER_CLASS, replace=False).tolist())
        test_idx.extend(rng.choice(te, TEST_PER_CLASS, replace=False).tolist())

    imgs_tr = imgs_tr[train_idx]
    y_tr = y_train[train_idx]
    imgs_te = imgs_te[test_idx]
    y_te = y_test[test_idx]

    tr_perm = rng.permutation(len(y_tr))
    te_perm = rng.permutation(len(y_te))
    imgs_tr, y_tr = imgs_tr[tr_perm], y_tr[tr_perm]
    imgs_te, y_te = imgs_te[te_perm], y_te[te_perm]

    print(
        f"Images ready: train={imgs_tr.shape} test={imgs_te.shape} "
        f"subset={TRAIN_PER_CLASS}/class train, {TEST_PER_CLASS}/class test"
    )
    return imgs_tr, y_tr, imgs_te, y_te


def tokenize_to_device(imgs_tr, y_tr, imgs_te, y_te, patch_size: int):
    x_tr = _images_to_sequence(imgs_tr, patch_size)
    x_te = _images_to_sequence(imgs_te, patch_size)

    flat = x_tr.reshape(-1, x_tr.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    x_tr = (x_tr - mean) / (std + 1e-8)
    x_te = (x_te - mean) / (std + 1e-8)

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    x_te_t = torch.tensor(x_te, dtype=torch.float32, device=DEVICE)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=DEVICE)

    meta = {
        "name": "cifar100",
        "patch_size": patch_size,
        "input_dim": int(x_tr_t.shape[-1]),
        "num_classes": NUM_CLASSES,
        "seq_len": int(x_tr_t.shape[1]),
        "n_train": int(x_tr_t.shape[0]),
        "n_test": int(x_te_t.shape[0]),
        "stress": stress_label(patch_size),
    }
    print(
        f"\n>>> Tokenize {meta['stress']}: train={tuple(x_tr_t.shape)} "
        f"test={tuple(x_te_t.shape)}"
    )
    return x_tr_t, y_tr_t, x_te_t, y_te_t, meta


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
    """Frozen FEN: bag | roll | hybrid. Deplete on; final concat read."""

    def __init__(self, input_dim, hidden_dim, num_classes, write_mode="bag", K=TAPE_K):
        super().__init__()
        assert write_mode in ("bag", "roll", "hybrid")
        self.hdim = hidden_dim
        self.write_mode = write_mode
        self.K = K
        self.is_hybrid = write_mode == "hybrid"

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        if write_mode in ("roll", "hybrid"):
            self.roll_gate = nn.Linear(hidden_dim, 1)
        else:
            self.roll_gate = None

        head_in = hidden_dim * 3 if self.is_hybrid else hidden_dim * 2
        self.head = _mlp_head(head_in, num_classes)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        E_roll = x.new_zeros(B, self.hdim) if self.is_hybrid else None
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
            else:
                E = E + v
                gamma = torch.sigmoid(self.roll_gate(f))
                E_roll = (
                    (1.0 - gamma) * E_roll
                    + gamma * torch.roll(E_roll, shifts=1, dims=-1)
                    + v
                )
            g_acc = g_acc + g.detach().mean()

        if self.is_hybrid:
            arch = torch.cat([E, E_roll], dim=-1)
            esc_norm = (
                E.detach().norm(dim=-1).mean() + E_roll.detach().norm(dim=-1).mean()
            )
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
    "fen_roll": dict(kind="fen", write_mode="roll"),
    "fen_hybrid": dict(kind="fen", write_mode="hybrid"),
    "lstm": dict(kind="lstm", num_layers=1),
}


def build(name, input_dim, num_classes, hidden_dim):
    spec = MODEL_SPECS[name]
    if spec["kind"] == "residual":
        return ResidualRNN(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "lstm":
        return LSTMBaseline(
            input_dim, hidden_dim, num_classes, num_layers=spec.get("num_layers", 1)
        )
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
        f"T={T}  C={in_dim}  batch={batch_size}"
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
                f"pipe={val['pipe']:.2f}  gate={val['gate']:.3f}  "
                f"[{time.time() - ep_t0:.1f}s]"
            )

    elapsed = time.time() - t0
    ep1 = history[0]
    ep2 = history[1] if len(history) > 1 else float("nan")
    print(
        f"  >> best acc={best_acc:.3f}  @ep{best_ep}  t={elapsed:.1f}s  "
        f"ep1={ep1:.3f}  ep2={ep2:.3f}  pipe={best_snap['pipe']:.2f}  "
        f"graph={'yes' if graph else 'no'}"
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
        "ep1": ep1,
        "ep2": ep2,
        "last": history[-1],
        "patch": meta["patch_size"],
        "T": meta["seq_len"],
        "C": meta["input_dim"],
    }


def print_level_summary(patch, by_model, meta):
    print("\n" + "-" * 88)
    print(
        f"SUMMARY  patch={patch}  {meta['stress']}  seeds={SEEDS}  "
        f"epochs={EPOCHS}  target_params≈{TARGET_PARAMS}"
    )
    print("-" * 88)
    print(
        f"{'model':<14} {'acc':>7} {'ep1':>6} {'ep2':>6} {'last':>6} "
        f"{'to_best':>8} {'pipe':>7} {'params':>8}"
    )
    rows_out = {}
    for name in MODEL_ORDER:
        rows = by_model[name]
        acc = np.mean([r["acc"] for r in rows])
        ep1 = np.mean([r["ep1"] for r in rows])
        ep2 = np.mean([r["ep2"] for r in rows])
        last = np.mean([r["last"] for r in rows])
        epb = np.mean([r["best_ep"] for r in rows])
        pipe = np.mean([r["pipe"] for r in rows])
        params = rows[0]["params"]
        rows_out[name] = {
            "acc": acc,
            "ep1": ep1,
            "ep2": ep2,
            "last": last,
            "pipe": pipe,
            "params": params,
        }
        print(
            f"{name:<14} {acc:7.3f} {ep1:6.3f} {ep2:6.3f} {last:6.3f} "
            f"{epb:8.1f} {pipe:7.2f} {params:8d}"
        )
    print("-" * 88)
    if "fen_roll" in rows_out and "fen_bag" in rows_out:
        r, b = rows_out["fen_roll"], rows_out["fen_bag"]
        print(
            f"Gaps: roll−bag peak={r['acc'] - b['acc']:+.3f}  "
            f"ep1={r['ep1'] - b['ep1']:+.3f}  ep2={r['ep2'] - b['ep2']:+.3f}"
        )
    if "fen_roll" in rows_out and "lstm" in rows_out:
        r, l = rows_out["fen_roll"], rows_out["lstm"]
        print(
            f"Gaps: roll−lstm peak={r['acc'] - l['acc']:+.3f}  "
            f"ep1={r['ep1'] - l['ep1']:+.3f}"
        )
    if "fen_hybrid" in rows_out and "fen_roll" in rows_out:
        h, r = rows_out["fen_hybrid"], rows_out["fen_roll"]
        print(
            f"Gaps: hybrid−roll peak={h['acc'] - r['acc']:+.3f}  "
            f"ep1={h['ep1'] - r['ep1']:+.3f}"
        )
    return rows_out


def print_regime_map(curve):
    """curve: list of (patch, T, C, rows_out dict)."""
    print("\n" + "=" * 88)
    print("REGIME MAP — sequential stress vs architecture gaps (paste this back)")
    print("=" * 88)
    print(
        f"{'patch':>5} {'T':>5} {'C':>4}  "
        f"{'roll':>6} {'bag':>6} {'hyb':>6} {'lstm':>6} {'res':>6}  "
        f"{'r-b_pk':>7} {'r-b_e1':>7} {'r-b_e2':>7} {'r-l_pk':>7} {'h-r_e1':>7}"
    )
    for patch, T, C, rows in curve:
        def g(name, key, default=float("nan")):
            return rows[name][key] if name in rows else default

        rb_pk = g("fen_roll", "acc") - g("fen_bag", "acc")
        rb_e1 = g("fen_roll", "ep1") - g("fen_bag", "ep1")
        rb_e2 = g("fen_roll", "ep2") - g("fen_bag", "ep2")
        rl_pk = g("fen_roll", "acc") - g("lstm", "acc")
        hr_e1 = g("fen_hybrid", "ep1") - g("fen_roll", "ep1")
        print(
            f"{patch:5d} {T:5d} {C:4d}  "
            f"{g('fen_roll', 'acc'):6.3f} {g('fen_bag', 'acc'):6.3f} "
            f"{g('fen_hybrid', 'acc'):6.3f} {g('lstm', 'acc'):6.3f} "
            f"{g('residual', 'acc'):6.3f}  "
            f"{rb_pk:+7.3f} {rb_e1:+7.3f} {rb_e2:+7.3f} {rl_pk:+7.3f} {hr_e1:+7.3f}"
        )
    print("=" * 88)
    print(
        "How to read:\n"
        "  • Success: r-b_pk and/or r-b_e1 grow as T grows (P8→P4→P2)\n"
        "  • exp10 anchors: P4 r-b_pk≈+0.02 | P2 r-b_pk≈+0.12\n"
        "  • P8 should look even more compressed (or bag competitive)\n"
        "  • hybrid−roll ep1 usually negative (bag half drags early)\n"
        "  • residual pipe pathology shows in low acc, not in this gap table\n"
        "  • This maps WHEN fen_roll default applies — not absolute CIFAR SOTA"
    )
    print("DONE — paste REGIME MAP + per-level SUMMARYs back for scoring.")


def main():
    for p in PATCH_SIZES:
        if 32 % p != 0:
            raise ValueError(f"PATCH_SIZES entry {p} does not divide 32")

    print(
        f"TARGET_PARAMS≈{TARGET_PARAMS} | EPOCHS={EPOCHS} | SEEDS={SEEDS}\n"
        f"Stress ladder (low→high): "
        + " | ".join(stress_label(p) for p in PATCH_SIZES)
    )

    imgs_tr, y_tr, imgs_te, y_te = load_cifar100_images()
    curve = []

    for patch in PATCH_SIZES:
        X_tr, ytr, X_te, yte, meta = tokenize_to_device(
            imgs_tr, y_tr, imgs_te, y_te, patch
        )
        bs = batch_size_for_T(meta["seq_len"])
        n_tr = (X_tr.shape[0] // bs) * bs
        n_te = (X_te.shape[0] // bs) * bs
        if n_tr < X_tr.shape[0] or n_te < X_te.shape[0]:
            print(f"  truncating to full batches: train {n_tr} test {n_te} (bs={bs})")
            X_tr, ytr = X_tr[:n_tr], ytr[:n_tr]
            X_te, yte = X_te[:n_te], yte[:n_te]

        by_model = defaultdict(list)
        for seed in SEEDS:
            print(f"\n### patch={patch}  seed={seed}  {meta['stress']}")
            for name in MODEL_ORDER:
                row = train_one(
                    name, X_tr, ytr, X_te, yte, meta, seed, EPOCHS, bs
                )
                by_model[name].append(row)

        rows_out = print_level_summary(patch, by_model, meta)
        curve.append((patch, meta["seq_len"], meta["input_dim"], rows_out))

        # free GPU tensors between stress levels
        del X_tr, ytr, X_te, yte
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print_regime_map(curve)
    return curve


if __name__ == "__main__":
    main()
