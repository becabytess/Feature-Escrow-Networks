# ==============================================================================
# fen_lab / EXP10 — Sequential CIFAR-100 (hard transfer after sMNIST / pMNIST)
# ==============================================================================
# Why this task
#   Foundation + sMNIST + pMNIST froze the story:
#     fen_roll = consistent default for long ordered classification
#     early accuracy (ep1–ep2) = first-class ranking signal
#     hybrid = raster peak specialist, weaker under non-raster order
#   Digit streams are still sparse/simple. CIFAR-100 is richer (color, 100 classes).
#
# Question
#   Do frozen fen_lab operators — especially fen_roll + final concat read —
#   still win on peak AND early accuracy when the sequence is harder than MNIST?
#
# Protocol
#   CIFAR-100 fine labels (100 classes)
#   INPUT_MODE:
#     "patch" (primary) — 4×4 non-overlap patches → T=64, C=48
#     "pixel" (optional) — raw raster → T=1024, C=3  (slow; legacy ~14% note)
#   Subset: 150 train / 20 test per class (15k / 2k), same spirit as sMNIST
#   ~100k params, AdamW, GPU preload, CUDA graphs
#   Report peak + ep1 + ep2 + pipe (same as exp08/09)
#
# Models (lean set — transfer of frozen story, not a new variant invent-a-thon)
#   residual | fen_bag | fen_copy | fen_roll | fen_hybrid | lstm | lstm_3L
#
# Chance floor = 0.01. History spatial-CNN CIFAR is NOT this experiment.
# Old sequential pixel note (~14%): not fen_lab evidence; re-measure under freeze.
#
# Kaggle/Colab: paste whole file → GPU → Run.
# Deps: torch, numpy. pickle in stdlib. Optional kagglehub for data fetch.
# Data: Kaggle fedesoriano/cifar100 (or any CIFAR-100 Python train/test pickles).
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

# "patch" = primary stress test after sMNIST/pMNIST
# "pixel" = long raw stream (T=1024); set only when you want the legacy protocol
INPUT_MODE = "patch"  # "patch" | "pixel"
PATCH_SIZE = 4  # only for patch → T = (32/P)^2, C = 3*P*P

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

# pixel T=1024 is heavy; patch T=64 can use larger batches
BATCH_SIZE = 128 if INPUT_MODE == "pixel" else 256
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128
TAPE_K = 8
EVENT_GATE_THRESH = 0.25
MIN_H, MAX_H = 16, 256
AUTO_MATCH_PARAMS = True

TRAIN_PER_CLASS = 150
TEST_PER_CLASS = 20
NUM_CLASSES = 100

USE_CUDA_GRAPHS = True
CUDA_GRAPH_WARMUP_STEPS = 3

MODEL_ORDER = [
    "residual",
    "fen_bag",
    "fen_copy",
    "fen_roll",
    "fen_hybrid",
    "lstm",
    "lstm_3L",
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
print(
    "EXP10 — Sequential CIFAR-100: does fen_roll transfer beyond digit streams?\n"
    "  Compare peak + ep1/ep2 (same ranking signal as exp08/09).\n"
    f"  Chance floor = 1/{NUM_CLASSES} = {1.0 / NUM_CLASSES:.3f}"
)
print(f"Models: {MODEL_ORDER}")


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
    """Locate train/test pickles under /kaggle/input, kagglehub, or local dirs."""
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
    """(N, 3072) → (N, 3, 32, 32) float32 in [0,1]."""
    x = raw.astype(np.float32) / 255.0
    return x.reshape(-1, 3, 32, 32)


def _images_to_sequence(imgs: np.ndarray, mode: str, patch: int) -> np.ndarray:
    """
    imgs: (N, 3, 32, 32)
    pixel → (N, 1024, 3)
    patch → (N, (32/P)^2, 3*P*P)
    """
    if mode == "pixel":
        return imgs.reshape(imgs.shape[0], 3, 1024).transpose(0, 2, 1).copy()

    if mode != "patch":
        raise ValueError(mode)
    if 32 % patch != 0:
        raise ValueError(f"PATCH_SIZE={patch} must divide 32")

    n, c, h, w = imgs.shape
    ph = pw = patch
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

    # shuffle within train/test (keep class balance via construction)
    tr_perm = rng.permutation(len(y_tr))
    te_perm = rng.permutation(len(y_te))
    imgs_tr, y_tr = imgs_tr[tr_perm], y_tr[tr_perm]
    imgs_te, y_te = imgs_te[te_perm], y_te[te_perm]

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
    Frozen FEN operators (same as exp08/09).
    write_mode: bag | copy | roll | hybrid
      copy = bag write, no deplete (h = f)
    Always: final head([h, arch]); no every-step reinject.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, write_mode="bag", K=TAPE_K):
        super().__init__()
        assert write_mode in ("bag", "copy", "roll", "hybrid")
        self.hdim = hidden_dim
        self.write_mode = write_mode
        self.K = K
        self.is_hybrid = write_mode == "hybrid"
        self.no_deplete = write_mode == "copy"

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        if write_mode in ("roll", "hybrid"):
            self.roll_gate = nn.Linear(hidden_dim, 1)
        else:
            self.roll_gate = None

        if self.is_hybrid:
            head_in = hidden_dim * 3  # h + E_bag + E_roll
        else:
            head_in = hidden_dim * 2  # h + E

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
            h = f if self.no_deplete else (f - D)

            if self.write_mode in ("bag", "copy"):
                E = E + v
            elif self.write_mode == "roll":
                gamma = torch.sigmoid(self.roll_gate(f))
                E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v
            else:  # hybrid
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
    "fen_copy": dict(kind="fen", write_mode="copy"),
    "fen_roll": dict(kind="fen", write_mode="roll"),
    "fen_hybrid": dict(kind="fen", write_mode="hybrid"),
    "lstm": dict(kind="lstm", num_layers=1),
    "lstm_3L": dict(kind="lstm", num_layers=3),
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
    }


def main():
    print(
        f"TARGET_PARAMS≈{TARGET_PARAMS} | EPOCHS={EPOCHS} | SEEDS={SEEDS} | "
        f"INPUT_MODE={INPUT_MODE} | PATCH_SIZE={PATCH_SIZE if INPUT_MODE == 'patch' else 'n/a'}\n"
        "Context from exp08/09 (digit streams):\n"
        "  sMNIST: roll≈0.88 ep1≈0.64 | hybrid≈0.91 ep1≈0.67 | bag≈0.66 | lstm1L≈0.11\n"
        "  pMNIST: roll≈0.88 ep1≈0.60 | hybrid≈0.84 ep1≈0.33 | bag≈0.40\n"
        "Success on CIFAR: roll still leads early + peak vs bag/LSTM; pipe stays lean.\n"
        "Failure (still useful): early roll edge dies → story is digit-stream limited."
    )

    X_tr, y_tr, X_te, y_te, meta = make_cifar100_seq()
    bs = BATCH_SIZE
    n_tr = (X_tr.shape[0] // bs) * bs
    n_te = (X_te.shape[0] // bs) * bs
    if n_tr < X_tr.shape[0] or n_te < X_te.shape[0]:
        print(f"  truncating to full batches: train {n_tr} test {n_te}")
        X_tr, y_tr = X_tr[:n_tr], y_tr[:n_tr]
        X_te, y_te = X_te[:n_te], y_te[:n_te]

    by_model = defaultdict(list)
    for seed in SEEDS:
        print(f"\n### seed={seed}")
        for name in MODEL_ORDER:
            row = train_one(name, X_tr, y_tr, X_te, y_te, meta, seed, EPOCHS, bs)
            by_model[name].append(row)

    print("\n" + "-" * 88)
    print(
        f"SUMMARY  cifar100  mode={meta['input_mode']}  T={meta['seq_len']}  "
        f"C={meta['input_dim']}  seeds={SEEDS}  target_params≈{TARGET_PARAMS}  "
        f"epochs={EPOCHS}"
    )
    print("-" * 88)
    print(
        f"{'model':<14} {'acc':>7} {'ep1':>6} {'ep2':>6} {'last':>6} "
        f"{'to_best':>8} {'pipe':>7} {'params':>8}"
    )
    for name in MODEL_ORDER:
        rows = by_model[name]
        acc = np.mean([r["acc"] for r in rows])
        ep1 = np.mean([r["ep1"] for r in rows])
        ep2 = np.mean([r["ep2"] for r in rows])
        last = np.mean([r["last"] for r in rows])
        epb = np.mean([r["best_ep"] for r in rows])
        pipe = np.mean([r["pipe"] for r in rows])
        params = rows[0]["params"]
        print(
            f"{name:<14} {acc:7.3f} {ep1:6.3f} {ep2:6.3f} {last:6.3f} "
            f"{epb:8.1f} {pipe:7.2f} {params:8d}"
        )
    print("-" * 88)
    print(f"Chance floor = {1.0 / NUM_CLASSES:.3f}")

    if "fen_roll" in by_model and "fen_bag" in by_model:
        r0, b0 = by_model["fen_roll"][0], by_model["fen_bag"][0]
        print(
            f"Gaps: roll−bag peak={r0['acc'] - b0['acc']:+.3f}  "
            f"ep1={r0['ep1'] - b0['ep1']:+.3f}  ep2={r0['ep2'] - b0['ep2']:+.3f}"
        )
    if "fen_roll" in by_model and "lstm" in by_model:
        r0, l0 = by_model["fen_roll"][0], by_model["lstm"][0]
        print(
            f"Gaps: roll−lstm1L peak={r0['acc'] - l0['acc']:+.3f}  "
            f"ep1={r0['ep1'] - l0['ep1']:+.3f}"
        )
    if "fen_hybrid" in by_model and "fen_roll" in by_model:
        h0, r0 = by_model["fen_hybrid"][0], by_model["fen_roll"][0]
        print(
            f"Gaps: hybrid−roll peak={h0['acc'] - r0['acc']:+.3f}  "
            f"ep1={h0['ep1'] - r0['ep1']:+.3f}"
        )

    print(
        "How to read:\n"
        "  • Transfer OK: roll leads bag/LSTM on peak and especially ep1–ep2\n"
        "  • Transfer weak: all models near chance, or residual/LSTM match roll\n"
        "  • Hybrid vs roll: if hybrid only wins peak but loses early → same as sMNIST\n"
        "  • copy vs bag: if copy > bag, deplete still task-dependent (as on sMNIST)\n"
        "  • This is sequential RNN/FEN — not a 2D CNN CIFAR leaderboard"
    )
    print("DONE — paste this SUMMARY (+ ep1/ep2) back for scoring.")
    return by_model


if __name__ == "__main__":
    main()
