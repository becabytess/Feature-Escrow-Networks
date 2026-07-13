# ==============================================================================
# fen_lab / EXP12 — Deplete law (write × deplete grid)
# ==============================================================================
# Step 2 after regime map (exp11). No scaling. No new write invent-a-thon.
#
# Question
#   When is subtractive depletion (h ← f − D) load-bearing vs optional?
#   Prior fragments (different runs):
#     dual-role foundation: bag+deplete solves; residual fails ID
#     sMNIST: fen_copy (bag, no deplete) beat fen_bag
#     CIFAR-P4: bag > copy; CIFAR-P2: both weak under long scan
#   This run puts bag/roll × deplete on/off on TWO task classes in one protocol.
#
# Tasks
#   A) distracted — static ID + noisy count (dual-role). ~15k params, T=96.
#      Metric: joint 30-way acc + id_acc + count_acc + pipe
#   B) smnist    — long ordered pixel scan. ~100k params, T=400 (20×20).
#      Metric: peak acc + ep1 + ep2 + pipe
#
# Models (2×2 only)
#   bag_dep     bag write + deplete      (classic fen_bag)
#   bag_nodep   bag write + NO deplete   (fen_copy)
#   roll_dep    roll write + deplete     (classic fen_roll)
#   roll_nodep  roll write + NO deplete
#
# Always: gate → D → write E; final head([h, E]); no reinject.
#
# How to read
#   deplete helps if dep − nodep > 0 on peak and/or early, and pipe lower.
#   Possible rule to test:
#     deplete when pipe must stay dual-role-clean (static fact + ongoing work)
#     optional when head mostly reads E and pipe is a scanner
#
# Kaggle/Colab: paste whole file → GPU → Run.
# Deps: torch, numpy; optional pandas/kagglehub/torchvision for MNIST.
# ==============================================================================

import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ CONFIG ----------------------------------------
FAST_MODE = True

# Which tasks to run (both recommended for the law)
TASKS = ["distracted", "smnist"]  # drop one for a faster partial run

if FAST_MODE:
    SEEDS = [1]
    EPOCHS_DIST = 12
    EPOCHS_SMNIST = 15
    DIST_TRAIN_N, DIST_TEST_N = 4000, 1000
    PRINT_EVERY = 1
else:
    SEEDS = [1, 2, 3]
    EPOCHS_DIST = 20
    EPOCHS_SMNIST = 20
    DIST_TRAIN_N, DIST_TEST_N = 8000, 2000
    PRINT_EVERY = 1

# distracted (foundation-scale)
DIST_SEQ_LEN = 96
DIST_NOISE_STD = 0.45
DIST_TARGET_PARAMS = 15000
DIST_BATCH = 128
DIST_LR = 1e-3
DIST_WD = 0.0

# smnist (hard-bench scale)
IMG_SIZE = 20
SMNIST_SEQ_LEN = IMG_SIZE * IMG_SIZE
SMNIST_TRAIN_PER = 1500
SMNIST_TEST_PER = 200
SMNIST_TARGET_PARAMS = 100000
SMNIST_BATCH = 128
SMNIST_LR = 1e-3
SMNIST_WD = 1e-4

GRAD_CLIP = 1.0
HEAD_WIDTH = 128
MIN_H, MAX_H = 8, 256
AUTO_MATCH_PARAMS = True

USE_CUDA_GRAPHS = True
CUDA_GRAPH_WARMUP_STEPS = 3

MODEL_ORDER = [
    "bag_dep",
    "bag_nodep",
    "roll_dep",
    "roll_nodep",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    USE_CUDA_GRAPHS = False

print(
    f"Device: {DEVICE} | EXP12 deplete law | FAST_MODE={FAST_MODE} | "
    f"TASKS={TASKS} | CUDA_GRAPHS={USE_CUDA_GRAPHS}"
)
print(
    "EXP12 — When is deplete (h ← f−D) load-bearing?\n"
    "  2×2: bag/roll × deplete on/off | dual-role (distracted) + long scan (sMNIST)\n"
    f"  Models: {MODEL_ORDER}"
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


# ------------------------------ DATA: distracted ------------------------------
def make_distracted(n, seed, seq_len=DIST_SEQ_LEN, noise_std=DIST_NOISE_STD):
    """Static ID at t=0 + active +/- counting with distractors. 30-way class."""
    rng = np.random.default_rng(seed)
    n_id, n_bins = 10, 3
    op_dims, noise_dims = 4, 16
    input_dim = n_id + op_dims + noise_dims

    X = rng.normal(0.0, noise_std, size=(n, seq_len, input_dim)).astype(np.float32)
    X[:, :, : n_id + op_dims] *= 0.10
    y = np.zeros((n,), dtype=np.int64)

    plus_dim, minus_dim = n_id, n_id + 1
    distract_a, distract_b = n_id + 2, n_id + 3

    for i in range(n):
        static_id = int(rng.integers(0, n_id))
        count_bin = int(rng.integers(0, n_bins))
        X[i, 0, static_id] += 2.0

        possible = np.arange(1, seq_len)
        n_events = int(rng.integers(18, 31))
        positions = rng.choice(possible, size=n_events, replace=False)
        p_plus = [0.25, 0.50, 0.75][count_bin]
        n_plus = int(round(n_events * p_plus))
        X[i, positions[:n_plus], plus_dim] += 1.5
        X[i, positions[n_plus:], minus_dim] += 1.5

        n_distract = int(rng.integers(10, 25))
        dpos = rng.choice(possible, size=n_distract, replace=False)
        half = n_distract // 2
        X[i, dpos[:half], distract_a] += 1.25
        X[i, dpos[half:], distract_b] += 1.25
        y[i] = static_id * n_bins + count_bin

    meta = {
        "task": "distracted",
        "input_dim": input_dim,
        "num_classes": n_id * n_bins,
        "n_id": n_id,
        "n_bins": n_bins,
        "seq_len": seq_len,
        "target_params": DIST_TARGET_PARAMS,
        "batch_size": DIST_BATCH,
        "lr": DIST_LR,
        "weight_decay": DIST_WD,
        "epochs": EPOCHS_DIST,
    }
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        meta,
    )


def load_distracted():
    Xtr, ytr, meta = make_distracted(DIST_TRAIN_N, seed=1000)
    Xte, yte, _ = make_distracted(DIST_TEST_N, seed=2000)
    Xtr, ytr = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xte, yte = Xte.to(DEVICE), yte.to(DEVICE)
    print(
        f"distracted ready: train={tuple(Xtr.shape)} test={tuple(Xte.shape)} "
        f"classes={meta['num_classes']} T={meta['seq_len']} "
        f"target_params≈{meta['target_params']}"
    )
    return Xtr, ytr, Xte, yte, meta


# ------------------------------ DATA: sMNIST ----------------------------------
def _load_mnist_arrays():
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        print("Searching /kaggle/input for mnist_*.csv ...")
        for root, _dirs, files in os.walk(kaggle_root):
            lower_map = {f.lower(): os.path.join(root, f) for f in files}
            if "mnist_train.csv" in lower_map and "mnist_test.csv" in lower_map:
                import pandas as pd

                train_csv = lower_map["mnist_train.csv"]
                test_csv = lower_map["mnist_test.csv"]
                print(f"Loading Kaggle CSVs:\n  {train_csv}\n  {test_csv}")
                tr = pd.read_csv(train_csv)
                te = pd.read_csv(test_csv)
                lab = "label" if "label" in tr.columns else tr.columns[0]
                y_tr = tr[lab].values.astype(np.int64)
                y_te = te[lab].values.astype(np.int64)
                x_tr = tr.drop(columns=[lab]).values.astype(np.float32) / 255.0
                x_te = te.drop(columns=[lab]).values.astype(np.float32) / 255.0
                return x_tr, y_tr, x_te, y_te

    try:
        import kagglehub
        import pandas as pd

        print("Downloading MNIST via kagglehub (oddrationale/mnist-in-csv)...")
        path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
        tr = pd.read_csv(os.path.join(path, "mnist_train.csv"))
        te = pd.read_csv(os.path.join(path, "mnist_test.csv"))
        y_tr = tr["label"].values.astype(np.int64)
        y_te = te["label"].values.astype(np.int64)
        x_tr = tr.drop(columns=["label"]).values.astype(np.float32) / 255.0
        x_te = te.drop(columns=["label"]).values.astype(np.float32) / 255.0
        return x_tr, y_tr, x_te, y_te
    except Exception as e:
        print(f"  kagglehub path failed ({type(e).__name__}: {e})")

    try:
        from torchvision import datasets

        print("Loading MNIST via torchvision → ./data ...")
        tr = datasets.MNIST(root="./data", train=True, download=True)
        te = datasets.MNIST(root="./data", train=False, download=True)
        x_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_tr = tr.targets.numpy().astype(np.int64)
        x_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_te = te.targets.numpy().astype(np.int64)
        return x_tr, y_tr, x_te, y_te
    except Exception as e:
        raise FileNotFoundError(
            "Could not load MNIST. On Kaggle: add mnist-in-csv. "
            f"Last error: {e}"
        ) from e


def load_smnist():
    x_tr, y_tr, x_te, y_te = _load_mnist_arrays()
    x_tr_t = torch.tensor(x_tr).view(-1, 1, 28, 28)
    x_te_t = torch.tensor(x_te).view(-1, 1, 28, 28)
    print(f"Downsampling 28×28 → {IMG_SIZE}×{IMG_SIZE} (T={SMNIST_SEQ_LEN})...")
    x_tr_t = F.interpolate(
        x_tr_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    x_te_t = F.interpolate(
        x_te_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    x_tr_seq = x_tr_t.squeeze(1).reshape(-1, SMNIST_SEQ_LEN, 1).numpy()
    x_te_seq = x_te_t.squeeze(1).reshape(-1, SMNIST_SEQ_LEN, 1).numpy()

    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for d in range(10):
        train_idx.extend(
            rng.choice(np.where(y_tr == d)[0], SMNIST_TRAIN_PER, replace=False).tolist()
        )
        test_idx.extend(
            rng.choice(np.where(y_te == d)[0], SMNIST_TEST_PER, replace=False).tolist()
        )
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    x_tr = x_tr_seq[train_idx]
    y_tr = y_tr[train_idx]
    x_te = x_te_seq[test_idx]
    y_te = y_te[test_idx]

    mean, std = x_tr.mean(), x_tr.std()
    x_tr = (x_tr - mean) / (std + 1e-8)
    x_te = (x_te - mean) / (std + 1e-8)

    Xtr = torch.tensor(x_tr, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    Xte = torch.tensor(x_te, dtype=torch.float32, device=DEVICE)
    yte = torch.tensor(y_te, dtype=torch.long, device=DEVICE)

    meta = {
        "task": "smnist",
        "input_dim": 1,
        "num_classes": 10,
        "seq_len": SMNIST_SEQ_LEN,
        "target_params": SMNIST_TARGET_PARAMS,
        "batch_size": SMNIST_BATCH,
        "lr": SMNIST_LR,
        "weight_decay": SMNIST_WD,
        "epochs": EPOCHS_SMNIST,
        "n_bins": None,
    }
    print(
        f"sMNIST ready: train={tuple(Xtr.shape)} test={tuple(Xte.shape)} "
        f"T={meta['seq_len']} target_params≈{meta['target_params']}"
    )
    return Xtr, ytr, Xte, yte, meta


# ------------------------------ MODELS ----------------------------------------
def _mlp_head(in_dim, out_dim, width=None):
    if width is None:
        width = min(HEAD_WIDTH, max(32, out_dim * 2))
    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.ReLU(),
        nn.Linear(width, out_dim),
    )


class SeqFEN(nn.Module):
    """
    write_mode: bag | roll
    deplete: if True, h = f - D; else h = f (copy-style)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        write_mode="bag",
        deplete=True,
        head_width=HEAD_WIDTH,
    ):
        super().__init__()
        assert write_mode in ("bag", "roll")
        self.hdim = hidden_dim
        self.write_mode = write_mode
        self.deplete = deplete

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.roll_gate = nn.Linear(hidden_dim, 1) if write_mode == "roll" else None
        self.head = _mlp_head(hidden_dim * 2, num_classes, width=head_width)

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
            h = (f - D) if self.deplete else f

            if self.write_mode == "bag":
                E = E + v
            else:
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


MODEL_SPECS = {
    "bag_dep": dict(write_mode="bag", deplete=True),
    "bag_nodep": dict(write_mode="bag", deplete=False),
    "roll_dep": dict(write_mode="roll", deplete=True),
    "roll_nodep": dict(write_mode="roll", deplete=False),
}


def build(name, input_dim, num_classes, hidden_dim, head_width=HEAD_WIDTH):
    spec = MODEL_SPECS[name]
    return SeqFEN(
        input_dim,
        hidden_dim,
        num_classes,
        write_mode=spec["write_mode"],
        deplete=spec["deplete"],
        head_width=head_width,
    )


_HIDDEN_CACHE = {}


def choose_hidden(name, input_dim, num_classes, target_params, head_width):
    key = (name, input_dim, num_classes, target_params, head_width)
    if key in _HIDDEN_CACHE:
        return _HIDDEN_CACHE[key]
    if not AUTO_MATCH_PARAMS:
        _HIDDEN_CACHE[key] = 48
        return 48
    lo, hi = MIN_H, min(MAX_H, 256)
    best_h, best_diff = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(build(name, input_dim, num_classes, mid, head_width))
        d = abs(n - target_params)
        if d < best_diff:
            best_h, best_diff = mid, d
        if n < target_params:
            lo = mid + 1
        elif n > target_params:
            hi = mid - 1
        else:
            break
    for h in range(max(MIN_H, best_h - 4), min(MAX_H, best_h + 4) + 1):
        n = count_params(build(name, input_dim, num_classes, h, head_width))
        d = abs(n - target_params)
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
def evaluate(model, X, y, batch_size, meta):
    model.eval()
    total_correct = 0
    total_n = 0
    id_correct = count_correct = 0
    pipe_sum = gate_sum = esc_sum = 0.0
    n_batches = 0
    n_bins = meta.get("n_bins")

    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits, st = model(xb, return_stats=True)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == yb).sum().item()
        total_n += yb.numel()
        if n_bins is not None:
            id_correct += (pred // n_bins == yb // n_bins).sum().item()
            count_correct += (pred % n_bins == yb % n_bins).sum().item()
        pipe_sum += float(st["pipe_norm"].item())
        g, e = st["gate"], st["escrow_norm"]
        if torch.isfinite(g):
            gate_sum += float(g.item())
        if torch.isfinite(e):
            esc_sum += float(e.item())
        n_batches += 1

    out = {
        "acc": total_correct / max(total_n, 1),
        "pipe": pipe_sum / max(n_batches, 1),
        "gate": gate_sum / max(n_batches, 1) if n_batches else float("nan"),
        "escrow": esc_sum / max(n_batches, 1) if n_batches else float("nan"),
    }
    if n_bins is not None:
        out["id_acc"] = id_correct / max(total_n, 1)
        out["count_acc"] = count_correct / max(total_n, 1)
    return out


def train_one(name, X_train, y_train, X_test, y_test, meta, seed):
    seed_everything(seed)
    in_dim = meta["input_dim"]
    n_cls = meta["num_classes"]
    T = meta["seq_len"]
    epochs = meta["epochs"]
    batch_size = meta["batch_size"]
    target_params = meta["target_params"]
    # smaller head on tiny foundation nets
    head_width = 64 if meta["task"] == "distracted" else HEAD_WIDTH

    h = choose_hidden(name, in_dim, n_cls, target_params, head_width)
    model = build(name, in_dim, n_cls, h, head_width).to(DEVICE)
    n_params = count_params(model)
    spec = MODEL_SPECS[name]

    print(f"\n--- {meta['task']} | {name} (write={spec['write_mode']}, "
          f"deplete={spec['deplete']}) ---")
    print(
        f"  hidden={h}  params={n_params}  epochs={epochs}  seed={seed}  "
        f"T={T}  batch={batch_size}"
    )

    capturable = DEVICE.type == "cuda"
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=meta["lr"],
        weight_decay=meta["weight_decay"],
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
            clean_p, clean_o = _snapshot_state(model, opt)
            graph = _try_build_cuda_graph(model, opt, criterion, static_x, static_y)
            _restore_state(model, opt, clean_p, clean_o)
            print("  [CUDA graph capture OK]")
        except Exception as e:
            print(f"  [CUDA graph failed ({type(e).__name__}: {e}); eager]")
            graph = None
            seed_everything(seed)
            model = build(name, in_dim, n_cls, h, head_width).to(DEVICE)
            opt = torch.optim.AdamW(
                model.parameters(), lr=meta["lr"], weight_decay=meta["weight_decay"]
            )

    best_acc, best_ep, best_snap = -1.0, 0, None
    history = []
    id_hist, count_hist = [], []
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

        val = evaluate(model, X_test, y_test, batch_size, meta)
        history.append(val["acc"])
        if "id_acc" in val:
            id_hist.append(val["id_acc"])
            count_hist.append(val["count_acc"])
        if val["acc"] > best_acc:
            best_acc, best_ep, best_snap = val["acc"], ep, dict(val)

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            extra = ""
            if "id_acc" in val:
                extra = f"  id={val['id_acc']:.3f}  cnt={val['count_acc']:.3f}"
            print(
                f"    ep {ep:02d}/{epochs}  acc={val['acc']:.3f}{extra}  "
                f"pipe={val['pipe']:.2f}  gate={val['gate']:.3f}  "
                f"[{time.time() - ep_t0:.1f}s]"
            )

    elapsed = time.time() - t0
    ep1 = history[0]
    ep2 = history[1] if len(history) > 1 else float("nan")
    id_best = best_snap.get("id_acc", float("nan"))
    cnt_best = best_snap.get("count_acc", float("nan"))
    print(
        f"  >> best acc={best_acc:.3f}  @ep{best_ep}  t={elapsed:.1f}s  "
        f"ep1={ep1:.3f}  ep2={ep2:.3f}  pipe={best_snap['pipe']:.2f}"
        + (
            f"  id={id_best:.3f}  cnt={cnt_best:.3f}"
            if "id_acc" in best_snap
            else ""
        )
    )
    return {
        "name": name,
        "task": meta["task"],
        "write": spec["write_mode"],
        "deplete": spec["deplete"],
        "acc": best_acc,
        "best_ep": best_ep,
        "pipe": best_snap["pipe"],
        "gate": best_snap["gate"],
        "params": n_params,
        "hidden": h,
        "ep1": ep1,
        "ep2": ep2,
        "last": history[-1],
        "id_acc": id_best,
        "count_acc": cnt_best,
        "time": elapsed,
    }


def print_task_summary(task, by_model):
    print("\n" + "-" * 96)
    print(f"SUMMARY  task={task}  seeds={SEEDS}")
    print("-" * 96)
    if task == "distracted":
        print(
            f"{'model':<12} {'acc':>7} {'id':>6} {'cnt':>6} "
            f"{'ep1':>6} {'ep2':>6} {'pipe':>7} {'params':>8}"
        )
    else:
        print(
            f"{'model':<12} {'acc':>7} {'ep1':>6} {'ep2':>6} "
            f"{'last':>6} {'pipe':>7} {'params':>8}"
        )
    rows = {}
    for name in MODEL_ORDER:
        r = by_model[name][0]
        rows[name] = r
        if task == "distracted":
            print(
                f"{name:<12} {r['acc']:7.3f} {r['id_acc']:6.3f} {r['count_acc']:6.3f} "
                f"{r['ep1']:6.3f} {r['ep2']:6.3f} {r['pipe']:7.2f} {r['params']:8d}"
            )
        else:
            print(
                f"{name:<12} {r['acc']:7.3f} {r['ep1']:6.3f} {r['ep2']:6.3f} "
                f"{r['last']:6.3f} {r['pipe']:7.2f} {r['params']:8d}"
            )
    print("-" * 96)
    # deplete deltas within write mode
    for write in ("bag", "roll"):
        dep = f"{write}_dep"
        nodep = f"{write}_nodep"
        if dep in rows and nodep in rows:
            d, n = rows[dep], rows[nodep]
            print(
                f"Δ deplete ({write}): peak={d['acc'] - n['acc']:+.3f}  "
                f"ep1={d['ep1'] - n['ep1']:+.3f}  ep2={d['ep2'] - n['ep2']:+.3f}  "
                f"pipe={d['pipe'] - n['pipe']:+.2f}"
                + (
                    f"  id={d['id_acc'] - n['id_acc']:+.3f}"
                    if task == "distracted"
                    else ""
                )
            )
    return rows


def print_deplete_law(all_rows):
    """all_rows: dict task -> {model_name: row}"""
    print("\n" + "=" * 96)
    print("DEPLETE LAW TABLE — paste this back (dep − nodep; + means deplete helps)")
    print("=" * 96)
    print(
        f"{'task':<12} {'write':<6} {'Δpeak':>7} {'Δep1':>7} {'Δep2':>7} "
        f"{'Δpipe':>7} {'Δid':>7}  note"
    )
    for task in TASKS:
        if task not in all_rows:
            continue
        rows = all_rows[task]
        for write in ("bag", "roll"):
            dep, nodep = f"{write}_dep", f"{write}_nodep"
            if dep not in rows or nodep not in rows:
                continue
            d, n = rows[dep], rows[nodep]
            dpeak = d["acc"] - n["acc"]
            dep1 = d["ep1"] - n["ep1"]
            dep2 = d["ep2"] - n["ep2"]
            dpipe = d["pipe"] - n["pipe"]
            did = (
                d["id_acc"] - n["id_acc"]
                if task == "distracted"
                else float("nan")
            )
            if dpeak > 0.03 or (task == "distracted" and did > 0.05):
                note = "deplete HELPS"
            elif dpeak < -0.03:
                note = "deplete HURTS"
            else:
                note = "weak / optional"
            did_s = f"{did:+7.3f}" if task == "distracted" else f"{'n/a':>7}"
            print(
                f"{task:<12} {write:<6} {dpeak:+7.3f} {dep1:+7.3f} {dep2:+7.3f} "
                f"{dpipe:+7.2f} {did_s}  {note}"
            )
    print("=" * 96)
    print(
        "How to read:\n"
        "  • distracted id_acc is the dual-role stress test (not just joint acc)\n"
        "  • smnist: early (ep1/ep2) matters as much as peak\n"
        "  • Δpipe < 0 means deplete keeps pipe leaner (expected)\n"
        "  • If bag needs deplete on dual-role but not on sMNIST → task-dependent law\n"
        "  • If roll_nodep ≈ roll_dep on scans → deplete optional for ordered escrow scans"
    )
    print("DONE — paste DEPLETE LAW TABLE + SUMMARYs back for scoring.")


def main():
    print(
        f"distracted: epochs={EPOCHS_DIST} n={DIST_TRAIN_N}/{DIST_TEST_N} "
        f"params≈{DIST_TARGET_PARAMS}\n"
        f"smnist: epochs={EPOCHS_SMNIST} T={SMNIST_SEQ_LEN} "
        f"params≈{SMNIST_TARGET_PARAMS}"
    )

    loaders = {
        "distracted": load_distracted,
        "smnist": load_smnist,
    }
    all_rows = {}

    for task in TASKS:
        if task not in loaders:
            raise ValueError(task)
        print(f"\n{'#' * 72}\n# TASK: {task}\n{'#' * 72}")
        Xtr, ytr, Xte, yte, meta = loaders[task]()
        bs = meta["batch_size"]
        n_tr = (Xtr.shape[0] // bs) * bs
        n_te = (Xte.shape[0] // bs) * bs
        if n_tr < Xtr.shape[0] or n_te < Xte.shape[0]:
            print(f"  truncating to full batches: train {n_tr} test {n_te}")
            Xtr, ytr = Xtr[:n_tr], ytr[:n_tr]
            Xte, yte = Xte[:n_te], yte[:n_te]

        by_model = defaultdict(list)
        for seed in SEEDS:
            print(f"\n### task={task} seed={seed}")
            for name in MODEL_ORDER:
                row = train_one(name, Xtr, ytr, Xte, yte, meta, seed)
                by_model[name].append(row)
        all_rows[task] = print_task_summary(task, by_model)

        del Xtr, ytr, Xte, yte
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print_deplete_law(all_rows)
    return all_rows


if __name__ == "__main__":
    main()
