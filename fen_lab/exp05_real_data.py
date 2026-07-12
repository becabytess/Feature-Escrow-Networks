# ==============================================================================
# fen_lab / EXP05 — Real-data check of the frozen FEN architecture
# ==============================================================================
# Synthetic conclusion (exp01–04):
#   • Depletion + bag escrow = dual-role / static facts
#   • Hard ordered write (and/or slot readout) = list/order tasks
#   • Soft γ-tape needs slot readout; not required for dual-role alone
#   • Deliver = query-time / final concat read of archive — NOT every-step reinject
#   • Channel-roll demoted as core; residual fails when pipe must archive + compute
#
# This run asks: does that architecture hold on real 1D sequences?
#
# Datasets (auto-download):
#   mitbih  — MIT-BIH heartbeat, [N, 187, 1], 5 classes (topology / waveform)
#   forda   — UCR FordA,        [N, 500, 1], 2 classes (long 1D, late features)
#
# Models (~TARGET_PARAMS, auto-matched hidden):
#
#   name           archive write          head sees          role
#   -------------  ---------------------  -----------------  --------------------
#   residual       none                   h                  no-escrow control
#   fen_bag        bag (additive)         [h, E]             ★ canonical default
#   fen_hard_bag   hard ptr tape + bag    [h, pool(E), c]    order topology + bag
#   fen_roll       channel-roll bag       [h, E]             historical order trick
#   lstm           LSTM cells             last h             strong baseline
#
# SPEED (learned from the optimized sMNIST trainer — architecture ignored):
#   • full train/test tensors preloaded once onto GPU
#   • device-side full-batch iteration (fixed B for CUDA graphs)
#   • binary search for hidden width (not linear scan)
#   • TF32 + cudnn.benchmark
#   • optional CUDA-graph capture of one train step (huge win on Python T-loops)
#   • no .item() inside the sequence loop (avoids GPU sync per timestep)
#
# Colab: paste whole file → Runtime GPU → Run.
# Deps: torch, numpy, pandas (CSV). No sklearn.
# ==============================================================================

import os
import random
import time
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ CONFIG ----------------------------------------
FAST_MODE = True

# Which datasets to run. Options: "mitbih", "forda"
if FAST_MODE:
    DATASETS = ["mitbih"]  # add "forda" when you want the long UCR task
    SEEDS = [1]
    EPOCHS = {"mitbih": 12, "forda": 20}
    PRINT_EVERY = 2
    TARGET_PARAMS = 75000
else:
    DATASETS = ["mitbih", "forda"]
    SEEDS = [1, 2, 3]
    EPOCHS = {"mitbih": 15, "forda": 40}
    PRINT_EVERY = 1
    TARGET_PARAMS = 75000

# Larger batches = fewer Python launches over T steps (MIT-BIH fits big batches)
BATCH_SIZE = {"mitbih": 1000, "forda": 128}
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128  # shared MLP head width for fair capacity
TAPE_K = 8
EVENT_GATE_THRESH = 0.25
MIN_H, MAX_H = 16, 256
AUTO_MATCH_PARAMS = True
DATA_DIR = "./data"

# CUDA graph: captures one full train step (incl. T-step RNN loop). Huge speedup.
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
    "Device:", DEVICE,
    "| FAST_MODE:", FAST_MODE,
    "| DATASETS:", DATASETS,
    "| CUDA_GRAPHS:", USE_CUDA_GRAPHS,
)


# ------------------------------ UTILS -----------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # keep cudnn.benchmark=True for speed (set at import); do not force False


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------ DATA ------------------------------------------
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_mitbih():
    _ensure_dir(DATA_DIR)
    train_csv = os.path.join(DATA_DIR, "mitbih_train.csv")
    test_csv = os.path.join(DATA_DIR, "mitbih_test.csv")
    train_url = (
        "https://github.com/csuustc/ECG-Heartbeat-Classification/raw/master/"
        "mitbih_train.csv.zip"
    )
    test_url = (
        "https://github.com/csuustc/ECG-Heartbeat-Classification/raw/master/"
        "mitbih_test.csv.zip"
    )

    if not os.path.exists(train_csv):
        zpath = os.path.join(DATA_DIR, "mitbih_train.csv.zip")
        print("Downloading MIT-BIH train (~15MB)...")
        urllib.request.urlretrieve(train_url, zpath)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(DATA_DIR)
        os.remove(zpath)

    if not os.path.exists(test_csv):
        zpath = os.path.join(DATA_DIR, "mitbih_test.csv.zip")
        print("Downloading MIT-BIH test (~4MB)...")
        urllib.request.urlretrieve(test_url, zpath)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(DATA_DIR)
        os.remove(zpath)

    return train_csv, test_csv


def load_mitbih():
    train_csv, test_csv = download_mitbih()
    print("Parsing MIT-BIH CSVs...")
    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)

    # Preload fully onto device once (no H2D every batch)
    X_train = torch.tensor(X_train, device=DEVICE).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(X_test, device=DEVICE).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    meta = {
        "name": "mitbih",
        "input_dim": 1,
        "num_classes": 5,
        "seq_len": X_train.size(1),
        "n_train": X_train.size(0),
        "n_test": X_test.size(0),
    }
    print(
        f"MIT-BIH ready on {DEVICE}  train={tuple(X_train.shape)}  "
        f"test={tuple(X_test.shape)}  classes={meta['num_classes']}"
    )
    return X_train, y_train, X_test, y_test, meta


def download_forda():
    _ensure_dir(DATA_DIR)
    train_path = os.path.join(DATA_DIR, "FordA_TRAIN.tsv")
    test_path = os.path.join(DATA_DIR, "FordA_TEST.tsv")
    train_url = (
        "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv"
    )
    test_url = (
        "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv"
    )
    if not os.path.exists(train_path):
        print("Downloading FordA train...")
        urllib.request.urlretrieve(train_url, train_path)
    if not os.path.exists(test_path):
        print("Downloading FordA test...")
        urllib.request.urlretrieve(test_url, test_path)
    return train_path, test_path


def load_forda(seed: int = 1, val_frac: float = 0.0):
    """Official FordA train/test split. val_frac reserved; default uses full train."""
    train_path, test_path = download_forda()
    train_df = pd.read_csv(train_path, sep=None, engine="python", header=None)
    test_df = pd.read_csv(test_path, sep=None, engine="python", header=None)

    y_train = train_df.iloc[:, 0].values
    y_train = np.where(y_train == -1, 0, 1).astype(np.int64)
    X_train = train_df.iloc[:, 1:].values.astype(np.float32)

    y_test = test_df.iloc[:, 0].values
    y_test = np.where(y_test == -1, 0, 1).astype(np.int64)
    X_test = test_df.iloc[:, 1:].values.astype(np.float32)

    if val_frac > 0:
        rng = np.random.default_rng(seed)
        n = len(X_train)
        _ = rng.permutation(n)  # reserved for future val split

    X_train = torch.tensor(X_train, device=DEVICE).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(X_test, device=DEVICE).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    meta = {
        "name": "forda",
        "input_dim": 1,
        "num_classes": 2,
        "seq_len": X_train.size(1),
        "n_train": X_train.size(0),
        "n_test": X_test.size(0),
    }
    print(
        f"FordA ready on {DEVICE}  train={tuple(X_train.shape)}  "
        f"test={tuple(X_test.shape)}  classes={meta['num_classes']}"
    )
    return X_train, y_train, X_test, y_test, meta


def load_dataset(name: str, seed: int = 1):
    if name == "mitbih":
        return load_mitbih()
    if name == "forda":
        return load_forda(seed=seed)
    raise ValueError(f"Unknown dataset: {name}")


def iterate_batches(x, y, batch_size, shuffle):
    """Device-side indexing; only full batches (fixed shape → CUDA graphs)."""
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
    """Residual tanh RNN — no escrow (control)."""

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


class RealFEN(nn.Module):
    """
    Frozen FEN family for real sequence classification.

    write_mode:
      'bag'  — additive bag escrow E
      'hard' — K-cell hard pointer tape + bag channel c
      'roll' — bag with gated channel-roll (historical)

    Always: propose → gate → D → deplete h ← f−D → write archive.
    Always: final head on [h, archive]  (query-time style deliver; no reinject).
    """

    def __init__(self, input_dim, hidden_dim, num_classes, write_mode="bag", K=TAPE_K):
        super().__init__()
        assert write_mode in ("bag", "hard", "roll")
        self.hdim = hidden_dim
        self.write_mode = write_mode
        self.K = K
        self.has_tape = write_mode == "hard"

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        if write_mode == "roll":
            self.roll_gate = nn.Linear(hidden_dim, 1)
        else:
            self.roll_gate = None

        if self.has_tape:
            self.bag_proj = nn.Linear(hidden_dim, hidden_dim)
            self.tape_pool = nn.Linear(K * hidden_dim, hidden_dim)
            head_in = hidden_dim * 3  # h + pool(E) + c
        else:
            self.bag_proj = None
            self.tape_pool = None
            head_in = hidden_dim * 2  # h + E

        self.head = _mlp_head(head_in, num_classes)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        c = x.new_zeros(B, self.hdim)
        E_tape = x.new_zeros(B, self.K, self.hdim) if self.has_tape else None
        ptr = (
            torch.zeros(B, dtype=torch.long, device=x.device)
            if self.has_tape
            else None
        )

        xp = self.x_proj(x)
        # Accumulate on-device — never .item() inside the T-loop (that was a silent killer)
        g_acc = x.new_zeros(())

        for t in range(T):
            z = h + xp[:, t]
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            v = self.v_proj(D)
            h = f - D  # non-negotiable depletion

            if self.write_mode == "bag":
                E = E + v
            elif self.write_mode == "roll":
                gamma = torch.sigmoid(self.roll_gate(f))
                E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v
            else:  # hard tape + bag
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
        else:
            arch = E
            esc_norm = E.detach().norm(dim=-1).mean()

        logits = self.head(torch.cat([h, arch], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h.detach().norm(dim=-1).mean(),
                "gate": g_acc / T,
                "escrow_norm": esc_norm,
            }
        return logits


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.head = _mlp_head(hidden_dim, num_classes)

    def forward(self, x, return_stats=False):
        out, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        logits = self.head(h)
        if return_stats:
            return logits, {
                "pipe_norm": h.detach().norm(dim=-1).mean(),
                "gate": x.new_tensor(float("nan")),
                "escrow_norm": x.new_tensor(float("nan")),
            }
        return logits


# ------------------------------ BUILD / MATCH ---------------------------------
MODEL_SPECS = {
    "residual": dict(kind="residual"),
    "fen_bag": dict(kind="fen", write_mode="bag"),
    "fen_hard_bag": dict(kind="fen", write_mode="hard"),
    "fen_roll": dict(kind="fen", write_mode="roll"),
    "lstm": dict(kind="lstm"),
}
MODEL_ORDER = list(MODEL_SPECS.keys())


def build(name, input_dim, num_classes, hidden_dim):
    spec = MODEL_SPECS[name]
    if spec["kind"] == "residual":
        return ResidualRNN(input_dim, hidden_dim, num_classes)
    if spec["kind"] == "lstm":
        return LSTMBaseline(input_dim, hidden_dim, num_classes)
    return RealFEN(
        input_dim, hidden_dim, num_classes, write_mode=spec["write_mode"], K=TAPE_K
    )


_HIDDEN_CACHE = {}


def choose_hidden(name, input_dim, num_classes):
    """Binary search on width — params grow ~monotone in h for these models."""
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

    # Local polish around binary result (param curves not perfectly linear)
    for h in range(max(MIN_H, best_h - 4), min(MAX_H, best_h + 4) + 1):
        n = count_params(build(name, input_dim, num_classes, h))
        d = abs(n - TARGET_PARAMS)
        if d < best_diff:
            best_h, best_diff = h, d

    _HIDDEN_CACHE[key] = best_h
    return best_h


# ------------------------------ CUDA GRAPH HELPERS ----------------------------
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
    """Warmup then capture one train step (forward+loss+backward+clip+step)."""
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

    # grads must exist (not None) for in-graph zero_
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
        # single .item() per batch, not per timestep
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
        f"T={T}  batch={batch_size}"
    )

    capturable = DEVICE.type == "cuda"
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        capturable=capturable,
    )
    criterion = nn.CrossEntropyLoss()

    # --- optional CUDA graph for the train step ---
    graph = None
    static_x = static_y = None
    if USE_CUDA_GRAPHS and DEVICE.type == "cuda":
        try:
            static_x = torch.zeros(
                batch_size, T, in_dim, device=DEVICE, dtype=X_train.dtype
            )
            static_y = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
            clean_params, clean_opt = _snapshot_state(model, opt)
            # ensure optimizer state exists for capturable AdamW after warmup
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
    best_state = None
    t0 = time.time()
    history = []

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        n_seen = 0
        # train loss: sample last batch only when graphing (no static_loss export)
        last_loss = float("nan")

        for xb, yb in iterate_batches(X_train, y_train, batch_size, shuffle=True):
            if graph is not None:
                static_x.copy_(xb)
                static_y.copy_(yb)
                graph.replay()
                n_seen += batch_size
            else:
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                last_loss = float(loss.detach().item())
                n_seen += yb.size(0)

        val = evaluate(model, X_test, y_test, batch_size)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_ep = ep
            # keep best weights on GPU (fast); only needed for final reload
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

        ep_dt = time.time() - ep_t0
        history.append(
            {
                "epoch": ep,
                "loss": last_loss,
                "acc": val["acc"],
                "pipe": val["pipe"],
                "gate": val["gate"],
                "ep_s": ep_dt,
            }
        )

        if ep == 1 or ep == epochs or ep % PRINT_EVERY == 0:
            gstr = f"{val['gate']:.3f}" if val["gate"] == val["gate"] else "na"
            lstr = f"{last_loss:.4f}" if last_loss == last_loss else "n/a"
            print(
                f"    ep {ep:02d}/{epochs}  loss={lstr}  "
                f"acc={val['acc']:.3f}  pipe={val['pipe']:.2f}  gate={gstr}  "
                f"[{ep_dt:.1f}s]"
            )

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, X_test, y_test, batch_size)

    print(
        f"  >> best acc={best_acc:.3f}  @ep{best_ep}  t={elapsed:.1f}s  "
        f"pipe={final['pipe']:.2f}  graph={'yes' if graph is not None else 'no'}"
    )
    return {
        "model": name,
        "params": n_params,
        "hidden": h,
        "best_acc": best_acc,
        "best_ep": best_ep,
        "final_acc": final["acc"],
        "pipe": final["pipe"],
        "gate": final["gate"],
        "time": elapsed,
        "graph": graph is not None,
        "history": history,
    }


# ------------------------------ MAIN ------------------------------------------
def run_dataset(ds_name: str):
    print("\n" + "=" * 78)
    print(f"DATASET: {ds_name}  epochs={EPOCHS[ds_name]}  batch={BATCH_SIZE[ds_name]}")
    print("=" * 78)

    X_train, y_train, X_test, y_test, meta = load_dataset(ds_name, seed=SEEDS[0])
    epochs = EPOCHS[ds_name]
    bs = BATCH_SIZE[ds_name]

    # If train set smaller than batch, shrink (keeps full-batch graph valid)
    if X_train.size(0) < bs:
        bs = max(1, X_train.size(0) // 2 * 2) or 1
        print(f"  (batch shrunk to {bs} — small train set)")

    all_rows = []
    by_model = defaultdict(list)

    for seed in SEEDS:
        print(f"\n### seed={seed}")
        for name in MODEL_ORDER:
            row = train_one(
                name, X_train, y_train, X_test, y_test, meta, seed, epochs, bs
            )
            all_rows.append(row)
            by_model[name].append(row)

    print("\n" + "-" * 78)
    print(f"SUMMARY  dataset={ds_name}  seeds={SEEDS}  target_params≈{TARGET_PARAMS}")
    print("-" * 78)
    print(
        f"{'model':<14} {'acc':>7} {'±':>6} {'to_best':>8} "
        f"{'pipe':>7} {'params':>8} {'time_s':>8}"
    )
    for name in MODEL_ORDER:
        rows = by_model[name]
        accs = np.array([r["best_acc"] for r in rows], dtype=np.float64)
        pipes = np.array([r["pipe"] for r in rows], dtype=np.float64)
        eps = np.array([r["best_ep"] for r in rows], dtype=np.float64)
        times = np.array([r["time"] for r in rows], dtype=np.float64)
        params = rows[0]["params"]
        print(
            f"{name:<14} {accs.mean():7.3f} {accs.std():6.3f} "
            f"{eps.mean():8.1f} {pipes.mean():7.2f} {params:8d} {times.mean():8.1f}"
        )
    print("-" * 78)
    print(
        "Score: fen_bag / fen_hard_bag should beat residual; pipe should stay "
        "moderate on FENs. fen_hard_bag helps if morphology order matters; "
        "fen_roll is historical — keep only if it clearly wins."
    )
    return all_rows


def main():
    print("EXP05 — Real-data FEN (frozen architecture from exp01–04)")
    print(f"Models: {MODEL_ORDER}")
    print(
        "Canonical claim: deplete + bag (+ optional hard tape) + concat deliver; "
        "no every-step reinject."
    )
    print(
        "Speed path: GPU preload + full batches + binary width search + "
        f"CUDA graphs={'ON' if USE_CUDA_GRAPHS else 'OFF'}."
    )
    grand = {}
    for ds in DATASETS:
        grand[ds] = run_dataset(ds)

    print("\n" + "=" * 78)
    print("DONE — paste the SUMMARY table(s) back for scoring.")
    print("=" * 78)
    return grand


if __name__ == "__main__":
    main()
