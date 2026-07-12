# ==============================================================================
# fen_lab / EXP05b — FordA only (parallel Colab cell)
# ==============================================================================
# Same frozen FEN models as exp05_real_data.py, dataset = UCR FordA only.
# Paste this whole file into a Colab GPU cell and run.
#
# Models (~TARGET_PARAMS):
#   residual | fen_bag | fen_hard_bag | fen_roll | lstm
#
# Pilot (done): seed=1, 20 ep → roll 0.80; bag/hard ~0.59–0.60; residual/lstm ~chance
# This config: seed=2, 40 ep — longer train + new seed; does bag/hard catch roll?
#
# Speed path (same lessons as the optimized trainer):
#   GPU preload · full fixed batches · binary width search · TF32 · CUDA graphs
#   no .item() inside the T-loop
#
# Deps: torch, numpy, pandas. Auto-downloads FordA TSVs.
# ==============================================================================

import os
import random
import time
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ CONFIG ----------------------------------------
# True  = one seed, longer single-run (default for this re-run)
# False = 3 seeds × 40 ep multi-seed confirmation
FAST_MODE = True

if FAST_MODE:
    # Longer FordA re-run: new seed (not 1), double epochs vs pilot
    SEEDS = [2]
    EPOCHS = 40
    PRINT_EVERY = 2
    TARGET_PARAMS = 75000
else:
    SEEDS = [1, 2, 3]
    EPOCHS = 40
    PRINT_EVERY = 1
    TARGET_PARAMS = 75000

BATCH_SIZE = 128          # FordA T=500; raise to 256 if GPU memory allows
LR = 1e-3                 # matches older FordA runs (long seq, smaller LR)
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128
TAPE_K = 8
EVENT_GATE_THRESH = 0.25
MIN_H, MAX_H = 16, 256
AUTO_MATCH_PARAMS = True
DATA_DIR = "./data"

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
    f"Device: {DEVICE} | FordA | FAST_MODE={FAST_MODE} | "
    f"SEEDS={SEEDS} | EPOCHS={EPOCHS} | BATCH={BATCH_SIZE} | "
    f"CUDA_GRAPHS={USE_CUDA_GRAPHS}"
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


# ------------------------------ DATA (FordA) ----------------------------------
def download_forda():
    os.makedirs(DATA_DIR, exist_ok=True)
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
        print("  done.")
    if not os.path.exists(test_path):
        print("Downloading FordA test...")
        urllib.request.urlretrieve(test_url, test_path)
        print("  done.")
    return train_path, test_path


def load_forda():
    train_path, test_path = download_forda()
    print("Parsing FordA TSVs...")
    train_df = pd.read_csv(train_path, sep=None, engine="python", header=None)
    test_df = pd.read_csv(test_path, sep=None, engine="python", header=None)

    y_train = np.where(train_df.iloc[:, 0].values == -1, 0, 1).astype(np.int64)
    X_train = train_df.iloc[:, 1:].values.astype(np.float32)
    y_test = np.where(test_df.iloc[:, 0].values == -1, 0, 1).astype(np.int64)
    X_test = test_df.iloc[:, 1:].values.astype(np.float32)

    # Full preload on device once
    X_train = torch.tensor(X_train, device=DEVICE).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(X_test, device=DEVICE).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    meta = {
        "name": "forda",
        "input_dim": 1,
        "num_classes": 2,
        "seq_len": int(X_train.size(1)),
        "n_train": int(X_train.size(0)),
        "n_test": int(X_test.size(0)),
    }
    print(
        f"FordA on {DEVICE}: train={tuple(X_train.shape)} test={tuple(X_test.shape)} "
        f"classes=2 T={meta['seq_len']}"
    )
    return X_train, y_train, X_test, y_test, meta


def iterate_batches(x, y, batch_size, shuffle):
    """Device-side full batches only (fixed B → CUDA graphs)."""
    n = x.shape[0]
    n_batches = n // batch_size
    if n_batches == 0:
        return
    perm = (
        torch.randperm(n, device=x.device)
        if shuffle
        else torch.arange(n, device=x.device)
    )
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


class RealFEN(nn.Module):
    """
    write_mode: 'bag' | 'hard' | 'roll'
    deplete always; final head([h, arch]); no every-step reinject.
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
            head_in = hidden_dim * 3
        else:
            self.bag_proj = None
            self.tape_pool = None
            head_in = hidden_dim * 2

        self.head = _mlp_head(head_in, num_classes)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
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
            else:
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
            esc_norm = (
                E_tape.detach().norm(dim=-1).mean() + c.detach().norm(dim=-1).mean()
            )
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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        capturable=(DEVICE.type == "cuda"),
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
    best_state = None
    t0 = time.time()
    history = []

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        last_loss = float("nan")

        for xb, yb in iterate_batches(X_train, y_train, batch_size, shuffle=True):
            if graph is not None:
                static_x.copy_(xb)
                static_y.copy_(yb)
                graph.replay()
            else:
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                last_loss = float(loss.detach().item())

        val = evaluate(model, X_test, y_test, batch_size)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
def main():
    print("EXP05b — FordA real-data FEN (frozen architecture from exp01–04)")
    print(f"Models: {MODEL_ORDER}")
    print(
        "Canonical: deplete + bag (+ optional hard tape) + concat deliver; "
        "no every-step reinject."
    )

    X_train, y_train, X_test, y_test, meta = load_forda()
    bs = BATCH_SIZE
    if X_train.size(0) < bs:
        bs = max(1, (X_train.size(0) // 2) * 2) or 1
        print(f"  batch shrunk to {bs}")

    by_model = defaultdict(list)

    for seed in SEEDS:
        print(f"\n### seed={seed}")
        for name in MODEL_ORDER:
            row = train_one(
                name, X_train, y_train, X_test, y_test, meta, seed, EPOCHS, bs
            )
            by_model[name].append(row)

    print("\n" + "-" * 78)
    print(f"SUMMARY  dataset=forda  seeds={SEEDS}  target_params≈{TARGET_PARAMS}")
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
        "Score (FordA longer): pilot seed1/20ep had roll 0.80 vs bag/hard ~0.60. "
        "Does bag/hard catch roll at 40 ep? Does roll still dominate on seed 2?"
    )
    print("DONE — paste this SUMMARY back for scoring.")
    return by_model


if __name__ == "__main__":
    main()
