# ==============================================================================
# fen_lab / EXP09b — Best-effort LSTM on sMNIST (honesty check for exp09)
# ==============================================================================
# exp09: 1-layer LSTM ~100k, 10 ep → ~chance (0.11) while fen_roll/hybrid ~0.88–0.91.
# Possible confound: under-powered LSTM recipe, not "LSTM cannot do sMNIST".
#
# This script ONLY sweeps LSTM variants on the SAME data protocol as exp09:
#   20×20 pixels → T=400, C=1
#   1500 train / 200 test per digit
#   GPU preload, full batches, TF32
#
# Variants (each ~TARGET_PARAMS when AUTO_MATCH)
#   lstm_1L          1-layer nn.LSTM  (exp09 style)
#   lstm_2L          2-layer nn.LSTM
#   lstm_3L          3-layer nn.LSTM
#   lstm_1L_wide     1-layer, TARGET_PARAMS×1.5 (more capacity)
#   lstm_2L_wide     2-layer, TARGET_PARAMS×1.5
#   lstm_2L_hiLR     2-layer, LR=3e-3
#   lstm_2L_drop     2-layer + dropout between layers
#
# Longer default training (LSTM is cheap via cuDNN).
# Goal: best accuracy any reasonable LSTM reaches under this data protocol.
# If still ≪ FEN ~0.9, exp09 ranking stands; if LSTM reaches high, re-tune FEN vs LSTM.
#
# Data: Kaggle CSV / kagglehub / torchvision (same as exp09).
# Paste whole file → GPU → Run. Deps: torch, numpy; optional pandas, kagglehub.
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

if FAST_MODE:
    SEEDS = [1]
    EPOCHS = 30
    PRINT_EVERY = 2
    BASE_TARGET_PARAMS = 100000
else:
    SEEDS = [1, 2, 3]
    EPOCHS = 50
    PRINT_EVERY = 1
    BASE_TARGET_PARAMS = 100000

IMG_SIZE = 20
SEQ_LEN = IMG_SIZE * IMG_SIZE
TRAIN_PER_CLASS = 1500
TEST_PER_CLASS = 200
NUM_CLASSES = 10

BATCH_SIZE = 256  # larger: LSTM is bandwidth-friendly
LR_DEFAULT = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128
MIN_H, MAX_H = 16, 512
AUTO_MATCH_PARAMS = True

USE_CUDA_GRAPHS = True  # usually fine for nn.LSTM; falls back if not
CUDA_GRAPH_WARMUP_STEPS = 3

# name → (num_layers, param_mult, lr, dropout)
VARIANT_SPECS = {
    "lstm_1L": (1, 1.0, LR_DEFAULT, 0.0),
    "lstm_2L": (2, 1.0, LR_DEFAULT, 0.0),
    "lstm_3L": (3, 1.0, LR_DEFAULT, 0.0),
    "lstm_1L_wide": (1, 1.5, LR_DEFAULT, 0.0),
    "lstm_2L_wide": (2, 1.5, LR_DEFAULT, 0.0),
    "lstm_2L_hiLR": (2, 1.0, 3e-3, 0.0),
    "lstm_2L_drop": (2, 1.0, LR_DEFAULT, 0.2),
}
MODEL_ORDER = list(VARIANT_SPECS.keys())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    USE_CUDA_GRAPHS = False

print(
    f"Device: {DEVICE} | sMNIST LSTM sweep | FAST_MODE={FAST_MODE} | "
    f"T={SEQ_LEN} | EPOCHS={EPOCHS} | BATCH={BATCH_SIZE} | "
    f"CUDA_GRAPHS={USE_CUDA_GRAPHS}"
)
print(
    "EXP09b — Best-effort LSTM variants on same sMNIST protocol as exp09 "
    "(give LSTM a fair chance)"
)
print(f"Variants: {MODEL_ORDER}")


# ------------------------------ UTILS -----------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------ DATA (same as exp09) --------------------------
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

        print("Downloading MNIST via kagglehub...")
        path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
        tr = pd.read_csv(os.path.join(path, "mnist_train.csv"))
        te = pd.read_csv(os.path.join(path, "mnist_test.csv"))
        y_tr = tr["label"].values.astype(np.int64)
        y_te = te["label"].values.astype(np.int64)
        x_tr = tr.drop(columns=["label"]).values.astype(np.float32) / 255.0
        x_te = te.drop(columns=["label"]).values.astype(np.float32) / 255.0
        return x_tr, y_tr, x_te, y_te
    except Exception as e:
        print(f"  kagglehub failed ({type(e).__name__}: {e})")

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
            "Could not load MNIST. On Kaggle add mnist-in-csv. "
            f"Last error: {e}"
        ) from e


def make_smnist():
    x_tr, y_tr, x_te, y_te = _load_mnist_arrays()
    x_tr_t = torch.tensor(x_tr).view(-1, 1, 28, 28)
    x_te_t = torch.tensor(x_te).view(-1, 1, 28, 28)
    print(f"Downsampling 28×28 → {IMG_SIZE}×{IMG_SIZE} (T={SEQ_LEN})...")
    x_tr_t = F.interpolate(
        x_tr_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    x_te_t = F.interpolate(
        x_te_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    x_tr_seq = x_tr_t.squeeze(1).reshape(-1, SEQ_LEN, 1).numpy()
    x_te_seq = x_te_t.squeeze(1).reshape(-1, SEQ_LEN, 1).numpy()

    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for d in range(10):
        train_idx.extend(
            rng.choice(np.where(y_tr == d)[0], TRAIN_PER_CLASS, replace=False).tolist()
        )
        test_idx.extend(
            rng.choice(np.where(y_te == d)[0], TEST_PER_CLASS, replace=False).tolist()
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

    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    x_te = torch.tensor(x_te, dtype=torch.float32, device=DEVICE)
    y_te = torch.tensor(y_te, dtype=torch.long, device=DEVICE)

    meta = {
        "input_dim": 1,
        "num_classes": NUM_CLASSES,
        "seq_len": SEQ_LEN,
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
    }
    print(
        f"sMNIST ready on {DEVICE}: train={tuple(x_tr.shape)} test={tuple(x_te.shape)}"
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


# ------------------------------ MODEL -----------------------------------------
class LSTMVar(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.0
    ):
        super().__init__()
        self.hdim = hidden_dim
        self.num_layers = num_layers
        # dropout only applies when num_layers > 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, HEAD_WIDTH),
            nn.ReLU(),
            nn.Linear(HEAD_WIDTH, num_classes),
        )

    def forward(self, x, return_stats=False):
        out, (h_n, c_n) = self.lstm(x)
        h = h_n[-1]
        logits = self.head(h)
        if return_stats:
            return logits, {
                "pipe_norm": h.detach().norm(dim=-1).mean(),
                "cell_norm": c_n[-1].detach().norm(dim=-1).mean(),
            }
        return logits


def build(name, input_dim, num_classes, hidden_dim):
    nL, _mult, _lr, drop = VARIANT_SPECS[name]
    return LSTMVar(
        input_dim, hidden_dim, num_classes, num_layers=nL, dropout=drop
    )


_HIDDEN_CACHE = {}


def choose_hidden(name, input_dim, num_classes):
    nL, mult, _lr, drop = VARIANT_SPECS[name]
    target = int(BASE_TARGET_PARAMS * mult)
    key = (name, input_dim, num_classes, target, nL, drop)
    if key in _HIDDEN_CACHE:
        return _HIDDEN_CACHE[key]
    if not AUTO_MATCH_PARAMS:
        _HIDDEN_CACHE[key] = 128
        return 128

    lo, hi = MIN_H, MAX_H
    best_h, best_diff = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(build(name, input_dim, num_classes, mid))
        d = abs(n - target)
        if d < best_diff:
            best_h, best_diff = mid, d
        if n < target:
            lo = mid + 1
        elif n > target:
            hi = mid - 1
        else:
            break
    for h in range(max(MIN_H, best_h - 6), min(MAX_H, best_h + 6) + 1):
        n = count_params(build(name, input_dim, num_classes, h))
        d = abs(n - target)
        if d < best_diff:
            best_h, best_diff = h, d
    _HIDDEN_CACHE[key] = best_h
    return best_h


def target_params(name):
    return int(BASE_TARGET_PARAMS * VARIANT_SPECS[name][1])


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
    correct = 0
    n = 0
    pipe_sum = cell_sum = 0.0
    nb = 0
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits, st = model(xb, return_stats=True)
        correct += (logits.argmax(-1) == yb).sum().item()
        n += yb.numel()
        pipe_sum += float(st["pipe_norm"].item())
        cell_sum += float(st["cell_norm"].item())
        nb += 1
    return {
        "acc": correct / max(n, 1),
        "pipe": pipe_sum / max(nb, 1),
        "cell": cell_sum / max(nb, 1),
    }


def train_one(name, X_train, y_train, X_test, y_test, meta, seed, epochs, batch_size):
    seed_everything(seed)
    in_dim = meta["input_dim"]
    n_cls = meta["num_classes"]
    T = meta["seq_len"]
    nL, mult, lr, drop = VARIANT_SPECS[name]
    h = choose_hidden(name, in_dim, n_cls)
    model = build(name, in_dim, n_cls, h).to(DEVICE)
    n_params = count_params(model)
    tgt = target_params(name)

    print(f"\n--- Model: {name} ---")
    print(
        f"  layers={nL}  hidden={h}  params={n_params}  target≈{tgt}  "
        f"lr={lr}  drop={drop}  epochs={epochs}  seed={seed}  T={T}"
    )

    capturable = DEVICE.type == "cuda"
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, capturable=capturable
    )
    # mild schedule: decay after half training
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 2, 1), gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    graph = None
    static_x = static_y = None
    use_graph = USE_CUDA_GRAPHS and DEVICE.type == "cuda"
    # StepLR + CUDA graph is awkward (lr outside graph); use eager when schedule matters
    # We step sched per epoch outside graph; graph only captures train step with fixed lr
    # After sched.step(), rebuild is heavy — use eager for simplicity + correctness with sched
    use_graph = False  # prefer reliable LR schedule over graph for this honesty sweep

    if use_graph:
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
                model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
            )
            sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=max(epochs // 2, 1), gamma=0.5
            )
    else:
        print("  [eager train + StepLR]")

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
        sched.step()

        val = evaluate(model, X_test, y_test, batch_size)
        history.append(val["acc"])
        if val["acc"] > best_acc:
            best_acc, best_ep, best_snap = val["acc"], ep, dict(val)

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            print(
                f"    ep {ep:02d}/{epochs}  acc={val['acc']:.3f}  "
                f"pipe={val['pipe']:.2f}  cell={val['cell']:.2f}  "
                f"lr={opt.param_groups[0]['lr']:.1e}  "
                f"[{time.time() - ep_t0:.1f}s]"
            )

    elapsed = time.time() - t0
    # epochs to first hit 0.5 / 0.8 if ever
    def first_above(th):
        for i, a in enumerate(history, 1):
            if a >= th:
                return i
        return None

    print(
        f"  >> best acc={best_acc:.3f}  @ep{best_ep}  t={elapsed:.1f}s  "
        f"to0.5={first_above(0.5)}  to0.8={first_above(0.8)}  "
        f"ep1={history[0]:.3f}  ep2={history[1] if len(history)>1 else float('nan'):.3f}"
    )
    return {
        "name": name,
        "acc": best_acc,
        "best_ep": best_ep,
        "pipe": best_snap["pipe"],
        "params": n_params,
        "hidden": h,
        "layers": nL,
        "lr": lr,
        "time": elapsed,
        "ep1": history[0],
        "ep2": history[1] if len(history) > 1 else float("nan"),
        "to50": first_above(0.5),
        "to80": first_above(0.8),
        "last": history[-1],
    }


def main():
    print(
        f"BASE_TARGET≈{BASE_TARGET_PARAMS} | EPOCHS={EPOCHS} | SEEDS={SEEDS}\n"
        "Compare to exp09 FEN: roll~0.88 hybrid~0.91 bag~0.66 residual~0.10 @10ep.\n"
        "If best LSTM still ~chance or <<0.8, exp09 FEN lead is not an epoch artifact."
    )

    X_tr, y_tr, X_te, y_te, meta = make_smnist()
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
            row = train_one(
                name, X_tr, y_tr, X_te, y_te, meta, seed, EPOCHS, bs
            )
            by_model[name].append(row)

    print("\n" + "-" * 88)
    print(
        f"SUMMARY  smnist_LSTM_sweep  T={meta['seq_len']}  seeds={SEEDS}  "
        f"epochs={EPOCHS}"
    )
    print("-" * 88)
    print(
        f"{'model':<16} {'acc':>7} {'ep1':>6} {'ep2':>6} {'last':>6} "
        f"{'to_best':>8} {'to50':>5} {'to80':>5} {'L':>2} {'params':>8} {'time':>7}"
    )
    for name in MODEL_ORDER:
        rows = by_model[name]
        acc = np.mean([r["acc"] for r in rows])
        ep1 = np.mean([r["ep1"] for r in rows])
        ep2 = np.mean([r["ep2"] for r in rows])
        last = np.mean([r["last"] for r in rows])
        epb = np.mean([r["best_ep"] for r in rows])
        t50 = rows[0]["to50"]
        t80 = rows[0]["to80"]
        L = rows[0]["layers"]
        params = rows[0]["params"]
        tm = np.mean([r["time"] for r in rows])
        s50 = f"{t50}" if t50 is not None else "-"
        s80 = f"{t80}" if t80 is not None else "-"
        print(
            f"{name:<16} {acc:7.3f} {ep1:6.3f} {ep2:6.3f} {last:6.3f} "
            f"{epb:8.1f} {s50:>5} {s80:>5} {L:2d} {params:8d} {tm:7.1f}"
        )
    print("-" * 88)
    best_name = max(MODEL_ORDER, key=lambda n: by_model[n][0]["acc"])
    best_acc = by_model[best_name][0]["acc"]
    print(
        f"BEST LSTM in this sweep: {best_name}  acc={best_acc:.3f}\n"
        f"exp09 reference: fen_hybrid≈0.906  fen_roll≈0.881  fen_bag≈0.661  "
        f"lstm_1L@10ep≈0.110\n"
        "If best LSTM ≥ ~0.85, re-compare FEN vs that recipe; "
        "if best LSTM ≪ FEN, hard-bench ranking stands."
    )
    print("DONE — paste this SUMMARY back for scoring.")
    return by_model


if __name__ == "__main__":
    main()
