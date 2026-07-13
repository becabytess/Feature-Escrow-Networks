# ==============================================================================
# fen_lab / EXP09 — Sequential MNIST (sMNIST) hard-bench variant sweep
# ==============================================================================
# Why this task
#   Foundation probes (distracted / recall) are at ceiling for many FEN variants —
#   bad for ranking. sMNIST is a long pixel stream with real headroom where LSTM
#   is competitive. If all FENs ≈ LSTM, the task/protocol still may be wrong;
#   if gaps open, we can rank write/delivery choices for hard seq vision.
#
# Protocol (matches older FEN/sMNIST style)
#   MNIST → bilinear resize 28→20 → sequence T=400, C=1
#   Stratified subset: 1500 train / 200 test per digit (15k / 2k)
#   ~100k params, AdamW, GPU preload, CUDA graphs
#
# Models
#   residual         residual tanh RNN, no escrow
#   fen_bag          deplete + bag + head([h,E])
#   fen_copy         bag write but NO deplete (h=f) — is subtraction load-bearing?
#   fen_hard_bag     hard pointer tape + bag
#   fen_roll         channel-roll escrow
#   fen_hybrid       bag + roll dual vault
#   fen_reinject     bag + every-step E→h (pathology control)
#   fen_2pass_cold   bag, discrete read between two passes
#   lstm             1-layer nn.LSTM baseline
#
# Data sources (first hit wins)
#   1) /kaggle/input/**  mnist_train.csv + mnist_test.csv
#   2) kagglehub  oddrationale/mnist-in-csv
#   3) torchvision MNIST (./data)
#
# Kaggle/Colab: paste whole file → GPU → Run.
# Deps: torch, numpy; optional pandas, kagglehub, torchvision.
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
    EPOCHS = 20
    PRINT_EVERY = 2
    TARGET_PARAMS = 100000
else:
    SEEDS = [1, 2, 3]
    EPOCHS = 30
    PRINT_EVERY = 1
    TARGET_PARAMS = 100000

IMG_SIZE = 20  # 20×20 → T=400
SEQ_LEN = IMG_SIZE * IMG_SIZE
TRAIN_PER_CLASS = 1500
TEST_PER_CLASS = 200
NUM_CLASSES = 10

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128
TAPE_K = 8
EVENT_GATE_THRESH = 0.25
MIN_H, MAX_H = 16, 256
AUTO_MATCH_PARAMS = True

USE_CUDA_GRAPHS = True
CUDA_GRAPH_WARMUP_STEPS = 3

MODEL_ORDER = [
    "residual",
    "fen_bag",
    "fen_copy",
    "fen_hard_bag",
    "fen_roll",
    "fen_hybrid",
    "fen_reinject",
    "fen_2pass_cold",
    "lstm",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    USE_CUDA_GRAPHS = False

print(
    f"Device: {DEVICE} | sMNIST | FAST_MODE={FAST_MODE} | "
    f"T={SEQ_LEN} | EPOCHS={EPOCHS} | BATCH={BATCH_SIZE} | "
    f"CUDA_GRAPHS={USE_CUDA_GRAPHS}"
)
print(
    "EXP09 — Hard-bench FEN variant sweep on sequential MNIST "
    "(rank writes under headroom; LSTM is the honesty check)"
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
def _load_mnist_arrays():
    """Return x_train (N,784), y_train, x_test, y_test as float32 [0,1] / int64."""

    # 1) Kaggle input CSVs
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        print("Searching /kaggle/input for mnist_*.csv ...")
        train_csv = test_csv = None
        for root, _dirs, files in os.walk(kaggle_root):
            fl = {f.lower(): os.path.join(root, f) for f in files}
            # common names
            for tk in ("mnist_train.csv", "train.csv"):
                if tk in fl or tk.replace(".csv", "") + ".csv" in [f.lower() for f in files]:
                    pass
            lower_map = {f.lower(): os.path.join(root, f) for f in files}
            if "mnist_train.csv" in lower_map and "mnist_test.csv" in lower_map:
                train_csv = lower_map["mnist_train.csv"]
                test_csv = lower_map["mnist_test.csv"]
                break
        if train_csv and test_csv:
            import pandas as pd

            print(f"Loading Kaggle CSVs:\n  {train_csv}\n  {test_csv}")
            tr = pd.read_csv(train_csv)
            te = pd.read_csv(test_csv)
            # column name may be label or Label
            lab = "label" if "label" in tr.columns else tr.columns[0]
            y_tr = tr[lab].values.astype(np.int64)
            y_te = te[lab].values.astype(np.int64)
            x_tr = tr.drop(columns=[lab]).values.astype(np.float32) / 255.0
            x_te = te.drop(columns=[lab]).values.astype(np.float32) / 255.0
            return x_tr, y_tr, x_te, y_te

    # 2) kagglehub
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

    # 3) torchvision
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
            "Could not load MNIST. On Kaggle: add mnist-in-csv dataset. "
            f"Last error: {e}"
        ) from e


def make_smnist():
    x_tr, y_tr, x_te, y_te = _load_mnist_arrays()

    # (N, 1, 28, 28) → resize → (N, T, 1)
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
        "name": "smnist",
        "input_dim": 1,
        "num_classes": NUM_CLASSES,
        "seq_len": SEQ_LEN,
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
    }
    print(
        f"sMNIST ready on {DEVICE}: train={tuple(x_tr.shape)} test={tuple(x_te.shape)} "
        f"classes={NUM_CLASSES}  subset={TRAIN_PER_CLASS}/class train, "
        f"{TEST_PER_CLASS}/class test"
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
    write_mode:
      bag | hard | roll | hybrid | reinject | copy | twopass_cold
    copy: write bag but h = f (no deplete)
    reinject: deplete + write, then h += tanh(rj(E)) every step
    twopass_cold: two scans, discrete read c between passes
    """

    def __init__(self, input_dim, hidden_dim, num_classes, write_mode="bag", K=TAPE_K):
        super().__init__()
        assert write_mode in (
            "bag",
            "hard",
            "roll",
            "hybrid",
            "reinject",
            "copy",
            "twopass_cold",
        )
        self.hdim = hidden_dim
        self.write_mode = write_mode
        self.K = K
        self.has_tape = write_mode == "hard"
        self.is_hybrid = write_mode == "hybrid"
        self.is_twopass = write_mode == "twopass_cold"
        self.no_deplete = write_mode == "copy"
        self.do_reinject = write_mode == "reinject"

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        if write_mode in ("roll", "hybrid"):
            self.roll_gate = nn.Linear(hidden_dim, 1)
        else:
            self.roll_gate = None

        if self.do_reinject:
            self.rj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.rj = None

        if self.is_twopass:
            self.read_proj = nn.Linear(hidden_dim, hidden_dim)
            self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.read_proj = None
            self.c_proj = None

        if self.has_tape:
            self.bag_proj = nn.Linear(hidden_dim, hidden_dim)
            self.tape_pool = nn.Linear(K * hidden_dim, hidden_dim)
            head_in = hidden_dim * 3
        elif self.is_hybrid:
            head_in = hidden_dim * 3
            self.bag_proj = None
            self.tape_pool = None
        else:
            head_in = hidden_dim * 2
            self.bag_proj = None
            self.tape_pool = None

        self.head = _mlp_head(head_in, num_classes)

    def _step(self, h, E, E_roll, E_tape, ptr, c_bag, x_t, c_ctx, g_acc):
        z = h + x_t
        if c_ctx is not None:
            z = z + self.c_proj(c_ctx)
        f = torch.tanh(self.core(z) + z)
        g = torch.sigmoid(self.gate(f))
        D = g * f
        v = self.v_proj(D)

        if self.no_deplete:
            h = f
        else:
            h = f - D

        mode = self.write_mode
        if mode in ("bag", "copy", "reinject", "twopass_cold"):
            E = E + v
        elif mode == "roll":
            gamma = torch.sigmoid(self.roll_gate(f))
            E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v
        elif mode == "hybrid":
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
            c_bag = c_bag + self.bag_proj(D)

        if self.do_reinject:
            h = h + torch.tanh(self.rj(E))

        if g_acc is not None:
            g_acc = g_acc + g.detach().mean()
        return h, E, E_roll, E_tape, ptr, c_bag, g_acc

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        E_roll = x.new_zeros(B, self.hdim) if self.is_hybrid else None
        c_bag = x.new_zeros(B, self.hdim)
        E_tape = x.new_zeros(B, self.K, self.hdim) if self.has_tape else None
        ptr = (
            torch.zeros(B, dtype=torch.long, device=x.device) if self.has_tape else None
        )
        xp = self.x_proj(x)
        g_acc = x.new_zeros(()) if return_stats else None

        def run_pass(h, E, E_roll, E_tape, ptr, c_bag, c_ctx, g_acc):
            for t in range(T):
                h, E, E_roll, E_tape, ptr, c_bag, g_acc = self._step(
                    h, E, E_roll, E_tape, ptr, c_bag, xp[:, t], c_ctx, g_acc
                )
            return h, E, E_roll, E_tape, ptr, c_bag, g_acc

        if self.is_twopass:
            h, E, E_roll, E_tape, ptr, c_bag, g_acc = run_pass(
                h, E, E_roll, E_tape, ptr, c_bag, None, g_acc
            )
            c_ctx = torch.tanh(self.read_proj(E))
            h = x.new_zeros(B, self.hdim)  # cold
            h, E, E_roll, E_tape, ptr, c_bag, g_acc = run_pass(
                h, E, E_roll, E_tape, ptr, c_bag, c_ctx, g_acc
            )
            n_steps = 2 * T
        else:
            h, E, E_roll, E_tape, ptr, c_bag, g_acc = run_pass(
                h, E, E_roll, E_tape, ptr, c_bag, None, g_acc
            )
            n_steps = T

        if self.has_tape:
            pooled = torch.tanh(self.tape_pool(E_tape.reshape(B, -1)))
            arch = torch.cat([pooled, c_bag], dim=-1)
            esc_norm = E_tape.detach().norm(dim=-1).mean() + c_bag.detach().norm(
                dim=-1
            ).mean()
        elif self.is_hybrid:
            arch = torch.cat([E, E_roll], dim=-1)
            esc_norm = E.detach().norm(dim=-1).mean() + E_roll.detach().norm(
                dim=-1
            ).mean()
        else:
            arch = E
            esc_norm = E.detach().norm(dim=-1).mean()

        logits = self.head(torch.cat([h, arch], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": h.detach().norm(dim=-1).mean(),
                "gate": g_acc / max(n_steps, 1)
                if g_acc is not None
                else x.new_tensor(float("nan")),
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
    "fen_hard_bag": dict(kind="fen", write_mode="hard"),
    "fen_roll": dict(kind="fen", write_mode="roll"),
    "fen_hybrid": dict(kind="fen", write_mode="hybrid"),
    "fen_reinject": dict(kind="fen", write_mode="reinject"),
    "fen_2pass_cold": dict(kind="fen", write_mode="twopass_cold"),
    "lstm": dict(kind="lstm"),
}


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
            best_acc, best_ep, best_snap = val["acc"], ep, dict(val)

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            print(
                f"    ep {ep:02d}/{epochs}  acc={val['acc']:.3f}  "
                f"pipe={val['pipe']:.2f}  gate={val['gate']:.3f}  "
                f"[{time.time() - ep_t0:.1f}s]"
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
    }


def main():
    print(
        f"TARGET_PARAMS≈{TARGET_PARAMS} | EPOCHS={EPOCHS} | SEEDS={SEEDS}\n"
        "Chance@10 ≈ 0.10. If all FEN ≈ LSTM, hard-task ranking is weak; "
        "if gaps open, write/delivery choices matter under headroom."
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

    print("\n" + "-" * 78)
    print(
        f"SUMMARY  smnist  T={meta['seq_len']}  seeds={SEEDS}  "
        f"target_params≈{TARGET_PARAMS}"
    )
    print("-" * 78)
    print(
        f"{'model':<16} {'acc':>7} {'±':>6} {'to_best':>8} "
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
            f"{name:<16} {accs.mean():7.3f} {accs.std():6.3f} "
            f"{eps.mean():8.1f} {pipes.mean():7.2f} {params:8d} {times.mean():8.1f}"
        )
    print("-" * 78)
    print(
        "How to read:\n"
        "  • residual fat pipe / low acc → dual-load still hurts\n"
        "  • fen_* vs lstm: if tied, ranking writes is weak on this protocol\n"
        "  • fen_roll vs fen_bag: ordered-scan bias under headroom?\n"
        "  • fen_copy: deplete off — does subtraction matter here?\n"
        "  • fen_reinject: expect fatter pipe; acc may or may not look OK\n"
        "  • fen_2pass_cold: discrete multi-pass on a hard task\n"
        "  Curves matter: to_best epoch + final plateau."
    )
    print("DONE — paste this SUMMARY back for scoring.")
    return by_model


if __name__ == "__main__":
    main()
