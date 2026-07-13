# ==============================================================================
# fen_lab / EXP12b — sMNIST: roll WITHOUT deplete only (early signal)
# ==============================================================================
# Point
#   Full bake-offs are too slow / too many variants for the deplete question.
#   Winner on sMNIST was fen_roll (+ deplete). We already know that number.
#   Only missing cell: same roll write, deplete OFF (h = f, still write E).
#
# Question
#   Does roll's early jump (exp08: ep1≈0.64, ep2≈0.80, peak≈0.88) need deplete?
#   If roll_nodep still opens strong early → deplete is NOT the roll story.
#   If ep1 collapses toward bag (~0.24) → deplete was load-bearing for roll.
#
# Run
#   ONE model only. Kill after epoch 2–5 if the answer is already obvious.
#   Optional: set EPOCHS higher if you want peak; early is enough for the law.
#
# Reference (do NOT re-run; exp08 seed1 ~100k T=400)
#   fen_roll + deplete:  ep1≈0.64  ep2≈0.80  best≈0.88
#   fen_bag  + deplete:  ep1≈0.24  ep2≈0.36  best≈0.66
#   fen_copy (bag, no dep): best≈0.78  (copy beat bag; roll was still king)
#
# Kaggle/Colab: paste whole file → GPU → Run.
# ==============================================================================

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ CONFIG ----------------------------------------
FAST_MODE = True

if FAST_MODE:
    SEEDS = [1]
    EPOCHS = 10  # early answer by ep2; rest optional
    PRINT_EVERY = 1
    TARGET_PARAMS = 100000
else:
    SEEDS = [1]
    EPOCHS = 15
    PRINT_EVERY = 1
    TARGET_PARAMS = 100000

IMG_SIZE = 20
SEQ_LEN = IMG_SIZE * IMG_SIZE
TRAIN_PER_CLASS = 1500
TEST_PER_CLASS = 200
NUM_CLASSES = 10

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HEAD_WIDTH = 128
MIN_H, MAX_H = 16, 256
AUTO_MATCH_PARAMS = True

USE_CUDA_GRAPHS = True
CUDA_GRAPH_WARMUP_STEPS = 3

# exp08 fen_roll (+ deplete) anchors — comparison only
REF_ROLL_DEP = {"ep1": 0.64, "ep2": 0.80, "best": 0.88, "pipe": 10.0}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    USE_CUDA_GRAPHS = False

print(
    f"Device: {DEVICE} | EXP12b roll_nodep sMNIST | T={SEQ_LEN} | "
    f"EPOCHS={EPOCHS} | CUDA_GRAPHS={USE_CUDA_GRAPHS}"
)
print(
    "ONE model: channel-roll write, deplete=OFF (h=f), final head([h,E]).\n"
    f"Compare early to exp08 roll+deplete: ep1≈{REF_ROLL_DEP['ep1']} "
    f"ep2≈{REF_ROLL_DEP['ep2']} best≈{REF_ROLL_DEP['best']}\n"
    "Enough to stop after ep2 if ep1 is clearly high or clearly dead."
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


# ------------------------------ DATA ------------------------------------------
def _load_mnist_arrays():
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        print("Searching /kaggle/input for mnist_*.csv ...")
        for root, _dirs, files in os.walk(kaggle_root):
            lower_map = {f.lower(): os.path.join(root, f) for f in files}
            if "mnist_train.csv" in lower_map and "mnist_test.csv" in lower_map:
                import pandas as pd

                tr = pd.read_csv(lower_map["mnist_train.csv"])
                te = pd.read_csv(lower_map["mnist_test.csv"])
                print(
                    f"Loading Kaggle CSVs:\n  {lower_map['mnist_train.csv']}\n  "
                    f"{lower_map['mnist_test.csv']}"
                )
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

    from torchvision import datasets

    print("Loading MNIST via torchvision → ./data ...")
    tr = datasets.MNIST(root="./data", train=True, download=True)
    te = datasets.MNIST(root="./data", train=False, download=True)
    x_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = tr.targets.numpy().astype(np.int64)
    x_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_te = te.targets.numpy().astype(np.int64)
    return x_tr, y_tr, x_te, y_te


def load_smnist():
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

    Xtr = torch.tensor(x_tr, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    Xte = torch.tensor(x_te, dtype=torch.float32, device=DEVICE)
    yte = torch.tensor(y_te, dtype=torch.long, device=DEVICE)
    print(f"sMNIST ready: train={tuple(Xtr.shape)} test={tuple(Xte.shape)}")
    return Xtr, ytr, Xte, yte


# ------------------------------ MODEL -----------------------------------------
def _mlp_head(in_dim, out_dim, width=HEAD_WIDTH):
    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.ReLU(),
        nn.Linear(width, out_dim),
    )


class RollFEN(nn.Module):
    """Channel-roll escrow; deplete flag only difference from classic fen_roll."""

    def __init__(self, input_dim, hidden_dim, num_classes, deplete: bool):
        super().__init__()
        self.hdim = hidden_dim
        self.deplete = deplete
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
            h = (f - D) if self.deplete else f
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


def choose_hidden():
    if not AUTO_MATCH_PARAMS:
        return 64
    lo, hi = MIN_H, MAX_H
    best_h, best_diff = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(RollFEN(1, mid, NUM_CLASSES, deplete=False))
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
        n = count_params(RollFEN(1, h, NUM_CLASSES, deplete=False))
        d = abs(n - TARGET_PARAMS)
        if d < best_diff:
            best_h, best_diff = h, d
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
            loss = criterion(model(static_x), static_y)
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
        loss = criterion(model(static_x), static_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    return g


@torch.no_grad()
def evaluate(model, X, y, batch_size):
    model.eval()
    correct = n = 0
    pipe_sum = gate_sum = 0.0
    nb = 0
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits, st = model(xb, return_stats=True)
        correct += (logits.argmax(-1) == yb).sum().item()
        n += yb.numel()
        pipe_sum += float(st["pipe_norm"].item())
        gate_sum += float(st["gate"].item())
        nb += 1
    return {
        "acc": correct / max(n, 1),
        "pipe": pipe_sum / max(nb, 1),
        "gate": gate_sum / max(nb, 1),
    }


def main():
    Xtr, ytr, Xte, yte = load_smnist()
    bs = BATCH_SIZE
    n_tr = (Xtr.shape[0] // bs) * bs
    n_te = (Xte.shape[0] // bs) * bs
    Xtr, ytr = Xtr[:n_tr], ytr[:n_tr]
    Xte, yte = Xte[:n_te], yte[:n_te]
    print(f"  full batches: train {n_tr} test {n_te}")

    seed = SEEDS[0]
    seed_everything(seed)
    h = choose_hidden()
    model = RollFEN(1, h, NUM_CLASSES, deplete=False).to(DEVICE)
    n_params = count_params(model)
    print(
        f"\n--- roll_nodep (roll write, deplete=OFF) ---\n"
        f"  hidden={h}  params={n_params}  seed={seed}  T={SEQ_LEN}"
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
            static_x = torch.zeros(bs, SEQ_LEN, 1, device=DEVICE, dtype=Xtr.dtype)
            static_y = torch.zeros(bs, dtype=torch.long, device=DEVICE)
            clean_p, clean_o = _snapshot_state(model, opt)
            graph = _try_build_cuda_graph(model, opt, criterion, static_x, static_y)
            _restore_state(model, opt, clean_p, clean_o)
            print("  [CUDA graph capture OK]")
        except Exception as e:
            print(f"  [CUDA graph failed ({type(e).__name__}: {e}); eager]")
            graph = None
            seed_everything(seed)
            model = RollFEN(1, h, NUM_CLASSES, deplete=False).to(DEVICE)
            opt = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )

    history = []
    best_acc, best_ep, best_pipe = -1.0, 0, float("nan")
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        ep_t0 = time.time()
        model.train()
        for xb, yb in iterate_batches(Xtr, ytr, bs, shuffle=True):
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

        val = evaluate(model, Xte, yte, bs)
        history.append(val["acc"])
        if val["acc"] > best_acc:
            best_acc, best_ep, best_pipe = val["acc"], ep, val["pipe"]

        print(
            f"  ep {ep:02d}/{EPOCHS}  acc={val['acc']:.3f}  "
            f"pipe={val['pipe']:.2f}  gate={val['gate']:.3f}  "
            f"[{time.time() - ep_t0:.1f}s]"
        )

        # early verdict banner after ep2
        if ep == 2:
            e1, e2 = history[0], history[1]
            print("\n  --- EARLY VERDICT (vs exp08 roll+deplete) ---")
            print(
                f"  roll_nodep:  ep1={e1:.3f}  ep2={e2:.3f}\n"
                f"  roll_dep:    ep1≈{REF_ROLL_DEP['ep1']:.2f}  "
                f"ep2≈{REF_ROLL_DEP['ep2']:.2f}"
            )
            if e1 >= 0.50:
                print(
                    "  → ep1 still STRONG without deplete. "
                    "Deplete is NOT required for roll's early sMNIST signal."
                )
            elif e1 >= 0.35:
                print(
                    "  → ep1 moderate; deplete may help early but roll write still works."
                )
            else:
                print(
                    "  → ep1 WEAK without deplete. "
                    "Deplete may be load-bearing for roll early on sMNIST."
                )
            print("  (You can stop the run now if that's enough.)\n")

    ep1 = history[0]
    ep2 = history[1] if len(history) > 1 else float("nan")
    print("\n" + "=" * 72)
    print("SUMMARY  roll_nodep sMNIST  (deplete=OFF)")
    print("=" * 72)
    print(
        f"roll_nodep  best={best_acc:.3f} @ep{best_ep}  "
        f"ep1={ep1:.3f}  ep2={ep2:.3f}  pipe@best≈{best_pipe:.2f}  "
        f"params={n_params}  t={time.time() - t0:.0f}s"
    )
    print(
        f"roll_dep (exp08 ref)  best≈{REF_ROLL_DEP['best']:.2f}  "
        f"ep1≈{REF_ROLL_DEP['ep1']:.2f}  ep2≈{REF_ROLL_DEP['ep2']:.2f}"
    )
    print(
        f"Δ (nodep − dep ref):  ep1={ep1 - REF_ROLL_DEP['ep1']:+.3f}  "
        f"ep2={ep2 - REF_ROLL_DEP['ep2']:+.3f}  "
        f"best={best_acc - REF_ROLL_DEP['best']:+.3f}"
    )
    print(
        "Law draft:\n"
        "  If ep1 still ~0.6+ without deplete → roll win is the WRITE, not subtract.\n"
        "  Deplete remains optional hygiene (pipe), not the reason roll crushes bag."
    )
    print("DONE — paste SUMMARY (or just ep1/ep2 lines) back.")
    return history


if __name__ == "__main__":
    main()
