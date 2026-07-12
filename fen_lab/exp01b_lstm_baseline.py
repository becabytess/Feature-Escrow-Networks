# ==============================================================================
# fen_lab / EXP01b — LSTM on the foundation synthetic tasks
# ==============================================================================
# Gap fix: exp01–04 compared FEN variants + residual but NEVER ran LSTM on
# the two probes that define the FEN claim:
#   1) recall5    — exact ordered delayed recall of 5 symbols
#   2) distracted — static ID + noisy counting (dual-role / overwrite)
#
# Same data generators, ~15k param match, same metrics as exp01.
# Models (minimal decisive set):
#   residual_rnn  — residual tanh RNN, no escrow
#   fen_bag       — classic FEN (should nail distracted, fail recall exact)
#   fen_slot      — hard slot FEN (should nail recall, fail distracted ID)
#   lstm          — classical baseline (expect ~0 exact on recall; weak dual-role)
#
# Primary metrics:
#   recall5    -> exact  (full sequence correct; NOT token acc alone)
#   distracted -> acc, id_acc, count_acc
#
# Colab: paste whole file → GPU → Run.
# Deps: torch, numpy. No downloads.
# ==============================================================================

import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------ CONFIG ----------------------------------------
FAST_MODE = True

if FAST_MODE:
    TASKS = ["recall5", "distracted"]
    SEEDS = [1]
    EPOCHS_RECALL = 25
    EPOCHS_DISTRACTED = 12
    TRAIN_N, TEST_N = 4000, 1000
    PRINT_EVERY = 5
else:
    TASKS = ["recall5", "distracted"]
    SEEDS = [1, 2, 3]
    EPOCHS_RECALL = 40
    EPOCHS_DISTRACTED = 20
    TRAIN_N, TEST_N = 8000, 2000
    PRINT_EVERY = 5

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0
SEQ_LEN = 96
NOISE_STD = 0.45

TARGET_PARAMS = 15000
AUTO_MATCH_PARAMS = True
MIN_H, MAX_H = 8, 128
TAPE_K = 8

# Decisive set only — not another FEN zoo
MODEL_ORDER = ["residual_rnn", "fen_bag", "fen_slot", "lstm"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Device: {DEVICE} | FAST_MODE={FAST_MODE} | "
    f"SEEDS={SEEDS} | Models={MODEL_ORDER}"
)
print(
    "EXP01b — LSTM vs residual / fen_bag / fen_slot on foundation synthetic tasks"
)
print("Metrics: recall5=exact sequence | distracted=acc + id + count")


# ------------------------------ UTILS -----------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------ DATA (same as exp01) --------------------------
def make_distracted(n, seed):
    """Static ID at t=0 + active +/- counting with distractors. 30-way class."""
    rng = np.random.default_rng(seed)
    n_id, n_bins = 10, 3
    op_dims, noise_dims = 4, 16
    input_dim = n_id + op_dims + noise_dims

    X = rng.normal(0.0, NOISE_STD, size=(n, SEQ_LEN, input_dim)).astype(np.float32)
    X[:, :, : n_id + op_dims] *= 0.10
    y = np.zeros((n,), dtype=np.int64)

    plus_dim, minus_dim = n_id, n_id + 1
    distract_a, distract_b = n_id + 2, n_id + 3

    for i in range(n):
        static_id = int(rng.integers(0, n_id))
        count_bin = int(rng.integers(0, n_bins))
        X[i, 0, static_id] += 2.0

        possible = np.arange(1, SEQ_LEN)
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
        "output_dim": n_id * n_bins,
        "n_id": n_id,
        "n_bins": n_bins,
    }
    return torch.tensor(X), torch.tensor(y), meta


def make_recall5(n, seed, random_positions=False):
    """Delayed recall of 5 symbols in chronological order."""
    rng = np.random.default_rng(seed)
    vocab, slots, noise_dims = 10, 5, 16
    input_dim = vocab + noise_dims

    X = rng.normal(0.0, NOISE_STD, size=(n, SEQ_LEN, input_dim)).astype(np.float32)
    X[:, :, :vocab] *= 0.10
    y = np.zeros((n, slots), dtype=np.int64)

    for i in range(n):
        symbols = rng.integers(0, vocab, size=slots)
        y[i] = symbols
        if random_positions:
            positions = np.sort(rng.choice(np.arange(SEQ_LEN), size=slots, replace=False))
        else:
            positions = np.arange(slots)
        for j, pos in enumerate(positions):
            X[i, int(pos), int(symbols[j])] += 2.0

        n_distract = int(rng.integers(12, 25))
        blocked = set(int(p) for p in positions)
        possible = np.array([t for t in range(SEQ_LEN) if t not in blocked])
        dpos = rng.choice(possible, size=n_distract, replace=False)
        dsym = rng.integers(0, vocab, size=n_distract)
        for pos, sym in zip(dpos, dsym):
            X[i, int(pos), int(sym)] += 0.75

    meta = {
        "task": "recall5",
        "input_dim": input_dim,
        "output_dim": slots * vocab,
        "vocab": vocab,
        "slots": slots,
    }
    return torch.tensor(X), torch.tensor(y), meta


def make_dataset(task, n, seed):
    if task == "distracted":
        return make_distracted(n, seed)
    if task == "recall5":
        return make_recall5(n, seed, random_positions=False)
    raise ValueError(task)


# ------------------------------ MODELS ----------------------------------------
class ResidualRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.h = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = torch.zeros(B, self.h, device=x.device)
        for t in range(T):
            z = h + self.x_proj(x[:, t])
            h = torch.tanh(self.core(z) + z)
        logits = self.head(h)
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": 0.0,
            }
        return logits


class FENBag(nn.Module):
    """Classic subtractive FEN + additive bag escrow."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.h = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = torch.zeros(B, self.h, device=x.device)
        E = torch.zeros(B, self.h, device=x.device)
        g_sum = 0.0
        for t in range(T):
            z = h + self.x_proj(x[:, t])
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            h = f - D
            E = E + self.escrow_proj(D)
            if return_stats:
                g_sum += float(g.detach().mean())
        logits = self.head(torch.cat([h, E], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
            }
        return logits


class FENSlot(nn.Module):
    """Hard write-pointer slot escrow (order upper bound from exp01/02)."""

    def __init__(self, input_dim, output_dim, hidden_dim, K=TAPE_K):
        super().__init__()
        self.h = hidden_dim
        self.K = K
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tape_pool = nn.Linear(K * hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = torch.zeros(B, self.h, device=x.device)
        E = torch.zeros(B, self.K, self.h, device=x.device)
        ptr = torch.zeros(B, dtype=torch.long, device=x.device)
        g_sum = 0.0
        for t in range(T):
            z = h + self.x_proj(x[:, t])
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            h = f - D
            v = self.escrow_proj(D)
            one = torch.nn.functional.one_hot(ptr, self.K).to(dtype=v.dtype)
            E = E + one.unsqueeze(-1) * v.unsqueeze(1)
            advance = (g.mean(dim=-1) > 0.25).long()
            ptr = (ptr + advance) % self.K
            if return_stats:
                g_sum += float(g.detach().mean())
        E_vec = torch.tanh(self.tape_pool(E.reshape(B, -1)))
        logits = self.head(torch.cat([h, E_vec], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
            }
        return logits


class LSTMBaseline(nn.Module):
    """
    1-layer LSTM → final hidden → linear head.
    Same input/output contract as residual/FEN (flat recall logits).
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.h = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_stats=False):
        out, (h_n, c_n) = self.lstm(x)
        h = h_n[-1]  # [B, H]
        logits = self.head(h)
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": 0.0,
                "cell_norm": float(c_n[-1].detach().norm(dim=-1).mean()),
            }
        return logits


MODEL_CTORS = {
    "residual_rnn": lambda idim, odim, h: ResidualRNN(idim, odim, h),
    "fen_bag": lambda idim, odim, h: FENBag(idim, odim, h),
    "fen_slot": lambda idim, odim, h: FENSlot(idim, odim, h, K=TAPE_K),
    "lstm": lambda idim, odim, h: LSTMBaseline(idim, odim, h),
}


def choose_hidden(name, input_dim, output_dim):
    if not AUTO_MATCH_PARAMS:
        return 48
    best_h, best_diff = 48, float("inf")
    for h in range(MIN_H, MAX_H + 1):
        m = MODEL_CTORS[name](input_dim, output_dim, h)
        diff = abs(count_params(m) - TARGET_PARAMS)
        if diff < best_diff:
            best_h, best_diff = h, diff
    return best_h


def build(name, input_dim, output_dim, hidden_dim):
    return MODEL_CTORS[name](input_dim, output_dim, hidden_dim)


# ------------------------------ LOSS / METRICS --------------------------------
def loss_and_metrics(logits, y, meta):
    if meta["task"] == "distracted":
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        n_bins = meta["n_bins"]
        id_acc = (pred // n_bins == y // n_bins).float().mean().item()
        count_acc = (pred % n_bins == y % n_bins).float().mean().item()
        return loss, {"acc": acc, "id_acc": id_acc, "count_acc": count_acc, "exact": acc}

    slots, vocab = meta["slots"], meta["vocab"]
    B = logits.size(0)
    logits_s = logits.view(B, slots, vocab)
    loss = 0.0
    for j in range(slots):
        loss = loss + nn.functional.cross_entropy(logits_s[:, j], y[:, j])
    loss = loss / slots
    pred = logits_s.argmax(dim=-1)
    token_acc = (pred == y).float().mean().item()
    exact = (pred == y).all(dim=-1).float().mean().item()
    return loss, {"acc": token_acc, "exact": exact, "token_acc": token_acc}


@torch.no_grad()
def evaluate(model, loader, meta):
    model.eval()
    totals = defaultdict(float)
    n = 0
    stats_acc = defaultdict(float)
    n_stats = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb, return_stats=True)
        if isinstance(out, tuple):
            logits, stats = out
            for k, v in stats.items():
                stats_acc[k] += float(v)
            n_stats += 1
        else:
            logits = out
        loss, metrics = loss_and_metrics(logits, yb, meta)
        bs = xb.size(0)
        totals["loss"] += loss.item() * bs
        for k, v in metrics.items():
            totals[k] += v * bs
        n += bs
    out = {k: v / n for k, v in totals.items()}
    if n_stats:
        for k, v in stats_acc.items():
            out[k] = v / n_stats
    return out


def train_one(model, train_loader, test_loader, meta, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    best = {"exact": -1.0, "acc": -1.0, "epoch": 0}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss, _ = loss_and_metrics(logits, yb, meta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val = evaluate(model, test_loader, meta)
        history.append(val)
        score = val.get("exact", val.get("acc", 0.0))
        if score > best["exact"]:
            best = {
                "exact": score,
                "acc": val.get("acc", score),
                "epoch": ep,
                **{k: val[k] for k in val if k not in ("loss",)},
            }

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            if meta["task"] == "distracted":
                extra = (
                    f" id={val.get('id_acc', 0):.3f} "
                    f"count={val.get('count_acc', 0):.3f}"
                )
            else:
                extra = (
                    f" exact={val.get('exact', 0):.3f} "
                    f"token={val.get('token_acc', 0):.3f}"
                )
            pn = val.get("pipe_norm", float("nan"))
            print(
                f"    ep {ep:02d}/{epochs}  loss={val['loss']:.4f}  "
                f"acc={val.get('acc', 0):.3f}{extra}  pipe={pn:.2f}"
            )

    elapsed = time.time() - t0
    target = 0.9 * max(best["exact"], 1e-8)
    first_good = None
    for i, h in enumerate(history, 1):
        if h.get("exact", h.get("acc", 0)) >= target:
            first_good = i
            break
    best["time_s"] = elapsed
    best["epochs_to_90pct_best"] = first_good if first_good is not None else epochs
    return best, history


# ------------------------------ MAIN ------------------------------------------
def run_task(task: str, epochs: int):
    print("\n" + "=" * 78)
    print(f"TASK: {task}  epochs={epochs}")
    print("=" * 78)

    Xtr, ytr, meta = make_dataset(task, TRAIN_N, seed=1000)
    Xte, yte, _ = make_dataset(task, TEST_N, seed=2000)
    summary = defaultdict(list)

    for model_name in MODEL_ORDER:
        print(f"\n--- Model: {model_name} ---")
        hdim = choose_hidden(model_name, meta["input_dim"], meta["output_dim"])
        probe = build(model_name, meta["input_dim"], meta["output_dim"], hdim)
        nparams = count_params(probe)
        print(f"  hidden={hdim}  params={nparams}")

        for seed in SEEDS:
            seed_everything(seed)
            model = build(
                model_name, meta["input_dim"], meta["output_dim"], hdim
            ).to(DEVICE)

            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                TensorDataset(Xtr, ytr),
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=g,
            )
            test_loader = DataLoader(
                TensorDataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False
            )

            print(f"  seed={seed}")
            best, _ = train_one(model, train_loader, test_loader, meta, epochs)
            summary[model_name].append(best)

            if meta["task"] == "recall5":
                print(
                    f"  >> best exact={best['exact']:.3f}  token={best['acc']:.3f}  "
                    f"@ep{best['epoch']}  t={best['time_s']:.1f}s  "
                    f"to90%={best['epochs_to_90pct_best']}"
                )
            else:
                print(
                    f"  >> best acc={best['acc']:.3f}  id={best.get('id_acc', 0):.3f}  "
                    f"count={best.get('count_acc', 0):.3f}  @ep{best['epoch']}  "
                    f"t={best['time_s']:.1f}s  to90%={best['epochs_to_90pct_best']}"
                )

    print("\n" + "-" * 78)
    print(f"SUMMARY  task={task}  seeds={SEEDS}  target_params≈{TARGET_PARAMS}")
    print("-" * 78)
    if task == "recall5":
        print(f"{'model':<16} {'exact':>8} {'token':>8} {'to_best':>8} {'to90%':>8} {'pipe':>8}")
        for name, runs in summary.items():
            exact = np.array([r["exact"] for r in runs], dtype=np.float64)
            token = np.array([r["acc"] for r in runs], dtype=np.float64)
            epb = np.array([r["epoch"] for r in runs], dtype=np.float64)
            spd = np.array([r["epochs_to_90pct_best"] for r in runs], dtype=np.float64)
            pipes = np.array(
                [r.get("pipe_norm", float("nan")) for r in runs], dtype=np.float64
            )
            print(
                f"{name:<16} {exact.mean():8.3f} {token.mean():8.3f} "
                f"{epb.mean():8.1f} {spd.mean():8.1f} {np.nanmean(pipes):8.2f}"
            )
    else:
        print(
            f"{'model':<16} {'acc':>8} {'id':>8} {'count':>8} "
            f"{'to_best':>8} {'to90%':>8} {'pipe':>8}"
        )
        for name, runs in summary.items():
            acc = np.array([r["acc"] for r in runs], dtype=np.float64)
            ida = np.array([r.get("id_acc", 0) for r in runs], dtype=np.float64)
            cta = np.array([r.get("count_acc", 0) for r in runs], dtype=np.float64)
            epb = np.array([r["epoch"] for r in runs], dtype=np.float64)
            spd = np.array([r["epochs_to_90pct_best"] for r in runs], dtype=np.float64)
            pipes = np.array(
                [r.get("pipe_norm", float("nan")) for r in runs], dtype=np.float64
            )
            print(
                f"{name:<16} {acc.mean():8.3f} {ida.mean():8.3f} {cta.mean():8.3f} "
                f"{epb.mean():8.1f} {spd.mean():8.1f} {np.nanmean(pipes):8.2f}"
            )

    print(
        "How to read: floors matter. recall exact~0 or distracted acc~0.1 = "
        "cannot learn (not 'slightly worse')."
    )
    return summary


def main():
    print(
        f"Tasks={TASKS} | recall_ep={EPOCHS_RECALL} | "
        f"distracted_ep={EPOCHS_DISTRACTED} | T={SEQ_LEN}"
    )
    all_results = {}
    for task in TASKS:
        epochs = EPOCHS_RECALL if task == "recall5" else EPOCHS_DISTRACTED
        all_results[task] = run_task(task, epochs)

    print("\n" + "=" * 78)
    print("DONE — paste BOTH SUMMARY tables back.")
    print("Expected sketch (hypotheses, not guarantees):")
    print("  recall5 exact:  fen_slot high | fen_bag / residual / lstm ~0")
    print("  distracted:     fen_bag high  | fen_slot weak ID | residual weak")
    print("                  lstm = the missing number — score it honestly")
    print("=" * 78)
    return all_results


if __name__ == "__main__":
    main()
