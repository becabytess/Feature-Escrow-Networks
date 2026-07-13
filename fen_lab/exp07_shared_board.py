# ==============================================================================
# fen_lab / EXP07 — Shared escrow as a communication board (dual experts)
# ==============================================================================
# Claim under test
#   Escrow is not only a private notebook for one RNN. It can act as a *shared
#   board / gateway*: multiple pipes commit into one E; the joint task needs
#   facts that no single worker saw alone.
#
# Task: PARTITIONED distracted counting
#   Two input streams (same T):
#     stream A — static ID at t=0, then noise (no count events)
#     stream B — count +/- events + distractors, NO ID pulse
#   Label: ID × count_bin (30-way), same as classic distracted.
#
#   A mono model on concat(A,B) can still see both (upper bound / cheat path).
#   Dual models: expert A only sees A, expert B only sees B.
#
# Models (~15k total params, matched)
#   mono_residual       one residual on concat(A,B)
#   mono_fen_bag        one bag FEN on concat(A,B)
#   dual_residual       two residuals, no E; head([hA,hB])
#   dual_fen_private    two bag FENs, private EA,EB; head([hA,hB,EA,EB])
#   dual_fen_shared ★   two pipes, ONE shared bag E; both commit+deplete
#                       head([hA,hB,E])
#   dual_fen_shared_rr  shared E but every-step reinject E→ each pipe (pollution)
#
# Predictions
#   dual_fen_shared should beat dual_residual and dual_fen_private if the board
#   is needed to exchange ID ↔ count.
#   mono_fen_bag is a soft upper bound (sees both streams).
#   shared_rr: fat pipes / worse — continuous mix is not "communication".
#
# Colab: paste whole file → GPU → Run. Deps: torch, numpy.
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
    SEEDS = [1]
    EPOCHS = 15
    TRAIN_N, TEST_N = 4000, 1000
    PRINT_EVERY = 3
else:
    SEEDS = [1, 2, 3]
    EPOCHS = 25
    TRAIN_N, TEST_N = 8000, 2000
    PRINT_EVERY = 5

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0
SEQ_LEN = 96
NOISE_STD = 0.45

TARGET_PARAMS = 15000
AUTO_MATCH_PARAMS = True
MIN_H, MAX_H = 8, 96  # dual models use 2× cores → cap width

MODEL_ORDER = [
    "mono_residual",
    "mono_fen_bag",
    "dual_residual",
    "dual_fen_private",
    "dual_fen_shared",
    "dual_fen_shared_rr",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Device: {DEVICE} | FAST_MODE={FAST_MODE} | SEEDS={SEEDS} | "
    f"EPOCHS={EPOCHS}"
)
print(
    "EXP07 — Shared-board dual experts on PARTITIONED distracted "
    "(ID stream A | count stream B | shared E gateway)"
)
print(f"Models: {MODEL_ORDER}")


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


# ------------------------------ DATA ------------------------------------------
def make_partitioned_distracted(n, seed):
    """
    Classic distracted physics, but split across two streams.

    stream A (input_dim): ID one-hot pulse at t=0 + noise. No count ops.
    stream B (input_dim): count +/- and distractors + noise. No ID pulse.

    y = id * n_bins + count_bin  (30-way)
    """
    rng = np.random.default_rng(seed)
    n_id, n_bins = 10, 3
    op_dims, noise_dims = 4, 16
    # A: id channels + noise | B: op channels + noise  (same width for simplicity)
    dim_a = n_id + noise_dims
    dim_b = op_dims + noise_dims

    A = rng.normal(0.0, NOISE_STD, size=(n, SEQ_LEN, dim_a)).astype(np.float32)
    B = rng.normal(0.0, NOISE_STD, size=(n, SEQ_LEN, dim_b)).astype(np.float32)
    # damp structured channels
    A[:, :, :n_id] *= 0.10
    B[:, :, :op_dims] *= 0.10
    y = np.zeros((n,), dtype=np.int64)

    plus_dim, minus_dim = 0, 1
    distract_a, distract_b = 2, 3

    for i in range(n):
        static_id = int(rng.integers(0, n_id))
        count_bin = int(rng.integers(0, n_bins))
        # ID only on stream A at t=0
        A[i, 0, static_id] += 2.0

        possible = np.arange(1, SEQ_LEN)
        n_events = int(rng.integers(18, 31))
        positions = rng.choice(possible, size=n_events, replace=False)
        p_plus = [0.25, 0.50, 0.75][count_bin]
        n_plus = int(round(n_events * p_plus))
        B[i, positions[:n_plus], plus_dim] += 1.5
        B[i, positions[n_plus:], minus_dim] += 1.5

        n_distract = int(rng.integers(10, 25))
        dpos = rng.choice(possible, size=n_distract, replace=False)
        half = n_distract // 2
        B[i, dpos[:half], distract_a] += 1.25
        B[i, dpos[half:], distract_b] += 1.25

        y[i] = static_id * n_bins + count_bin

    meta = {
        "task": "partitioned_distracted",
        "dim_a": dim_a,
        "dim_b": dim_b,
        "dim_cat": dim_a + dim_b,
        "output_dim": n_id * n_bins,
        "n_id": n_id,
        "n_bins": n_bins,
    }
    return (
        torch.tensor(A),
        torch.tensor(B),
        torch.tensor(y),
        meta,
    )


class TwinDataset(torch.utils.data.Dataset):
    def __init__(self, A, B, y):
        self.A, self.B, self.y = A, B, y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.A[i], self.B[i], self.y[i]


# ------------------------------ BLOCKS ----------------------------------------
class ResidualCore(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hdim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)

    def step(self, h, x_t):
        z = h + self.x_proj(x_t)
        return torch.tanh(self.core(z) + z)

    def scan(self, x, h0=None):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim) if h0 is None else h0
        for t in range(T):
            h = self.step(h, x[:, t])
        return h


class FENBagCore(nn.Module):
    """One pipe + bag write into a provided escrow tensor E (shared or private)."""

    def __init__(self, input_dim, hidden_dim, reinject=False):
        super().__init__()
        self.hdim = hidden_dim
        self.reinject = reinject
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        if reinject:
            self.rj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.rj = None

    def step(self, h, E, x_t):
        z = h + self.x_proj(x_t)
        f = torch.tanh(self.core(z) + z)
        g = torch.sigmoid(self.gate(f))
        D = g * f
        h = f - D
        E = E + self.escrow_proj(D)
        if self.rj is not None:
            h = h + torch.tanh(self.rj(E))
        return h, E, g


# ------------------------------ MODELS ----------------------------------------
class MonoResidual(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim, hidden_dim):
        super().__init__()
        self.core = ResidualCore(dim_a + dim_b, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, a, b, return_stats=False):
        x = torch.cat([a, b], dim=-1)
        h = self.core.scan(x)
        logits = self.head(h)
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "pipe_a": float("nan"),
                "pipe_b": float("nan"),
                "escrow_norm": float("nan"),
                "gate": float("nan"),
            }
        return logits


class MonoFENBag(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim, hidden_dim):
        super().__init__()
        self.core = FENBagCore(dim_a + dim_b, hidden_dim, reinject=False)
        self.hdim = hidden_dim
        self.head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, a, b, return_stats=False):
        x = torch.cat([a, b], dim=-1)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        g_sum = 0.0
        for t in range(T):
            h, E, g = self.core.step(h, E, x[:, t])
            if return_stats:
                g_sum += float(g.detach().mean())
        logits = self.head(torch.cat([h, E], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "pipe_a": float("nan"),
                "pipe_b": float("nan"),
                "escrow_norm": float(E.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
            }
        return logits


class DualResidual(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim, hidden_dim):
        super().__init__()
        self.core_a = ResidualCore(dim_a, hidden_dim)
        self.core_b = ResidualCore(dim_b, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, a, b, return_stats=False):
        h_a = self.core_a.scan(a)
        h_b = self.core_b.scan(b)
        logits = self.head(torch.cat([h_a, h_b], dim=-1))
        if return_stats:
            pn = 0.5 * (
                float(h_a.detach().norm(dim=-1).mean())
                + float(h_b.detach().norm(dim=-1).mean())
            )
            return logits, {
                "pipe_norm": pn,
                "pipe_a": float(h_a.detach().norm(dim=-1).mean()),
                "pipe_b": float(h_b.detach().norm(dim=-1).mean()),
                "escrow_norm": float("nan"),
                "gate": float("nan"),
            }
        return logits


class DualFENPrivate(nn.Module):
    """Two bag FENs, separate escrows — no shared board."""

    def __init__(self, dim_a, dim_b, out_dim, hidden_dim):
        super().__init__()
        self.hdim = hidden_dim
        self.core_a = FENBagCore(dim_a, hidden_dim, reinject=False)
        self.core_b = FENBagCore(dim_b, hidden_dim, reinject=False)
        self.head = nn.Linear(hidden_dim * 4, out_dim)  # hA,hB,EA,EB

    def forward(self, a, b, return_stats=False):
        B, T, _ = a.shape
        h_a = a.new_zeros(B, self.hdim)
        h_b = b.new_zeros(B, self.hdim)
        E_a = a.new_zeros(B, self.hdim)
        E_b = b.new_zeros(B, self.hdim)
        g_sum = 0.0
        for t in range(T):
            h_a, E_a, ga = self.core_a.step(h_a, E_a, a[:, t])
            h_b, E_b, gb = self.core_b.step(h_b, E_b, b[:, t])
            if return_stats:
                g_sum += 0.5 * (float(ga.detach().mean()) + float(gb.detach().mean()))
        logits = self.head(torch.cat([h_a, h_b, E_a, E_b], dim=-1))
        if return_stats:
            pn = 0.5 * (
                float(h_a.detach().norm(dim=-1).mean())
                + float(h_b.detach().norm(dim=-1).mean())
            )
            en = 0.5 * (
                float(E_a.detach().norm(dim=-1).mean())
                + float(E_b.detach().norm(dim=-1).mean())
            )
            return logits, {
                "pipe_norm": pn,
                "pipe_a": float(h_a.detach().norm(dim=-1).mean()),
                "pipe_b": float(h_b.detach().norm(dim=-1).mean()),
                "escrow_norm": en,
                "gate": g_sum / T,
            }
        return logits


class DualFENShared(nn.Module):
    """
    Two pipes, ONE shared bag escrow E.
    Both experts commit+deplete into the same board.
    reinject=True → every-step E→h on both pipes (pollution control).
    """

    def __init__(self, dim_a, dim_b, out_dim, hidden_dim, reinject=False):
        super().__init__()
        self.hdim = hidden_dim
        self.reinject = reinject
        self.core_a = FENBagCore(dim_a, hidden_dim, reinject=reinject)
        self.core_b = FENBagCore(dim_b, hidden_dim, reinject=reinject)
        self.head = nn.Linear(hidden_dim * 3, out_dim)  # hA, hB, E

    def forward(self, a, b, return_stats=False):
        B, T, _ = a.shape
        h_a = a.new_zeros(B, self.hdim)
        h_b = b.new_zeros(B, self.hdim)
        E = a.new_zeros(B, self.hdim)
        g_sum = 0.0
        for t in range(T):
            # sequential commit into shared board (A then B at each t)
            h_a, E, ga = self.core_a.step(h_a, E, a[:, t])
            h_b, E, gb = self.core_b.step(h_b, E, b[:, t])
            if return_stats:
                g_sum += 0.5 * (float(ga.detach().mean()) + float(gb.detach().mean()))
        logits = self.head(torch.cat([h_a, h_b, E], dim=-1))
        if return_stats:
            pn = 0.5 * (
                float(h_a.detach().norm(dim=-1).mean())
                + float(h_b.detach().norm(dim=-1).mean())
            )
            return logits, {
                "pipe_norm": pn,
                "pipe_a": float(h_a.detach().norm(dim=-1).mean()),
                "pipe_b": float(h_b.detach().norm(dim=-1).mean()),
                "escrow_norm": float(E.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
            }
        return logits


# ------------------------------ BUILD / MATCH ---------------------------------
def build(name, meta, hidden_dim):
    da, db, out = meta["dim_a"], meta["dim_b"], meta["output_dim"]
    if name == "mono_residual":
        return MonoResidual(da, db, out, hidden_dim)
    if name == "mono_fen_bag":
        return MonoFENBag(da, db, out, hidden_dim)
    if name == "dual_residual":
        return DualResidual(da, db, out, hidden_dim)
    if name == "dual_fen_private":
        return DualFENPrivate(da, db, out, hidden_dim)
    if name == "dual_fen_shared":
        return DualFENShared(da, db, out, hidden_dim, reinject=False)
    if name == "dual_fen_shared_rr":
        return DualFENShared(da, db, out, hidden_dim, reinject=True)
    raise ValueError(name)


_HIDDEN_CACHE = {}


def choose_hidden(name, meta):
    key = (name, meta["dim_a"], meta["dim_b"], meta["output_dim"], TARGET_PARAMS)
    if key in _HIDDEN_CACHE:
        return _HIDDEN_CACHE[key]
    if not AUTO_MATCH_PARAMS:
        _HIDDEN_CACHE[key] = 32
        return 32

    best_h, best_diff = 32, float("inf")
    for h in range(MIN_H, MAX_H + 1):
        n = count_params(build(name, meta, h))
        d = abs(n - TARGET_PARAMS)
        if d < best_diff:
            best_h, best_diff = h, d
    _HIDDEN_CACHE[key] = best_h
    return best_h


# ------------------------------ TRAIN -----------------------------------------
def loss_and_metrics(logits, y, meta):
    loss = nn.functional.cross_entropy(logits, y)
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    n_bins = meta["n_bins"]
    id_acc = (pred // n_bins == y // n_bins).float().mean().item()
    count_acc = (pred % n_bins == y % n_bins).float().mean().item()
    return loss, {"acc": acc, "id_acc": id_acc, "count_acc": count_acc}


@torch.no_grad()
def evaluate(model, loader, meta):
    model.eval()
    totals = defaultdict(float)
    n = 0
    stats_acc = defaultdict(float)
    n_stats = 0
    for a, b, y in loader:
        a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
        out = model(a, b, return_stats=True)
        logits, stats = out
        for k, v in stats.items():
            if isinstance(v, (int, float)) and v == v:
                stats_acc[k] += float(v)
        n_stats += 1
        loss, metrics = loss_and_metrics(logits, y, meta)
        bs = y.size(0)
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
    best = {"acc": -1.0, "epoch": 0}
    t0 = time.time()
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        for a, b, y in train_loader:
            a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(a, b)
            loss, _ = loss_and_metrics(logits, y, meta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val = evaluate(model, test_loader, meta)
        history.append(val)
        if val["acc"] > best["acc"]:
            best = {
                "acc": val["acc"],
                "id_acc": val.get("id_acc", 0.0),
                "count_acc": val.get("count_acc", 0.0),
                "pipe_norm": val.get("pipe_norm", float("nan")),
                "pipe_a": val.get("pipe_a", float("nan")),
                "pipe_b": val.get("pipe_b", float("nan")),
                "escrow_norm": val.get("escrow_norm", float("nan")),
                "gate": val.get("gate", float("nan")),
                "epoch": ep,
            }

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            print(
                f"    ep {ep:02d}/{epochs}  loss={val['loss']:.4f}  "
                f"acc={val['acc']:.3f}  id={val['id_acc']:.3f}  "
                f"count={val['count_acc']:.3f}  "
                f"pipe={val.get('pipe_norm', float('nan')):.2f}  "
                f"E={val.get('escrow_norm', float('nan')):.2f}"
            )

    best["time_s"] = time.time() - t0
    target = 0.9 * max(best["acc"], 1e-8)
    best["to90"] = epochs
    for i, h in enumerate(history, 1):
        if h["acc"] >= target:
            best["to90"] = i
            break
    return best


def main():
    print(
        f"Task=partitioned_distracted | T={SEQ_LEN} | "
        f"target_params≈{TARGET_PARAMS} | train_n={TRAIN_N} test_n={TEST_N}"
    )
    print(
        "Partition: expert A sees ID only | expert B sees count only | "
        "shared E is the gateway hypothesis."
    )

    Atr, Btr, ytr, meta = make_partitioned_distracted(TRAIN_N, seed=1000)
    Ate, Bte, yte, _ = make_partitioned_distracted(TEST_N, seed=2000)
    print(
        f"  dim_a={meta['dim_a']} dim_b={meta['dim_b']} "
        f"out={meta['output_dim']} classes (ID×count)"
    )

    summary = defaultdict(list)

    for model_name in MODEL_ORDER:
        print(f"\n--- Model: {model_name} ---")
        hdim = choose_hidden(model_name, meta)
        probe = build(model_name, meta, hdim)
        nparams = count_params(probe)
        print(f"  hidden={hdim}  params={nparams}")

        for seed in SEEDS:
            seed_everything(seed)
            model = build(model_name, meta, hdim).to(DEVICE)
            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                TwinDataset(Atr, Btr, ytr),
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=g,
            )
            test_loader = DataLoader(
                TwinDataset(Ate, Bte, yte),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            print(f"  seed={seed}")
            best = train_one(model, train_loader, test_loader, meta, EPOCHS)
            summary[model_name].append(best)
            print(
                f"  >> best acc={best['acc']:.3f}  id={best['id_acc']:.3f}  "
                f"count={best['count_acc']:.3f}  @ep{best['epoch']}  "
                f"pipe={best['pipe_norm']:.2f}  E={best['escrow_norm']:.2f}  "
                f"t={best['time_s']:.1f}s  to90%={best['to90']}"
            )

    print("\n" + "-" * 80)
    print(
        f"SUMMARY  task=partitioned_distracted  seeds={SEEDS}  "
        f"target_params≈{TARGET_PARAMS}"
    )
    print("-" * 80)
    print(
        f"{'model':<22} {'acc':>7} {'id':>7} {'count':>7} "
        f"{'pipe':>7} {'E':>7} {'to_best':>8}"
    )
    for name in MODEL_ORDER:
        rows = summary[name]
        acc = np.mean([r["acc"] for r in rows])
        ida = np.mean([r["id_acc"] for r in rows])
        cta = np.mean([r["count_acc"] for r in rows])
        pipe = np.nanmean([r["pipe_norm"] for r in rows])
        esc = np.nanmean([r["escrow_norm"] for r in rows])
        epb = np.mean([r["epoch"] for r in rows])
        print(
            f"{name:<22} {acc:7.3f} {ida:7.3f} {cta:7.3f} "
            f"{pipe:7.2f} {esc:7.2f} {epb:8.1f}"
        )
    print("-" * 80)
    print(
        "How to read:\n"
        "  • dual_fen_shared ★  — shared board; want high id AND count\n"
        "  • dual_fen_private   — two notebooks, no bus; if worse, board matters\n"
        "  • dual_residual      — two pipes, no escrow\n"
        "  • mono_fen_bag       — sees concat(A,B); soft upper bound\n"
        "  • dual_fen_shared_rr — shared E + every-step reinject; expect fat pipe\n"
        "  Floors: joint ~0.1 or id/count one-sided = no real communication."
    )
    print("DONE — paste this SUMMARY back for scoring.")
    return summary


if __name__ == "__main__":
    main()
