# ==============================================================================
# fen_lab / EXP06 — Multi-pass read of escrow (discrete read events)
# ==============================================================================
# Motivation
#   exp04: every-step reinject of E → h is bad (pipe bloat, dual-role dies).
#   Legal use of E is *read* (mid/final head([h,E])), not continuous mix into the pipe.
#
#   Question: after a full scan that *fills* escrow, can we do a *second* pass over
#   the same sequence conditioned on a *once-read* summary of E — so the model can
#   *act on* what it read — without recreating every-step pollution?
#
# Task: distracted counting only (dual-role probe; foundation for FEN)
#   T=96, ~15k params, same generator as exp01 / exp01b.
#
# Models
#   residual_1pass     residual RNN, one scan
#   residual_2pass     residual RNN, two scans (compute control; no escrow)
#   fen_bag_1pass      classic bag FEN, one scan, final head([h,E])
#   fen_bag_2pass_cold pass1 fill E; c=read(E); h←0; pass2 FEN with c as fixed context
#   fen_bag_2pass_warm same, but h←init(c) once between passes
#   fen_bag_reinject   every-step dump of E into h (should lose — exp04 control)
#
# Predictions
#   fen_bag_1pass strong; reinject worse / fatter pipe
#   2pass cold/warm ≥ 1pass if discrete read helps; residual_2pass << fen 2pass
#   2pass should keep leaner pipe than reinject
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
MIN_H, MAX_H = 8, 128

MODEL_ORDER = [
    "residual_1pass",
    "residual_2pass",
    "fen_bag_1pass",
    "fen_bag_2pass_cold",
    "fen_bag_2pass_warm",
    "fen_bag_reinject",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Device: {DEVICE} | FAST_MODE={FAST_MODE} | SEEDS={SEEDS} | "
    f"EPOCHS={EPOCHS} | Models={MODEL_ORDER}"
)
print(
    "EXP06 — Multi-pass escrow read on distracted counting "
    "(discrete read between passes vs every-step reinject)"
)


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


# ------------------------------ MODELS ----------------------------------------
class ResidualPass(nn.Module):
    """
    Residual tanh RNN.
    n_passes=1: one scan of x
    n_passes=2: full second scan (no escrow; pure compute control)
    """

    def __init__(self, input_dim, output_dim, hidden_dim, n_passes=1):
        super().__init__()
        assert n_passes in (1, 2)
        self.hdim = hidden_dim
        self.n_passes = n_passes
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def _scan(self, xp, h):
        B, T, _ = xp.shape
        for t in range(T):
            z = h + xp[:, t]
            h = torch.tanh(self.core(z) + z)
        return h

    def forward(self, x, return_stats=False):
        xp = self.x_proj(x)
        h = x.new_zeros(x.size(0), self.hdim)
        h = self._scan(xp, h)
        if self.n_passes == 2:
            # second pass over the same projected inputs; h continues (warm carry)
            h = self._scan(xp, h)
        logits = self.head(h)
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": float("nan"),
                "escrow_norm": float("nan"),
                "n_passes": float(self.n_passes),
            }
        return logits


class FENBag1Pass(nn.Module):
    """Classic bag FEN: one scan, final head([h, E])."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.hdim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        xp = self.x_proj(x)
        g_sum = 0.0
        for t in range(T):
            z = h + xp[:, t]
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
                "escrow_norm": float(E.detach().norm(dim=-1).mean()),
                "n_passes": 1.0,
            }
        return logits


class FENBag2Pass(nn.Module):
    """
    Two-pass bag FEN with a *discrete* read event between passes.

    Pass 1: standard deplete + bag write (no E→h mix).
    Read:   c = read_proj(E)   # once
    Init:   cold → h = 0
            warm → h = tanh(h_init(c))
    Pass 2: same FEN step, but context c is added every step as *fixed* context
            (not a running dump of live E into residual — c is the read snapshot).
            Still depletes and continues writing E.
    Final:  head([h, E])
    """

    def __init__(self, input_dim, output_dim, hidden_dim, mode="cold"):
        super().__init__()
        assert mode in ("cold", "warm")
        self.hdim = hidden_dim
        self.mode = mode
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        self.read_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)  # fixed context into propose
        if mode == "warm":
            self.h_init = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.h_init = None
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def _step(self, h, E, x_t, c=None, accum_gate=None):
        # c is optional fixed context (pass 2 only)
        z = h + x_t
        if c is not None:
            z = z + self.c_proj(c)
        f = torch.tanh(self.core(z) + z)
        g = torch.sigmoid(self.gate(f))
        D = g * f
        h = f - D
        E = E + self.escrow_proj(D)
        if accum_gate is not None:
            accum_gate[0] += float(g.detach().mean())
        return h, E

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        xp = self.x_proj(x)
        g_acc = [0.0]

        # ---- pass 1: fill escrow ----
        for t in range(T):
            h, E = self._step(h, E, xp[:, t], c=None, accum_gate=g_acc if return_stats else None)

        # ---- discrete read event (once) ----
        c = torch.tanh(self.read_proj(E))

        if self.mode == "cold":
            h = x.new_zeros(B, self.hdim)
        else:
            h = torch.tanh(self.h_init(c))

        # ---- pass 2: re-scan, conditioned on fixed c; still FEN-deplete ----
        for t in range(T):
            h, E = self._step(h, E, xp[:, t], c=c, accum_gate=g_acc if return_stats else None)

        logits = self.head(torch.cat([h, E], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": g_acc[0] / (2 * T),
                "escrow_norm": float(E.detach().norm(dim=-1).mean()),
                "n_passes": 2.0,
                "read_norm": float(c.detach().norm(dim=-1).mean()),
            }
        return logits


class FENBagReinject(nn.Module):
    """
    Every-step reinject control (exp04 pathology).
    After deplete+write: h ← h + reinject(E)  — continuous pollution of the pipe.
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.hdim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        self.reinject = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E = x.new_zeros(B, self.hdim)
        xp = self.x_proj(x)
        g_sum = 0.0
        for t in range(T):
            z = h + xp[:, t]
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            h = f - D
            E = E + self.escrow_proj(D)
            # continuous mix of archive back into pipe (the thing exp04 killed)
            h = h + torch.tanh(self.reinject(E))
            if return_stats:
                g_sum += float(g.detach().mean())
        logits = self.head(torch.cat([h, E], dim=-1))
        if return_stats:
            return logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
                "escrow_norm": float(E.detach().norm(dim=-1).mean()),
                "n_passes": 1.0,
            }
        return logits


# ------------------------------ BUILD / MATCH ---------------------------------
def build(name, input_dim, output_dim, hidden_dim):
    if name == "residual_1pass":
        return ResidualPass(input_dim, output_dim, hidden_dim, n_passes=1)
    if name == "residual_2pass":
        return ResidualPass(input_dim, output_dim, hidden_dim, n_passes=2)
    if name == "fen_bag_1pass":
        return FENBag1Pass(input_dim, output_dim, hidden_dim)
    if name == "fen_bag_2pass_cold":
        return FENBag2Pass(input_dim, output_dim, hidden_dim, mode="cold")
    if name == "fen_bag_2pass_warm":
        return FENBag2Pass(input_dim, output_dim, hidden_dim, mode="warm")
    if name == "fen_bag_reinject":
        return FENBagReinject(input_dim, output_dim, hidden_dim)
    raise ValueError(name)


_HIDDEN_CACHE = {}


def choose_hidden(name, input_dim, output_dim):
    key = (name, input_dim, output_dim, TARGET_PARAMS)
    if key in _HIDDEN_CACHE:
        return _HIDDEN_CACHE[key]
    if not AUTO_MATCH_PARAMS:
        _HIDDEN_CACHE[key] = 48
        return 48

    best_h, best_diff = 48, float("inf")
    for h in range(MIN_H, MAX_H + 1):
        n = count_params(build(name, input_dim, output_dim, h))
        d = abs(n - TARGET_PARAMS)
        if d < best_diff:
            best_h, best_diff = h, d
    _HIDDEN_CACHE[key] = best_h
    return best_h


# ------------------------------ LOSS / METRICS --------------------------------
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
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb, return_stats=True)
        if isinstance(out, tuple):
            logits, stats = out
            for k, v in stats.items():
                if isinstance(v, (int, float)) and v == v:  # skip nan
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
    best = {"acc": -1.0, "epoch": 0}
    t0 = time.time()
    history = []

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
        if val["acc"] > best["acc"]:
            best = {
                "acc": val["acc"],
                "id_acc": val.get("id_acc", 0.0),
                "count_acc": val.get("count_acc", 0.0),
                "pipe_norm": val.get("pipe_norm", float("nan")),
                "escrow_norm": val.get("escrow_norm", float("nan")),
                "gate": val.get("gate", float("nan")),
                "epoch": ep,
            }

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            print(
                f"    ep {ep:02d}/{epochs}  loss={val['loss']:.4f}  "
                f"acc={val['acc']:.3f}  id={val['id_acc']:.3f}  "
                f"count={val['count_acc']:.3f}  "
                f"pipe={val.get('pipe_norm', float('nan')):.2f}"
            )

    best["time_s"] = time.time() - t0
    # epochs to 90% of best acc
    target = 0.9 * max(best["acc"], 1e-8)
    first = epochs
    for i, h in enumerate(history, 1):
        if h["acc"] >= target:
            first = i
            break
    best["to90"] = first
    return best


def main():
    print(
        f"Task=distracted | T={SEQ_LEN} | target_params≈{TARGET_PARAMS} | "
        f"train_n={TRAIN_N} test_n={TEST_N}"
    )
    print(
        "Score guide: fen_bag_1pass should be high; reinject should hurt / fat pipe; "
        "2pass cold/warm test discrete multi-pass read."
    )

    Xtr, ytr, meta = make_distracted(TRAIN_N, seed=1000)
    Xte, yte, _ = make_distracted(TEST_N, seed=2000)

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
            best = train_one(model, train_loader, test_loader, meta, EPOCHS)
            summary[model_name].append(best)
            print(
                f"  >> best acc={best['acc']:.3f}  id={best['id_acc']:.3f}  "
                f"count={best['count_acc']:.3f}  @ep{best['epoch']}  "
                f"pipe={best['pipe_norm']:.2f}  t={best['time_s']:.1f}s  "
                f"to90%={best['to90']}"
            )

    print("\n" + "-" * 78)
    print(
        f"SUMMARY  task=distracted  seeds={SEEDS}  target_params≈{TARGET_PARAMS}"
    )
    print("-" * 78)
    print(
        f"{'model':<22} {'acc':>7} {'id':>7} {'count':>7} "
        f"{'pipe':>7} {'to_best':>8} {'to90%':>6}"
    )
    for name in MODEL_ORDER:
        rows = summary[name]
        acc = np.mean([r["acc"] for r in rows])
        ida = np.mean([r["id_acc"] for r in rows])
        cta = np.mean([r["count_acc"] for r in rows])
        pipe = np.nanmean([r["pipe_norm"] for r in rows])
        epb = np.mean([r["epoch"] for r in rows])
        t90 = np.mean([r["to90"] for r in rows])
        print(
            f"{name:<22} {acc:7.3f} {ida:7.3f} {cta:7.3f} "
            f"{pipe:7.2f} {epb:8.1f} {t90:6.1f}"
        )
    print("-" * 78)
    print(
        "How to read:\n"
        "  • fen_bag_1pass high + lean pipe = baseline FEN success\n"
        "  • fen_bag_reinject: expect worse acc and/or much fatter pipe\n"
        "  • fen_bag_2pass_*: discrete read between passes — does it beat 1pass?\n"
        "  • residual_2pass: more compute alone (no escrow) should not match bag\n"
        "  Floors matter: joint acc ~0.1 = cannot do dual-role."
    )
    print("DONE — paste this SUMMARY back for scoring.")
    return summary


if __name__ == "__main__":
    main()
