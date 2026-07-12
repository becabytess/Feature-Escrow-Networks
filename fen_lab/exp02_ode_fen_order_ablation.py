# ==============================================================================
# fen_lab / EXP02 — ODE-FEN order ablations (synthetic, Colab-ready)
# ==============================================================================
# Follow-up to exp01. Question:
#   Why did soft-tape ODE-FEN collapse to bag behavior on recall5?
#
# Hypotheses tested:
#   H1  Bag channel c steals the task (commutative escape hatch)
#   H2  Write head p stays diffuse (needs sharpening pressure)
#   H3  Head advances every step, not on events (smears order)
#   H4  Pooled tape readout throws away cell index (needs slot-aligned head)
#
# Models (~15k, auto-matched unless noted):
#   fen_bag              — commutative baseline (should fail exact)
#   fen_slot             — hard pointer upper bound (should win exact)
#   fen_roll             — channel-roll (continuous structure baseline)
#   ode_full             — exp01 ODE-FEN (bag + soft shift + pool)
#   ode_no_bag           — H1: remove bag channel
#   ode_sharp            — H1+H2: no bag + entropy penalty on p
#   ode_event            — H1+H3: no bag + advance only on high gate
#   ode_slot_read        — H1+H4: no bag + per-cell readout for first 5 symbols
#   ode_all              — H1+H2+H3+H4 combined
#
# Primary task: recall5 (exact sequence accuracy).
# Secondary: distracted (sanity — no_bag should not totally die).
#
# Paste into one Colab cell. Deps: torch, numpy. No downloads.
# ==============================================================================

import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------ CONFIG ----------------------------------------
FAST_MODE = True

if FAST_MODE:
    # Longer than exp01 on purpose: soft order needs runway
    TASKS = ["recall5", "distracted"]
    SEEDS = [1]
    EPOCHS_RECALL = 25
    EPOCHS_DISTRACTED = 12
    TRAIN_N, TEST_N = 5000, 1200
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

# Regularizers for sharp variants
ENTROPY_COEF = 0.02          # encourage peaked write head (lower entropy)
EVENT_GATE_THRESH = 0.30     # mean(g) above this counts as "event"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "| FAST_MODE:", FAST_MODE)


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
        "slots": None,
        "vocab": None,
    }
    return torch.tensor(X), torch.tensor(y), meta


def make_recall5(n, seed, random_positions=False):
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
        "n_id": None,
        "n_bins": None,
    }
    return torch.tensor(X), torch.tensor(y), meta


def make_dataset(task, n, seed):
    if task == "distracted":
        return make_distracted(n, seed)
    if task == "recall5":
        return make_recall5(n, seed, random_positions=False)
    raise ValueError(task)


# ------------------------------ BASELINES -------------------------------------
class FENBag(nn.Module):
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
        h = x.new_zeros(B, self.h)
        E = x.new_zeros(B, self.h)
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
        aux = logits.new_zeros(())
        if return_stats:
            return logits, aux, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
                "head_entropy": float("nan"),
                "gamma": float("nan"),
            }
        return logits, aux


class FENRoll(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.h = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        self.roll_gate = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.h)
        E = x.new_zeros(B, self.h)
        g_sum = 0.0
        for t in range(T):
            z = h + self.x_proj(x[:, t])
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            h = f - D
            gamma = torch.sigmoid(self.roll_gate(f))
            v = self.escrow_proj(D)
            E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v
            if return_stats:
                g_sum += float(g.detach().mean())
        logits = self.head(torch.cat([h, E], dim=-1))
        aux = logits.new_zeros(())
        if return_stats:
            return logits, aux, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
                "head_entropy": float("nan"),
                "gamma": float("nan"),
            }
        return logits, aux


class FENSlot(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, K=TAPE_K, n_sym_slots=5):
        super().__init__()
        self.h = hidden_dim
        self.K = K
        self.n_sym = n_sym_slots
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.escrow_proj = nn.Linear(hidden_dim, hidden_dim)
        # slot-aligned readout for recall (cells 0..4 -> symbol logits)
        self.cell_head = nn.Linear(hidden_dim, 10)  # vocab=10; unused dims ok for distracted via pool path
        self.tape_pool = nn.Linear(K * hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, output_dim)
        self.output_dim = output_dim

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.h)
        E = x.new_zeros(B, self.K, self.h)
        ptr = torch.zeros(B, dtype=torch.long, device=x.device)
        g_sum = 0.0
        for t in range(T):
            z = h + self.x_proj(x[:, t])
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            h = f - D
            v = self.escrow_proj(D)
            one = F.one_hot(ptr, self.K).to(dtype=v.dtype)
            E = E + one.unsqueeze(-1) * v.unsqueeze(1)
            advance = (g.mean(dim=-1) > 0.25).long()
            ptr = (ptr + advance) % self.K
            if return_stats:
                g_sum += float(g.detach().mean())

        # If output looks like recall5 (slots*vocab with vocab=10), use per-cell heads
        if self.output_dim == self.n_sym * 10:
            # cells 0..n_sym-1 each predict a symbol
            cell_logits = self.cell_head(E[:, : self.n_sym, :])  # [B,5,10]
            logits = cell_logits.reshape(B, -1)
        else:
            E_vec = torch.tanh(self.tape_pool(E.reshape(B, -1)))
            logits = self.head(torch.cat([h, E_vec], dim=-1))

        aux = logits.new_zeros(())
        if return_stats:
            return logits, aux, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
                "head_entropy": float("nan"),
                "gamma": float("nan"),
            }
        return logits, aux


# ------------------------------ ODE-FEN FAMILY --------------------------------
class ODEFEN(nn.Module):
    """
    Configurable soft-tape FEN.

    Flags:
      use_bag       — commutative bag channel c (exp01 default True)
      sharpen       — entropy penalty on write head p
      event_shift   — only advance p when mean(g) > threshold
      slot_readout  — for recall5: cell k predicts symbol k (k=0..4)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        K=TAPE_K,
        use_bag=True,
        sharpen=False,
        event_shift=False,
        slot_readout=False,
        n_sym_slots=5,
        vocab=10,
    ):
        super().__init__()
        self.h = hidden_dim
        self.K = K
        self.use_bag = use_bag
        self.sharpen = sharpen
        self.event_shift = event_shift
        self.slot_readout = slot_readout
        self.n_sym = n_sym_slots
        self.vocab = vocab
        self.output_dim = output_dim

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gamma_head = nn.Linear(hidden_dim, 1)

        if use_bag:
            self.bag_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.bag_proj = None

        self.tape_pool = nn.Linear(K * hidden_dim, hidden_dim)
        # readout: [h, E_pool] or [h, E_pool, c]
        head_in = hidden_dim * (3 if use_bag else 2)
        self.head = nn.Linear(head_in, output_dim)
        self.cell_head = nn.Linear(hidden_dim, vocab)

        self.p0 = nn.Parameter(torch.zeros(K))
        with torch.no_grad():
            self.p0.zero_()
            self.p0[0] = 2.0

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.h)
        E = x.new_zeros(B, self.K, self.h)
        c = x.new_zeros(B, self.h)
        p = torch.softmax(self.p0, dim=0).unsqueeze(0).expand(B, -1)

        g_sum = 0.0
        gamma_sum = 0.0
        ent_acc = h.new_zeros(())

        for t in range(T):
            z = h + self.x_proj(x[:, t])
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            v = self.v_proj(D)
            g_mean = g.mean(dim=-1, keepdim=True)  # [B,1]

            # proposed shift
            gamma = torch.sigmoid(self.gamma_head(f))  # [B,1]
            if self.event_shift:
                # only peristalse on event-like commits
                event = (g_mean > EVENT_GATE_THRESH).to(dtype=f.dtype)
                gamma_eff = gamma * event
            else:
                gamma_eff = gamma

            p_shift = torch.roll(p, shifts=1, dims=-1)
            p = (1.0 - gamma_eff) * p + gamma_eff * p_shift
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)

            E = E + p.unsqueeze(-1) * v.unsqueeze(1)

            if self.use_bag:
                c = c + self.bag_proj(D)

            h = f - D

            # entropy of p (for sharpening); average over time
            ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
            ent_acc = ent_acc + ent

            if return_stats:
                g_sum += float(g.detach().mean())
                gamma_sum += float(gamma_eff.detach().mean())

        # aux: encourage low entropy (peaked head) when sharpen=True
        mean_ent = ent_acc / T
        if self.sharpen:
            # max entropy for K-simplex is log(K); normalize for scale stability
            aux = ENTROPY_COEF * mean_ent
        else:
            aux = h.new_zeros(())

        use_slot_head = (
            self.slot_readout
            and self.output_dim == self.n_sym * self.vocab
        )
        if use_slot_head:
            cell_logits = self.cell_head(E[:, : self.n_sym, :])  # [B,5,V]
            logits = cell_logits.reshape(B, -1)
        else:
            E_vec = torch.tanh(self.tape_pool(E.reshape(B, -1)))
            if self.use_bag:
                logits = self.head(torch.cat([h, E_vec, c], dim=-1))
            else:
                logits = self.head(torch.cat([h, E_vec], dim=-1))

        if return_stats:
            return logits, aux, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean()),
                "gate": g_sum / T,
                "gamma": gamma_sum / T,
                "head_entropy": float(mean_ent.detach()),
            }
        return logits, aux


# ------------------------------ BUILD / MATCH ---------------------------------
def _ode(idim, odim, h, **kw):
    return ODEFEN(idim, odim, h, K=TAPE_K, **kw)


MODEL_CTORS = {
    "fen_bag": lambda i, o, h: FENBag(i, o, h),
    "fen_roll": lambda i, o, h: FENRoll(i, o, h),
    "fen_slot": lambda i, o, h: FENSlot(i, o, h, K=TAPE_K),
    "ode_full": lambda i, o, h: _ode(i, o, h, use_bag=True, sharpen=False, event_shift=False, slot_readout=False),
    "ode_no_bag": lambda i, o, h: _ode(i, o, h, use_bag=False, sharpen=False, event_shift=False, slot_readout=False),
    "ode_sharp": lambda i, o, h: _ode(i, o, h, use_bag=False, sharpen=True, event_shift=False, slot_readout=False),
    "ode_event": lambda i, o, h: _ode(i, o, h, use_bag=False, sharpen=False, event_shift=True, slot_readout=False),
    "ode_slot_read": lambda i, o, h: _ode(i, o, h, use_bag=False, sharpen=False, event_shift=False, slot_readout=True),
    "ode_all": lambda i, o, h: _ode(i, o, h, use_bag=False, sharpen=True, event_shift=True, slot_readout=True),
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


# ------------------------------ LOSS / TRAIN ----------------------------------
def task_loss_and_metrics(logits, y, meta):
    if meta["task"] == "distracted":
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        n_bins = meta["n_bins"]
        id_acc = (pred // n_bins == y // n_bins).float().mean().item()
        count_acc = (pred % n_bins == y % n_bins).float().mean().item()
        return loss, {"acc": acc, "id_acc": id_acc, "count_acc": count_acc, "exact": acc}

    slots, vocab = meta["slots"], meta["vocab"]
    B = logits.size(0)
    logits_s = logits.view(B, slots, vocab)
    loss = sum(F.cross_entropy(logits_s[:, j], y[:, j]) for j in range(slots)) / slots
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
        logits, aux, stats = out
        for k, v in stats.items():
            if v == v:  # not NaN
                stats_acc[k] += float(v)
        n_stats += 1
        loss, metrics = task_loss_and_metrics(logits, yb, meta)
        bs = xb.size(0)
        totals["loss"] += float(loss.item()) * bs
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
            logits, aux = model(xb, return_stats=False)
            loss, _ = task_loss_and_metrics(logits, yb, meta)
            (loss + aux).backward()
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
                **{k: val[k] for k in val if k != "loss"},
            }

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            if meta["task"] == "distracted":
                extra = f" id={val.get('id_acc',0):.3f} count={val.get('count_acc',0):.3f}"
            else:
                extra = f" exact={val.get('exact',0):.3f} token={val.get('token_acc',0):.3f}"
            he = val.get("head_entropy", float("nan"))
            he_s = f"{he:.3f}" if he == he else "na"
            print(
                f"    ep {ep:02d}/{epochs}  loss={val['loss']:.4f}  "
                f"acc={val.get('acc',0):.3f}{extra}  "
                f"pipe={val.get('pipe_norm', float('nan')):.2f}  "
                f"gate={val.get('gate', float('nan')):.3f}  "
                f"H(p)={he_s}  gma={val.get('gamma', float('nan')):.3f}"
            )

    elapsed = time.time() - t0
    target = 0.9 * max(best["exact"], 1e-8)
    first_good = next(
        (i for i, h in enumerate(history, 1) if h.get("exact", h.get("acc", 0)) >= target),
        epochs,
    )
    best["time_s"] = elapsed
    best["epochs_to_90pct_best"] = first_good
    return best, history


# ------------------------------ MAIN ------------------------------------------
def run_task(task: str):
    epochs = EPOCHS_RECALL if task == "recall5" else EPOCHS_DISTRACTED
    print("\n" + "=" * 78)
    print(f"TASK: {task}  epochs={epochs}")
    print("=" * 78)

    Xtr, ytr, meta = make_dataset(task, TRAIN_N, seed=1000)
    Xte, yte, _ = make_dataset(task, TEST_N, seed=2000)
    summary = defaultdict(list)

    for model_name in MODEL_CTORS:
        print(f"\n--- Model: {model_name} ---")
        hdim = choose_hidden(model_name, meta["input_dim"], meta["output_dim"])
        nparams = count_params(build(model_name, meta["input_dim"], meta["output_dim"], hdim))
        print(f"  hidden={hdim}  params={nparams}")

        for seed in SEEDS:
            seed_everything(seed)
            model = build(model_name, meta["input_dim"], meta["output_dim"], hdim).to(DEVICE)
            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, generator=g
            )
            test_loader = DataLoader(
                TensorDataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False
            )
            print(f"  seed={seed}")
            best, _ = train_one(model, train_loader, test_loader, meta, epochs)
            summary[model_name].append(best)
            if task == "recall5":
                print(
                    f"  >> best exact={best['exact']:.3f}  token={best['acc']:.3f}  "
                    f"@ep{best['epoch']}  t={best['time_s']:.1f}s  "
                    f"H(p)={best.get('head_entropy', float('nan'))}"
                )
            else:
                print(
                    f"  >> best acc={best['acc']:.3f}  id={best.get('id_acc',0):.3f}  "
                    f"count={best.get('count_acc',0):.3f}  @ep{best['epoch']}  "
                    f"t={best['time_s']:.1f}s"
                )

    print("\n" + "-" * 78)
    print(f"SUMMARY  task={task}  seeds={SEEDS}")
    print("-" * 78)
    key = "exact" if task == "recall5" else "acc"
    print(f"{'model':<16} {key:>8} {'token/id':>8} {'to90%':>7} {'H(p)':>7} {'pipe':>7}")
    for name, runs in summary.items():
        vals = np.array([r[key] for r in runs], dtype=np.float64)
        secondary = np.array(
            [r.get("token_acc", r.get("id_acc", r.get("acc", 0))) for r in runs],
            dtype=np.float64,
        )
        speeds = np.array([r["epochs_to_90pct_best"] for r in runs], dtype=np.float64)
        ents = np.array([r.get("head_entropy", np.nan) for r in runs], dtype=np.float64)
        pipes = np.array([r.get("pipe_norm", np.nan) for r in runs], dtype=np.float64)
        print(
            f"{name:<16} {vals.mean():8.3f} {secondary.mean():8.3f} "
            f"{speeds.mean():7.1f} {np.nanmean(ents):7.3f} {np.nanmean(pipes):7.2f}"
        )
    return summary


def main():
    print("EXP02 — ODE-FEN order ablations")
    print(f"Tasks={TASKS}  Seeds={SEEDS}")
    print(f"Models={list(MODEL_CTORS.keys())}")
    print(
        "Hypotheses: H1 no_bag | H2 sharp H(p) | H3 event_shift | H4 slot_readout | all"
    )
    all_results = {}
    for task in TASKS:
        all_results[task] = run_task(task)

    print("\n" + "=" * 78)
    print("DONE — how to interpret recall5 exact:")
    print("  fen_bag ~0          commutative trap confirmed")
    print("  fen_slot high       ordered write works")
    print("  ode_full ~ bag      bag escape still on")
    print("  ode_no_bag > full   H1 supported (bag was stealing)")
    print("  ode_sharp > no_bag  H2 supported (need peaked head)")
    print("  ode_event helps     H3 supported (advance on events)")
    print("  ode_slot_read helps H4 supported (keep cell index)")
    print("  ode_all best soft   combined fix for soft tape")
    print("On distracted: bag/full should stay strong; pure order tricks may lag.")
    print("=" * 78)
    return all_results


if __name__ == "__main__":
    main()
