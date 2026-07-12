# ==============================================================================
# fen_lab / EXP04 — Mid-sequence delivery (T4: can escrow be *used* mid-run?)
# ==============================================================================
# Prior findings:
#   exp01–03: dual-role needs bag; order needs hard write and/or slot readout;
#   soft_bag_slot hybrid covers both at *final* readout.
#
# Missing operator (theory T4 — Deliver):
#   Classic FEN only concatenates E at the end. That never tests whether the
#   archive is readable *during* the sequence without permanently re-poisoning
#   the pipe.
#
# Task: interrupted distracted counting
#   t=0:        static ID pulse
#   t=1..T-1:   + / − count events + distractors + noise
#   t=QUERY_T:  query flag channel ON  → model must predict ID *now* (mid head)
#   end:        final label = ID × count_bin  (same as distracted)
#
# Delivery factor (all ~15k params):
#
#   name            archive     mid head sees          gated reinject each step?
#   --------------  ----------  ---------------------  -------------------------
#   residual        none        h only                 no
#   bag_h_only      bag E       h only                 no   ← ID should be in E
#   bag_read        bag E       [h, E]                 no   ← explicit mid read
#   bag_gated       bag E       h (after reinject)     yes  ← deliver into pipe
#   soft_bag_read   tape+bag    [h, pool(E), c]        no
#   soft_bag_gated  tape+bag    h (after reinject)     yes
#
# Predictions:
#   residual:        mid_id weak; final_id weak
#   bag_h_only:      mid_id ~chance (ID retired to E); final_id high
#   bag_read:        mid_id high; final high     ← proves mid-usable archive
#   bag_gated:       mid_id high if reinject works; pipe may stay lean if gate sparse
#   soft_bag_*:      same pattern
#
# Metrics: mid_id  |  final acc / id / count  |  pipe L2
# Loss:    CE(final) + MID_LOSS_W * CE(mid_id)
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
    SEEDS = [1]
    EPOCHS = 15
    TRAIN_N, TEST_N = 5000, 1200
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
QUERY_T = 40
NOISE_STD = 0.45
MID_LOSS_W = 1.0

TARGET_PARAMS = 15000
AUTO_MATCH_PARAMS = True
MIN_H, MAX_H = 8, 128
TAPE_K = 8
N_ID = 10
N_BINS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "| FAST_MODE:", FAST_MODE, "| QUERY_T:", QUERY_T)


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
def make_interrupted(n, seed):
    """
    Distracted counting + mid-sequence ID query.

    Channels:
      [0 : N_ID)     ID pulse at t=0
      N_ID+0,1       plus / minus
      N_ID+2,3       distractors
      N_ID+4         QUERY flag at QUERY_T
      N_ID+5:        noise

    y_final: ID * N_BINS + count_bin
    y_mid:   ID
    """
    rng = np.random.default_rng(seed)
    n_id, n_bins = N_ID, N_BINS
    op_dims, query_dim, noise_dims = 4, 1, 16
    input_dim = n_id + op_dims + query_dim + noise_dims

    plus_dim = n_id
    minus_dim = n_id + 1
    distract_a = n_id + 2
    distract_b = n_id + 3
    query_ch = n_id + 4
    noise_start = n_id + 5

    X = rng.normal(0.0, NOISE_STD, size=(n, SEQ_LEN, input_dim)).astype(np.float32)
    X[:, :, :noise_start] *= 0.10
    y_final = np.zeros((n,), dtype=np.int64)
    y_mid = np.zeros((n,), dtype=np.int64)

    assert 1 <= QUERY_T < SEQ_LEN

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

        X[i, QUERY_T, query_ch] = 1.5

        y_final[i] = static_id * n_bins + count_bin
        y_mid[i] = static_id

    meta = {
        "task": "interrupted",
        "input_dim": input_dim,
        "final_dim": n_id * n_bins,
        "mid_dim": n_id,
        "n_id": n_id,
        "n_bins": n_bins,
        "query_t": QUERY_T,
    }
    return torch.tensor(X), torch.tensor(y_final), torch.tensor(y_mid), meta


# ------------------------------ MODELS ----------------------------------------
class ResidualMid(nn.Module):
    """Residual tanh RNN — no escrow. Mid + final from h only."""

    def __init__(self, input_dim, final_dim, mid_dim, hidden_dim):
        super().__init__()
        self.hdim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.mid_head = nn.Linear(hidden_dim, mid_dim)
        self.final_head = nn.Linear(hidden_dim, final_dim)

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        mid_logits = None
        pipe_at_q = None
        for t in range(T):
            z = h + self.x_proj(x[:, t])
            h = torch.tanh(self.core(z) + z)
            if t == QUERY_T:
                mid_logits = self.mid_head(h)
                pipe_at_q = h
        final_logits = self.final_head(h)
        if return_stats:
            return final_logits, mid_logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean().item()),
                "pipe_at_q": float(pipe_at_q.detach().norm(dim=-1).mean().item())
                if pipe_at_q is not None
                else float("nan"),
                "gate": float("nan"),
                "read_gate": float("nan"),
            }
        return final_logits, mid_logits


class DeliverFEN(nn.Module):
    """
    Depleting FEN + mid-sequence delivery modes.

    write_mode: 'bag' | 'soft'  (soft = K-tape + bag channel)
    mid_mode:
      'h_only' — mid head sees only h
      'read'   — mid head sees [h, arch]
      'gated'  — each step reinject r=proj(arch) into h via read-gate; mid from h
    """

    def __init__(
        self,
        input_dim,
        final_dim,
        mid_dim,
        hidden_dim,
        write_mode="bag",
        mid_mode="read",
        K=TAPE_K,
    ):
        super().__init__()
        assert write_mode in ("bag", "soft")
        assert mid_mode in ("h_only", "read", "gated")
        self.hdim = hidden_dim
        self.write_mode = write_mode
        self.mid_mode = mid_mode
        self.K = K
        self.has_tape = write_mode == "soft"

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        if self.has_tape:
            self.gamma_head = nn.Linear(hidden_dim, 1)
            self.p0 = nn.Parameter(torch.zeros(K))
            with torch.no_grad():
                self.p0.zero_()
                self.p0[0] = 2.0
            self.bag_proj = nn.Linear(hidden_dim, hidden_dim)
            self.tape_pool = nn.Linear(K * hidden_dim, hidden_dim)
            self.arch_dim = hidden_dim * 2  # pool(E) + c
        else:
            self.gamma_head = None
            self.p0 = None
            self.bag_proj = None
            self.tape_pool = None
            self.arch_dim = hidden_dim

        # gated deliver: archive → r ∈ R^h, then h ← h + σ(W[h;r]) ⊙ r
        if mid_mode == "gated":
            self.r_proj = nn.Linear(self.arch_dim, hidden_dim)
            self.read_gate = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        else:
            self.r_proj = None
            self.read_gate = None

        if mid_mode == "read":
            mid_in = hidden_dim + self.arch_dim
        else:
            mid_in = hidden_dim
        self.mid_head = nn.Linear(mid_in, mid_dim)
        self.final_head = nn.Linear(hidden_dim + self.arch_dim, final_dim)

    def _arch(self, E_bag, E_tape, c):
        if self.has_tape:
            E_vec = torch.tanh(self.tape_pool(E_tape.reshape(E_tape.size(0), -1)))
            return torch.cat([E_vec, c], dim=-1)
        return E_bag

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hdim)
        E_bag = x.new_zeros(B, self.hdim)
        c = x.new_zeros(B, self.hdim)
        E_tape = x.new_zeros(B, self.K, self.hdim) if self.has_tape else None
        p = (
            torch.softmax(self.p0, dim=0).unsqueeze(0).expand(B, -1)
            if self.has_tape
            else None
        )

        mid_logits = None
        pipe_at_q = None
        g_sum = 0.0
        rg_sum = 0.0

        for t in range(T):
            z = h + self.x_proj(x[:, t])
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            v = self.v_proj(D)
            h = f - D

            if self.has_tape:
                gamma = torch.sigmoid(self.gamma_head(f))
                p = (1.0 - gamma) * p + gamma * torch.roll(p, shifts=1, dims=-1)
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
                E_tape = E_tape + p.unsqueeze(-1) * v.unsqueeze(1)
                c = c + self.bag_proj(D)
            else:
                E_bag = E_bag + v

            arch = self._arch(E_bag, E_tape, c)

            if self.mid_mode == "gated":
                r = self.r_proj(arch)
                gr = torch.sigmoid(self.read_gate(torch.cat([h, r], dim=-1)))
                h = h + gr * r
                if return_stats:
                    rg_sum += float(gr.detach().mean().item())

            if return_stats:
                g_sum += float(g.detach().mean().item())

            if t == QUERY_T:
                pipe_at_q = h
                if self.mid_mode == "read":
                    mid_logits = self.mid_head(torch.cat([h, arch], dim=-1))
                else:
                    # h_only or gated (post-reinject h)
                    mid_logits = self.mid_head(h)

        arch = self._arch(E_bag, E_tape, c)
        final_logits = self.final_head(torch.cat([h, arch], dim=-1))

        if return_stats:
            return final_logits, mid_logits, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean().item()),
                "pipe_at_q": float(pipe_at_q.detach().norm(dim=-1).mean().item())
                if pipe_at_q is not None
                else float("nan"),
                "gate": g_sum / T,
                "read_gate": (rg_sum / T) if self.mid_mode == "gated" else float("nan"),
            }
        return final_logits, mid_logits


# ------------------------------ BUILD / MATCH ---------------------------------
MODEL_SPECS = {
    "residual": dict(kind="residual"),
    "bag_h_only": dict(kind="fen", write_mode="bag", mid_mode="h_only"),
    "bag_read": dict(kind="fen", write_mode="bag", mid_mode="read"),
    "bag_gated": dict(kind="fen", write_mode="bag", mid_mode="gated"),
    "soft_bag_read": dict(kind="fen", write_mode="soft", mid_mode="read"),
    "soft_bag_gated": dict(kind="fen", write_mode="soft", mid_mode="gated"),
}
MODEL_ORDER = list(MODEL_SPECS.keys())


def build(name, input_dim, final_dim, mid_dim, hidden_dim):
    spec = MODEL_SPECS[name]
    if spec["kind"] == "residual":
        return ResidualMid(input_dim, final_dim, mid_dim, hidden_dim)
    return DeliverFEN(
        input_dim,
        final_dim,
        mid_dim,
        hidden_dim,
        write_mode=spec["write_mode"],
        mid_mode=spec["mid_mode"],
    )


def choose_hidden(name, input_dim, final_dim, mid_dim):
    if not AUTO_MATCH_PARAMS:
        return 48
    best_h, best_diff = 48, float("inf")
    for h in range(MIN_H, MAX_H + 1):
        m = build(name, input_dim, final_dim, mid_dim, h)
        diff = abs(count_params(m) - TARGET_PARAMS)
        if diff < best_diff:
            best_h, best_diff = h, diff
    return best_h


# ------------------------------ LOSS / TRAIN ----------------------------------
def loss_and_metrics(final_logits, mid_logits, y_final, y_mid, meta):
    loss_f = F.cross_entropy(final_logits, y_final)
    loss_m = F.cross_entropy(mid_logits, y_mid)
    loss = loss_f + MID_LOSS_W * loss_m

    pred_f = final_logits.argmax(dim=-1)
    pred_m = mid_logits.argmax(dim=-1)
    n_bins = meta["n_bins"]
    acc = (pred_f == y_final).float().mean().item()
    id_acc = (pred_f // n_bins == y_final // n_bins).float().mean().item()
    count_acc = (pred_f % n_bins == y_final % n_bins).float().mean().item()
    mid_id = (pred_m == y_mid).float().mean().item()
    return loss, {
        "acc": acc,
        "id_acc": id_acc,
        "count_acc": count_acc,
        "mid_id": mid_id,
        "loss_final": float(loss_f.detach().item()),
        "loss_mid": float(loss_m.detach().item()),
    }


@torch.no_grad()
def evaluate(model, loader, meta):
    model.eval()
    totals = defaultdict(float)
    n = 0
    stats_acc = defaultdict(float)
    n_stats = 0
    for xb, yf, ym in loader:
        xb = xb.to(DEVICE)
        yf = yf.to(DEVICE)
        ym = ym.to(DEVICE)
        final_logits, mid_logits, stats = model(xb, return_stats=True)
        for k, v in stats.items():
            if v == v:
                stats_acc[k] += float(v)
        n_stats += 1
        loss, metrics = loss_and_metrics(final_logits, mid_logits, yf, ym, meta)
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
    # best by mid_id first (the T4 metric), tie-break on final acc
    best = {"mid_id": -1.0, "acc": -1.0, "epoch": 0}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yf, ym in train_loader:
            xb = xb.to(DEVICE)
            yf = yf.to(DEVICE)
            ym = ym.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            final_logits, mid_logits = model(xb, return_stats=False)
            loss, _ = loss_and_metrics(final_logits, mid_logits, yf, ym, meta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val = evaluate(model, test_loader, meta)
        history.append(val)
        better = val["mid_id"] > best["mid_id"] or (
            val["mid_id"] == best["mid_id"] and val["acc"] > best["acc"]
        )
        if better:
            best = {
                "mid_id": val["mid_id"],
                "acc": val["acc"],
                "epoch": ep,
                **{k: val[k] for k in val if k != "loss"},
            }

        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            rg = val.get("read_gate", float("nan"))
            rg_s = f"{rg:.3f}" if rg == rg else "na"
            print(
                f"    ep {ep:02d}/{epochs}  loss={val['loss']:.4f}  "
                f"mid_id={val['mid_id']:.3f}  "
                f"final={val['acc']:.3f} id={val['id_acc']:.3f} "
                f"count={val['count_acc']:.3f}  "
                f"pipe={val.get('pipe_norm', float('nan')):.2f}  "
                f"pipe@q={val.get('pipe_at_q', float('nan')):.2f}  "
                f"gate={val.get('gate', float('nan')):.3f}  rg={rg_s}"
            )

    elapsed = time.time() - t0
    # epochs to 90% of best mid_id
    target = 0.9 * max(best["mid_id"], 1e-8)
    first_good = next(
        (i for i, h in enumerate(history, 1) if h["mid_id"] >= target),
        epochs,
    )
    best["time_s"] = elapsed
    best["epochs_to_90pct_mid"] = first_good
    return best, history


# ------------------------------ MAIN ------------------------------------------
def main():
    print("EXP04 — Mid-sequence delivery (T4)")
    print(f"Seeds={SEEDS}  Epochs={EPOCHS}  SEQ={SEQ_LEN}  QUERY_T={QUERY_T}")
    print(f"Models={MODEL_ORDER}")
    print("Primary metric: mid_id  |  Secondary: final acc/id/count  |  pipe norms")

    Xtr, yf_tr, ym_tr, meta = make_interrupted(TRAIN_N, seed=1000)
    Xte, yf_te, ym_te, _ = make_interrupted(TEST_N, seed=2000)
    print(
        f"Data: train={tuple(Xtr.shape)} test={tuple(Xte.shape)} "
        f"final_dim={meta['final_dim']} mid_dim={meta['mid_dim']}"
    )

    summary = defaultdict(list)

    for model_name in MODEL_ORDER:
        spec = MODEL_SPECS[model_name]
        print(f"\n--- Model: {model_name} ---")
        if spec["kind"] == "fen":
            print(f"  write={spec['write_mode']}  mid={spec['mid_mode']}")
        else:
            print("  residual RNN (no escrow)")

        hdim = choose_hidden(
            model_name, meta["input_dim"], meta["final_dim"], meta["mid_dim"]
        )
        nparams = count_params(
            build(
                model_name,
                meta["input_dim"],
                meta["final_dim"],
                meta["mid_dim"],
                hdim,
            )
        )
        print(f"  hidden={hdim}  params={nparams}")

        for seed in SEEDS:
            seed_everything(seed)
            model = build(
                model_name,
                meta["input_dim"],
                meta["final_dim"],
                meta["mid_dim"],
                hdim,
            ).to(DEVICE)
            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                TensorDataset(Xtr, yf_tr, ym_tr),
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=g,
            )
            test_loader = DataLoader(
                TensorDataset(Xte, yf_te, ym_te),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            print(f"  seed={seed}")
            best, _ = train_one(model, train_loader, test_loader, meta, EPOCHS)
            summary[model_name].append(best)
            print(
                f"  >> best mid_id={best['mid_id']:.3f}  final={best['acc']:.3f}  "
                f"id={best.get('id_acc', 0):.3f}  count={best.get('count_acc', 0):.3f}  "
                f"@ep{best['epoch']}  t={best['time_s']:.1f}s  "
                f"pipe={best.get('pipe_norm', float('nan')):.2f}"
            )

    print("\n" + "-" * 78)
    print(f"SUMMARY  task=interrupted  seeds={SEEDS}  QUERY_T={QUERY_T}")
    print("-" * 78)
    print(
        f"{'model':<16} {'mid_id':>7} {'final':>7} {'id':>7} {'count':>7} "
        f"{'to90%':>6} {'pipe':>6} {'pipe@q':>7}"
    )
    for name, runs in summary.items():
        def mean(key, default=np.nan):
            vals = [r.get(key, default) for r in runs]
            arr = np.array(vals, dtype=np.float64)
            return float(np.nanmean(arr))

        print(
            f"{name:<16} {mean('mid_id'):7.3f} {mean('acc'):7.3f} "
            f"{mean('id_acc'):7.3f} {mean('count_acc'):7.3f} "
            f"{mean('epochs_to_90pct_mid'):6.1f} {mean('pipe_norm'):6.2f} "
            f"{mean('pipe_at_q'):7.2f}"
        )

    print("\n" + "=" * 78)
    print("DONE — how to score (use numbers):")
    print("  mid_id ~ chance(0.1) on bag_h_only  → ID was retired from pipe (T2 works)")
    print("  mid_id high on bag_read             → archive is mid-usable (T4 read works)")
    print("  mid_id high on bag_gated            → reinject delivery works")
    print("  final id high on *read/*gated       → mid use did not destroy dual-role")
    print("  residual mid+final weak             → no external archive")
    print("  pipe@q / pipe                       → gated should not bloat forever")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()
