# ==============================================================================
# fen_lab / EXP03 — Write algebra × Readout algebra (synthetic, Colab-ready)
# ==============================================================================
# Follow-up to exp02. Finding so far:
#   H4 (slot-aligned readout) unlocked order; H1–H3 (no_bag / sharp / event) did not.
#   Bag is load-bearing for distracted dual-role.
#
# Question:
#   Is order solved by WRITE structure, READOUT structure, or both?
#   Can bag (static archive) + slot readout (ordered delivery) coexist?
#
# Factor design (~15k params, auto-matched):
#
#   name              write              readout              bag
#   ----------------  -----------------  -------------------  -----
#   bag_pool          additive bag       concat [h,E]         yes
#   hard_pool         hard write ptr     pooled tape          no
#   hard_slot         hard write ptr     cell k → symbol k    no
#   soft_pool         soft γ-shift head  pooled tape          no
#   soft_slot         soft γ-shift head  cell k → symbol k    no
#   soft_bag_pool     soft γ-shift head  pooled + bag         yes   (≈ ode_full)
#   soft_bag_slot     soft γ-shift head  cell k → symbol k    yes   ★ hybrid
#
# Predictions (recall5 exact):
#   bag_pool ~ 0
#   hard_pool  ?   (if high → write alone enough; if ~0 → readout is the story)
#   hard_slot  ~ 1
#   soft_pool  ~ 0
#   soft_slot  ~ 1
#   soft_bag_slot ~ 1 if bag does not destroy order under slot head
#
# Predictions (distracted acc / id):
#   bag_pool, soft_bag_* high
#   hard_slot, soft_slot (no bag) low id
#   soft_bag_slot should stay high if bag path still works for non-recall heads
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
N_SYM = 5
VOCAB = 10
EVENT_GATE_THRESH = 0.25  # hard pointer advance threshold

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
    vocab, slots, noise_dims = VOCAB, N_SYM, 16
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


# ------------------------------ MODELS ----------------------------------------
class FactorFEN(nn.Module):
    """
    Unified depleting FEN with factorized write / readout / bag.

    write_mode:
      'bag'   — single additive escrow vector (no tape)
      'hard'  — K-cell tape, hard write pointer advanced on high gate
      'soft'  — K-cell tape, soft write head p with learned γ circular shift

    readout_mode:
      'pool'  — pool tape (or bag) and MLP head on [h, ...]
      'slot'  — cell k → symbol k (only when output_dim == N_SYM * VOCAB)

    use_bag:
      extra commutative channel c always available when True.
      For write_mode='bag', the escrow *is* the bag (use_bag ignored as True).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        write_mode="soft",
        readout_mode="pool",
        use_bag=False,
        K=TAPE_K,
        n_sym=N_SYM,
        vocab=VOCAB,
    ):
        super().__init__()
        assert write_mode in ("bag", "hard", "soft")
        assert readout_mode in ("pool", "slot")
        self.h = hidden_dim
        self.K = K
        self.write_mode = write_mode
        self.readout_mode = readout_mode
        # Extra bag channel only on top of a tape (hybrid). Pure bag write is its own escrow.
        self.extra_bag = bool(use_bag) and write_mode != "bag"
        self.use_bag = bool(use_bag) or write_mode == "bag"
        self.n_sym = n_sym
        self.vocab = vocab
        self.output_dim = output_dim
        self.has_tape = write_mode in ("hard", "soft")

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        if write_mode == "soft":
            self.gamma_head = nn.Linear(hidden_dim, 1)
            self.p0 = nn.Parameter(torch.zeros(K))
            with torch.no_grad():
                self.p0.zero_()
                self.p0[0] = 2.0
        else:
            self.gamma_head = None
            self.p0 = None

        if self.extra_bag:
            self.bag_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.bag_proj = None

        # pool path always built (needed for distracted / non-slot outputs)
        if self.has_tape:
            self.tape_pool = nn.Linear(K * hidden_dim, hidden_dim)
            head_in = hidden_dim * (3 if self.extra_bag else 2)
        else:
            self.tape_pool = None
            head_in = hidden_dim * 2  # [h, E_bag]
        self.head = nn.Linear(head_in, output_dim)
        self.cell_head = nn.Linear(hidden_dim, vocab)

    def _slot_readout_active(self):
        return (
            self.readout_mode == "slot"
            and self.has_tape
            and self.output_dim == self.n_sym * self.vocab
        )

    def forward(self, x, return_stats=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.h)
        c = x.new_zeros(B, self.h)

        if self.write_mode == "bag":
            E_bag = x.new_zeros(B, self.h)
            E_tape = None
            ptr = None
            p = None
        else:
            E_bag = None
            E_tape = x.new_zeros(B, self.K, self.h)
            if self.write_mode == "hard":
                ptr = torch.zeros(B, dtype=torch.long, device=x.device)
                p = None
            else:
                ptr = None
                p = torch.softmax(self.p0, dim=0).unsqueeze(0).expand(B, -1)

        g_sum = 0.0
        gamma_sum = 0.0
        ent_acc = h.new_zeros(())
        n_ent = 0

        for t in range(T):
            z = h + self.x_proj(x[:, t])
            f = torch.tanh(self.core(z) + z)
            g = torch.sigmoid(self.gate(f))
            D = g * f
            v = self.v_proj(D)
            h = f - D  # mandatory depletion

            if self.write_mode == "bag":
                E_bag = E_bag + v
            elif self.write_mode == "hard":
                one = F.one_hot(ptr, self.K).to(dtype=v.dtype)
                E_tape = E_tape + one.unsqueeze(-1) * v.unsqueeze(1)
                advance = (g.mean(dim=-1) > EVENT_GATE_THRESH).long()
                ptr = (ptr + advance) % self.K
            else:  # soft
                gamma = torch.sigmoid(self.gamma_head(f))
                p_shift = torch.roll(p, shifts=1, dims=-1)
                p = (1.0 - gamma) * p + gamma * p_shift
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
                E_tape = E_tape + p.unsqueeze(-1) * v.unsqueeze(1)
                ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
                ent_acc = ent_acc + ent
                n_ent += 1
                if return_stats:
                    gamma_sum += float(gamma.detach().mean())

            # optional extra bag channel (hybrid soft_bag_*)
            if self.extra_bag:
                c = c + self.bag_proj(D)

            if return_stats:
                g_sum += float(g.detach().mean().item())

        # ---- readout ----
        aux = h.new_zeros(())
        if self._slot_readout_active():
            cell_logits = self.cell_head(E_tape[:, : self.n_sym, :])  # [B,5,V]
            logits = cell_logits.reshape(B, -1)
        else:
            if self.write_mode == "bag":
                logits = self.head(torch.cat([h, E_bag], dim=-1))
            else:
                E_vec = torch.tanh(self.tape_pool(E_tape.reshape(B, -1)))
                if self.extra_bag:
                    logits = self.head(torch.cat([h, E_vec, c], dim=-1))
                else:
                    logits = self.head(torch.cat([h, E_vec], dim=-1))

        if return_stats:
            if n_ent:
                mean_ent = float((ent_acc / n_ent).detach().item())
            else:
                mean_ent = float("nan")
            return logits, aux, {
                "pipe_norm": float(h.detach().norm(dim=-1).mean().item()),
                "gate": g_sum / T,
                "gamma": (gamma_sum / T) if self.write_mode == "soft" else float("nan"),
                "head_entropy": mean_ent,
            }
        return logits, aux


# ------------------------------ BUILD / MATCH ---------------------------------
# write × readout × bag factor grid (named for logs)
MODEL_SPECS = {
    "bag_pool": dict(write_mode="bag", readout_mode="pool", use_bag=True),
    "hard_pool": dict(write_mode="hard", readout_mode="pool", use_bag=False),
    "hard_slot": dict(write_mode="hard", readout_mode="slot", use_bag=False),
    "soft_pool": dict(write_mode="soft", readout_mode="pool", use_bag=False),
    "soft_slot": dict(write_mode="soft", readout_mode="slot", use_bag=False),
    "soft_bag_pool": dict(write_mode="soft", readout_mode="pool", use_bag=True),
    "soft_bag_slot": dict(write_mode="soft", readout_mode="slot", use_bag=True),
}

MODEL_ORDER = list(MODEL_SPECS.keys())


def build(name, input_dim, output_dim, hidden_dim):
    return FactorFEN(input_dim, output_dim, hidden_dim, **MODEL_SPECS[name])


def choose_hidden(name, input_dim, output_dim):
    if not AUTO_MATCH_PARAMS:
        return 48
    best_h, best_diff = 48, float("inf")
    for h in range(MIN_H, MAX_H + 1):
        m = build(name, input_dim, output_dim, h)
        diff = abs(count_params(m) - TARGET_PARAMS)
        if diff < best_diff:
            best_h, best_diff = h, diff
    return best_h


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
        logits, aux, stats = model(xb, return_stats=True)
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
            gma = val.get("gamma", float("nan"))
            gma_s = f"{gma:.3f}" if gma == gma else "nan"
            print(
                f"    ep {ep:02d}/{epochs}  loss={val['loss']:.4f}  "
                f"acc={val.get('acc',0):.3f}{extra}  "
                f"pipe={val.get('pipe_norm', float('nan')):.2f}  "
                f"gate={val.get('gate', float('nan')):.3f}  "
                f"H(p)={he_s}  gma={gma_s}"
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

    for model_name in MODEL_ORDER:
        spec = MODEL_SPECS[model_name]
        print(f"\n--- Model: {model_name} ---")
        print(
            f"  write={spec['write_mode']}  read={spec['readout_mode']}  "
            f"bag={spec['use_bag']}"
        )
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
        ent_s = f"{np.nanmean(ents):7.3f}" if np.any(np.isfinite(ents)) else f"{'nan':>7}"
        pipe_s = f"{np.nanmean(pipes):7.2f}" if np.any(np.isfinite(pipes)) else f"{'nan':>7}"
        print(
            f"{name:<16} {vals.mean():8.3f} {secondary.mean():8.3f} "
            f"{speeds.mean():7.1f} {ent_s} {pipe_s}"
        )
    return summary


def main():
    print("EXP03 — Write algebra × Readout algebra")
    print(f"Tasks={TASKS}  Seeds={SEEDS}")
    print(f"Models={MODEL_ORDER}")
    print("Factors: write={bag,hard,soft} × read={pool,slot} × bag channel")
    print("Key new cells: hard_pool (write-only?)  soft_bag_slot (hybrid dual-role+order)")
    all_results = {}
    for task in TASKS:
        all_results[task] = run_task(task)

    print("\n" + "=" * 78)
    print("DONE — how to score (use numbers, not this list as truth):")
    print("  recall5 exact:")
    print("    bag_pool ~0              bag cannot order")
    print("    hard_pool high?          WRITE alone enough")
    print("    hard_pool ~0, hard_slot~1  READOUT is the order story")
    print("    soft_pool ~0, soft_slot~1  same cut for soft write")
    print("    soft_bag_slot ~ soft_slot  bag does not kill slot order")
    print("  distracted:")
    print("    bag_pool / soft_bag_* high id   bag needed for dual-role")
    print("    hard_slot / soft_slot low id    pure order structure fails dual-role")
    print("    soft_bag_slot high id           hybrid covers both if true")
    print("=" * 78)
    return all_results


if __name__ == "__main__":
    main()
