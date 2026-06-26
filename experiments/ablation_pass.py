# ============================================================
# PDN Reproducibility / Ablation Pass
# Colab-ready single-cell script
# ============================================================

import math, random, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 1. EDIT THIS CONFIG BLOCK ONLY
# ============================================================

TASK = "distracted"
# Options:
#   "distracted"       -> static ID + active counting task
#   "recall5_fixed"   -> delayed recall of 5 symbols placed at beginning
#   "recall5_random"  -> delayed recall of 5 symbols at random positions

RUN_MODE = "pdn_full"
# Options:
#   Baselines:
#     "rnn"
#     "gru"
#     "lstm"
#
#   PDN variants:
#     "pdn_full"
#     "pdn_no_subtraction"  -> vault accumulates, but pipe is not emptied
#     "pdn_no_skip"         -> vault becomes recurrent/tanh, so archived memory can drift
#     "pdn_no_archive"      -> pipe subtracts diffused info, but vault is disabled
#     "pdn_no_gate"         -> everything diffuses; gate forced open
#     "pdn_leaky_vault"     -> vault decays over time
#     "pdn_mlp_head"        -> full PDN with nonlinear readout head

# Use "all_quick" if you want one run to compare many modes quickly.
# RUN_MODE = "all_quick"

POSITION = "beginning"
# Only used for TASK="distracted".
# Options: "beginning", "random"

SEEDS = [1, 2, 3]
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0

TRAIN_N = 8000
TEST_N = 2000

SEQ_LEN = 96
NOISE_STD = 0.45

TARGET_PARAMS = 15000
AUTO_MATCH_PARAMS = True
MIN_HIDDEN = 8
MAX_HIDDEN = 128

PRINT_EVERY_EPOCH = True

# ============================================================
# 2. DEVICE / SEEDING
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# 3. SYNTHETIC DATASETS
# ============================================================

def make_distracted_dataset(n, seed):
    """
    Static ID + active counting.
    Input:
      - one static ID token from 10 classes
      - many + / - pulses over the sequence
      - distractor pulses
      - Gaussian noise
    Target:
      combined class = ID * 3 + count_bin
    """
    rng = np.random.default_rng(seed)

    n_id = 10
    n_bins = 3
    op_dims = 4
    noise_dims = 16
    input_dim = n_id + op_dims + noise_dims

    X = rng.normal(0.0, NOISE_STD, size=(n, SEQ_LEN, input_dim)).astype(np.float32)

    # Keep symbolic/event channels cleaner than noise channels.
    X[:, :, :n_id + op_dims] *= 0.10

    y = np.zeros((n,), dtype=np.int64)

    plus_dim = n_id
    minus_dim = n_id + 1
    distract_a = n_id + 2
    distract_b = n_id + 3

    for i in range(n):
        static_id = rng.integers(0, n_id)
        count_bin = rng.integers(0, n_bins)

        if POSITION == "beginning":
            id_pos = 0
        elif POSITION == "random":
            id_pos = rng.integers(0, SEQ_LEN)
        else:
            raise ValueError("POSITION must be 'beginning' or 'random'")

        X[i, id_pos, static_id] += 2.0

        possible = np.array([t for t in range(SEQ_LEN) if t != id_pos])
        n_events = rng.integers(18, 31)
        positions = rng.choice(possible, size=n_events, replace=False)
        rng.shuffle(positions)

        if count_bin == 0:
            p_plus = 0.25
        elif count_bin == 1:
            p_plus = 0.50
        else:
            p_plus = 0.75

        n_plus = int(round(n_events * p_plus))
        plus_positions = positions[:n_plus]
        minus_positions = positions[n_plus:]

        X[i, plus_positions, plus_dim] += 1.5
        X[i, minus_positions, minus_dim] += 1.5

        # Distractor impulses unrelated to the label.
        n_distract = rng.integers(10, 25)
        dpos = rng.choice(possible, size=n_distract, replace=False)
        half = n_distract // 2
        X[i, dpos[:half], distract_a] += 1.25
        X[i, dpos[half:], distract_b] += 1.25

        y[i] = static_id * n_bins + count_bin

    meta = {
        "input_dim": input_dim,
        "output_dim": n_id * n_bins,
        "task_type": "single",
        "n_id": n_id,
        "n_bins": n_bins,
    }

    return torch.tensor(X), torch.tensor(y), meta


def make_recall5_dataset(n, seed, random_positions=False):
    """
    Delayed recall of 5 symbols.
    The model must output the 5 symbols in chronological order.
    """
    rng = np.random.default_rng(seed)

    vocab = 10
    slots = 5
    noise_dims = 16
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
            X[i, pos, symbols[j]] += 2.0

        # Extra distractor symbol flashes.
        n_distract = rng.integers(12, 25)
        blocked = set(positions.tolist())
        possible = np.array([t for t in range(SEQ_LEN) if t not in blocked])
        dpos = rng.choice(possible, size=n_distract, replace=False)
        dsym = rng.integers(0, vocab, size=n_distract)
        for pos, sym in zip(dpos, dsym):
            X[i, pos, sym] += 0.75

    meta = {
        "input_dim": input_dim,
        "output_dim": slots * vocab,
        "task_type": "recall",
        "vocab": vocab,
        "slots": slots,
    }

    return torch.tensor(X), torch.tensor(y), meta


def make_dataset(task, n, seed):
    if task == "distracted":
        return make_distracted_dataset(n, seed)
    if task == "recall5_fixed":
        return make_recall5_dataset(n, seed, random_positions=False)
    if task == "recall5_random":
        return make_recall5_dataset(n, seed, random_positions=True)
    raise ValueError(f"Unknown TASK: {task}")

# ============================================================
# 4. MODELS
# ============================================================

class BaselineRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, kind):
        super().__init__()
        self.kind = kind

        if kind == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        elif kind == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif kind == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(kind)

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_stats=False):
        out, state = self.rnn(x)

        if self.kind == "lstm":
            h = state[0][-1]
        else:
            h = state[-1]

        logits = self.head(h)

        if return_stats:
            return logits, {}
        return logits


class PDN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, mode):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.vault_proj = nn.Linear(hidden_dim, hidden_dim)

        # Used only by pdn_no_skip, but always defined for clean loading/counting.
        self.vault_recur = nn.Linear(hidden_dim, hidden_dim)

        head_in = hidden_dim * 2

        if mode == "pdn_mlp_head":
            self.head = nn.Sequential(
                nn.Linear(head_in, head_in),
                nn.ReLU(),
                nn.Linear(head_in, output_dim),
            )
        else:
            self.head = nn.Linear(head_in, output_dim)

    def forward(self, x, return_stats=False):
        batch = x.shape[0]
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        S = torch.zeros(batch, self.hidden_dim, device=x.device)

        gate_means = []
        diffuse_norms = []
        raw_norms = []

        actual_mode = "pdn_full" if self.mode == "pdn_mlp_head" else self.mode

        for t in range(x.shape[1]):
            xt = self.x_proj(x[:, t])
            z = h + xt

            h_raw = torch.tanh(self.core(z) + z)

            if actual_mode == "pdn_no_gate":
                g = torch.ones_like(h_raw)
            else:
                g = torch.sigmoid(self.gate(h_raw))

            D = g * h_raw

            if actual_mode == "pdn_no_subtraction":
                h = h_raw
            else:
                h = h_raw - D

            if actual_mode == "pdn_no_archive":
                S = torch.zeros_like(S)
            elif actual_mode == "pdn_no_skip":
                S = torch.tanh(self.vault_recur(S) + self.vault_proj(D))
            elif actual_mode == "pdn_leaky_vault":
                S = 0.95 * S + self.vault_proj(D)
            else:
                S = S + self.vault_proj(D)

            if return_stats:
                gate_means.append(g.mean().detach())
                diffuse_norms.append(D.norm(dim=-1).mean().detach())
                raw_norms.append(h_raw.norm(dim=-1).mean().detach())

        combined = torch.cat([h, S], dim=-1)
        logits = self.head(combined)

        if return_stats:
            stats = {
                "gate_mean": torch.stack(gate_means).mean().item(),
                "diffuse_norm": torch.stack(diffuse_norms).mean().item(),
                "raw_norm": torch.stack(raw_norms).mean().item(),
                "pipe_norm": h.norm(dim=-1).mean().item(),
                "vault_norm": S.norm(dim=-1).mean().item(),
            }
            return logits, stats

        return logits


def build_model(mode, input_dim, output_dim, hidden_dim):
    if mode in ["rnn", "gru", "lstm"]:
        return BaselineRNN(input_dim, output_dim, hidden_dim, mode)
    if mode.startswith("pdn"):
        return PDN(input_dim, output_dim, hidden_dim, mode)
    raise ValueError(f"Unknown RUN_MODE: {mode}")


def choose_hidden(mode, input_dim, output_dim):
    if not AUTO_MATCH_PARAMS:
        return 48

    best_h = None
    best_diff = None
    best_params = None

    for h in range(MIN_HIDDEN, MAX_HIDDEN + 1):
        model = build_model(mode, input_dim, output_dim, h)
        params = count_params(model)
        diff = abs(params - TARGET_PARAMS)
        if best_diff is None or diff < best_diff:
            best_h = h
            best_diff = diff
            best_params = params

    return best_h

# ============================================================
# 5. LOSS / METRICS
# ============================================================

def compute_loss_and_metrics(logits, y, meta):
    if meta["task_type"] == "single":
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)

        acc = (pred == y).float().mean().item()

        n_bins = meta["n_bins"]
        pred_id = pred // n_bins
        true_id = y // n_bins
        pred_bin = pred % n_bins
        true_bin = y % n_bins

        id_acc = (pred_id == true_id).float().mean().item()
        bin_acc = (pred_bin == true_bin).float().mean().item()

        metrics = {
            "acc": acc,
            "id_acc": id_acc,
            "count_acc": bin_acc,
        }
        return loss, metrics

    if meta["task_type"] == "recall":
        slots = meta["slots"]
        vocab = meta["vocab"]

        logits = logits.view(logits.shape[0], slots, vocab)

        loss = 0.0
        for j in range(slots):
            loss = loss + nn.functional.cross_entropy(logits[:, j, :], y[:, j])
        loss = loss / slots

        pred = logits.argmax(dim=-1)
        token_acc = (pred == y).float().mean().item()
        exact_acc = (pred == y).all(dim=1).float().mean().item()

        metrics = {
            "acc": exact_acc,
            "token_acc": token_acc,
            "exact_acc": exact_acc,
        }
        return loss, metrics

    raise ValueError(meta["task_type"])

# ============================================================
# 6. TRAIN / EVAL
# ============================================================

@torch.no_grad()
def evaluate(model, loader, meta):
    model.eval()

    total_loss = 0.0
    total_n = 0
    metric_sums = {}
    stat_sums = {}
    stat_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits, stats = model(xb, return_stats=True)
        loss, metrics = compute_loss_and_metrics(logits, yb, meta)

        bs = xb.shape[0]
        total_loss += loss.item() * bs
        total_n += bs

        for k, v in metrics.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v * bs

        for k, v in stats.items():
            stat_sums[k] = stat_sums.get(k, 0.0) + v
        if stats:
            stat_count += 1

    out = {"loss": total_loss / total_n}
    for k, v in metric_sums.items():
        out[k] = v / total_n

    if stat_count > 0:
        for k, v in stat_sums.items():
            out[k] = v / stat_count

    return out


def train_one(seed, mode):
    seed_everything(seed)

    X_train, y_train, meta = make_dataset(TASK, TRAIN_N, seed=1000 + seed)
    X_test, y_test, _ = make_dataset(TASK, TEST_N, seed=2000 + seed)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    hidden = choose_hidden(mode, meta["input_dim"], meta["output_dim"])
    model = build_model(mode, meta["input_dim"], meta["output_dim"], hidden).to(device)

    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("\n" + "=" * 80)
    print(f"TASK={TASK} | POSITION={POSITION} | MODE={mode} | SEED={seed}")
    print(f"hidden={hidden} | params={params:,} | target_params={TARGET_PARAMS:,}")
    print("=" * 80)

    best = {
        "epoch": 0,
        "acc": -1.0,
        "metrics": None,
    }

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss, _ = compute_loss_and_metrics(logits, yb, meta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            bs = xb.shape[0]
            train_loss_sum += loss.item() * bs
            train_n += bs

        train_loss = train_loss_sum / train_n
        val = evaluate(model, test_loader, meta)

        if val["acc"] > best["acc"]:
            best = {
                "epoch": epoch,
                "acc": val["acc"],
                "metrics": dict(val),
            }

        if PRINT_EVERY_EPOCH:
            if meta["task_type"] == "single":
                extra = f"id={val['id_acc']:.3f} count={val['count_acc']:.3f}"
            else:
                extra = f"token={val['token_acc']:.3f} exact={val['exact_acc']:.3f}"

            pdn_extra = ""
            if "gate_mean" in val:
                pdn_extra = (
                    f" gate={val['gate_mean']:.3f}"
                    f" pipe={val['pipe_norm']:.2f}"
                    f" vault={val['vault_norm']:.2f}"
                )

            print(
                f"ep {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val['loss']:.4f} | "
                f"acc={val['acc']:.3f} | "
                f"{extra} | "
                f"best={best['acc']:.3f}@{best['epoch']}"
                f"{pdn_extra}"
            )

    elapsed = time.time() - start

    print("-" * 80)
    print(f"FINAL MODE={mode} SEED={seed}")
    print(f"Best acc: {best['acc']:.4f} at epoch {best['epoch']}")
    print(f"Final acc: {val['acc']:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("-" * 80)

    return {
        "mode": mode,
        "seed": seed,
        "params": params,
        "hidden": hidden,
        "best_epoch": best["epoch"],
        "best_acc": best["acc"],
        "final_acc": val["acc"],
        "best_metrics": best["metrics"],
        "final_metrics": val,
    }

# ============================================================
# 7. RUN
# ============================================================

if RUN_MODE == "all_quick":
    modes = [
        "rnn",
        "gru",
        "lstm",
        "pdn_full",
        "pdn_no_subtraction",
        "pdn_no_skip",
        "pdn_no_archive",
        "pdn_no_gate",
        "pdn_leaky_vault",
        "pdn_mlp_head",
    ]
    run_seeds = [SEEDS[0]]
else:
    modes = [RUN_MODE]
    run_seeds = SEEDS

all_results = []

for mode in modes:
    for seed in run_seeds:
        result = train_one(seed, mode)
        all_results.append(result)

print("\n\n" + "#" * 80)
print("SUMMARY")
print("#" * 80)

for mode in sorted(set(r["mode"] for r in all_results)):
    rows = [r for r in all_results if r["mode"] == mode]

    best_accs = np.array([r["best_acc"] for r in rows])
    final_accs = np.array([r["final_acc"] for r in rows])
    params = np.array([r["params"] for r in rows])
    best_epochs = np.array([r["best_epoch"] for r in rows])

    print(f"\nMODE: {mode}")
    print(f"params:     mean={params.mean():.0f}")
    print(f"best_acc:   mean={best_accs.mean():.4f} std={best_accs.std(ddof=0):.4f}")
    print(f"final_acc:  mean={final_accs.mean():.4f} std={final_accs.std(ddof=0):.4f}")
    print(f"best_epoch: mean={best_epochs.mean():.2f}")

print("\nDone. Paste this SUMMARY plus any weird epoch behavior back to me.")
