# ============================================================
# Patch-FEN Image Prototype
# Synthetic glyph image task, no downloads needed.
# ============================================================

import os, time, random, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 1. EDIT CONFIG
# ============================================================

TASK = "first_match"
# Options:
#   "first_match"   -> patch 0 is a query digit; output first later patch matching it
#   "marked_recall" -> 5 marked digit patches; output their digits in scan order

RUN_MODE = "patch_FEN_spatial"
# Options:
#   "patch_rnn"
#   "patch_gru"
#   "patch_lstm"
#   "patch_FEN_global"
#   "patch_FEN_global_no_subtraction"
#   "patch_FEN_global_no_skip"
#   "patch_FEN_spatial"
#   "patch_FEN_spatial_no_subtraction"
#   "patch_FEN_spatial_no_archive"
#   "all_quick"

SEEDS = [1, 2, 3]
EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.0

TRAIN_N = 12000
TEST_N = 3000

GRID = 4
PATCH = 8
IMG_SIZE = GRID * PATCH
N_PATCHES = GRID * GRID
PATCH_DIM = PATCH * PATCH

NUM_DIGITS = 10
NUM_TARGETS = 5
NOISE_STD = 0.10
ABSENT_PROB = 0.25

TARGET_PARAMS = 80000
AUTO_MATCH_PARAMS = True
MIN_HIDDEN = 8
MAX_HIDDEN = 192
READOUT_WIDTH = 128

PRINT_EVERY_EPOCH = True
SHOW_EXAMPLES = False

# ============================================================
# 2. DEVICE / SEED
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
# 3. GLYPHS
# ============================================================

def make_digit_glyphs():
    segs = {
        0: "abcedf".replace("c", "c"),
        1: "bc",
        2: "abged",
        3: "abgcd",
        4: "fgbc",
        5: "afgcd",
        6: "afgecd",
        7: "abc",
        8: "abcdefg",
        9: "abfgcd",
    }

    # Canonical seven-segment map.
    segs = {
        0: ["a", "b", "c", "d", "e", "f"],
        1: ["b", "c"],
        2: ["a", "b", "g", "e", "d"],
        3: ["a", "b", "g", "c", "d"],
        4: ["f", "g", "b", "c"],
        5: ["a", "f", "g", "c", "d"],
        6: ["a", "f", "g", "e", "c", "d"],
        7: ["a", "b", "c"],
        8: ["a", "b", "c", "d", "e", "f", "g"],
        9: ["a", "b", "c", "d", "f", "g"],
    }

    glyphs = np.zeros((10, PATCH, PATCH), dtype=np.float32)

    def draw(g, s):
        if s == "a": g[1, 2:6] = 1
        if s == "b": g[2:4, 6] = 1
        if s == "c": g[4:7, 6] = 1
        if s == "d": g[6, 2:6] = 1
        if s == "e": g[4:7, 1] = 1
        if s == "f": g[2:4, 1] = 1
        if s == "g": g[3, 2:6] = 1

    for d in range(10):
        g = glyphs[d]
        for s in segs[d]:
            draw(g, s)

    return glyphs

GLYPHS = make_digit_glyphs()

def add_digit_patch(patch, digit, rng, strength=1.0):
    brightness = rng.normal(strength, 0.08)
    patch += brightness * GLYPHS[digit]

def add_border(patch, value=1.0):
    patch[0, :] += value
    patch[-1, :] += value
    patch[:, 0] += value
    patch[:, -1] += value

def make_patch_sequence_from_patches(patches):
    return patches.reshape(N_PATCHES, PATCH_DIM)

# ============================================================
# 4. DATASETS
# ============================================================

def make_first_match_dataset(n, seed):
    """
    Balanced image reasoning task:
      - patch 0 contains a query digit.
      - patches 1..15 contain candidate digits.
      - labels are uniformly balanced across:
          0..14 -> first matching patch positions 1..15
          15    -> absent

    This removes the majority-class absent shortcut.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, N_PATCHES, PATCH_DIM), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    absent_class = N_PATCHES - 1

    for i in range(n):
        patches = rng.normal(0.0, NOISE_STD, size=(N_PATCHES, PATCH, PATCH)).astype(np.float32)

        q = rng.integers(0, NUM_DIGITS)
        add_digit_patch(patches[0], q, rng, strength=1.05)
        add_border(patches[0], value=0.75)

        label = rng.integers(0, N_PATCHES)
        y[i] = label

        if label == absent_class:
            for pos in range(1, N_PATCHES):
                d = rng.integers(0, NUM_DIGITS - 1)
                if d >= q:
                    d += 1
                add_digit_patch(patches[pos], d, rng, strength=0.95)
        else:
            match_pos = label + 1

            for pos in range(1, N_PATCHES):
                if pos < match_pos:
                    d = rng.integers(0, NUM_DIGITS - 1)
                    if d >= q:
                        d += 1
                    add_digit_patch(patches[pos], d, rng, strength=0.95)

                elif pos == match_pos:
                    add_digit_patch(patches[pos], q, rng, strength=0.95)

                else:
                    # After the first match, anything can appear, including more matches.
                    d = rng.integers(0, NUM_DIGITS)
                    add_digit_patch(patches[pos], d, rng, strength=0.95)

        patches = np.clip(patches, 0.0, 1.5)
        X[i] = make_patch_sequence_from_patches(patches)

    meta = {
        "task_type": "single",
        "input_dim": PATCH_DIM,
        "output_dim": N_PATCHES,
        "n_patches": N_PATCHES,
        "description": "balanced first matching candidate position or absent",
    }

    return torch.tensor(X), torch.tensor(y), meta


def make_marked_recall_dataset(n, seed):
    """
    Image binding task:
      - 5 randomly placed patches are marked with a border.
      - output the marked digits in scan order.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, N_PATCHES, PATCH_DIM), dtype=np.float32)
    y = np.zeros((n, NUM_TARGETS), dtype=np.int64)

    for i in range(n):
        patches = rng.normal(0.0, NOISE_STD, size=(N_PATCHES, PATCH, PATCH)).astype(np.float32)

        target_positions = np.sort(
            rng.choice(np.arange(N_PATCHES), size=NUM_TARGETS, replace=False)
        )
        target_digits = rng.integers(0, NUM_DIGITS, size=NUM_TARGETS)
        y[i] = target_digits

        target_set = set(target_positions.tolist())

        for pos in range(N_PATCHES):
            if pos in target_set:
                idx = np.where(target_positions == pos)[0][0]
                add_digit_patch(patches[pos], int(target_digits[idx]), rng, strength=1.05)
                add_border(patches[pos], value=0.75)
            else:
                if rng.random() < 0.85:
                    d = rng.integers(0, NUM_DIGITS)
                    add_digit_patch(patches[pos], d, rng, strength=0.75)

        patches = np.clip(patches, 0.0, 1.5)
        X[i] = make_patch_sequence_from_patches(patches)

    meta = {
        "task_type": "recall",
        "input_dim": PATCH_DIM,
        "output_dim": NUM_TARGETS * NUM_DIGITS,
        "slots": NUM_TARGETS,
        "vocab": NUM_DIGITS,
        "n_patches": N_PATCHES,
        "description": "marked digit recall in scan order",
    }

    return torch.tensor(X), torch.tensor(y), meta


def make_dataset(task, n, seed):
    if task == "first_match":
        return make_first_match_dataset(n, seed)
    if task == "marked_recall":
        return make_marked_recall_dataset(n, seed)
    raise ValueError(f"Unknown TASK: {task}")

# ============================================================
# 5. MODELS
# ============================================================

def make_head(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, READOUT_WIDTH),
        nn.ReLU(),
        nn.Linear(READOUT_WIDTH, out_dim),
    )

class PatchBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, kind):
        super().__init__()
        self.kind = kind

        if kind == "patch_rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        elif kind == "patch_gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif kind == "patch_lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(kind)

        self.head = make_head(hidden_dim, output_dim)

    def forward(self, x, return_stats=False):
        out, state = self.rnn(x)

        if self.kind == "patch_lstm":
            h = state[0][-1]
        else:
            h = state[-1]

        logits = self.head(h)

        if return_stats:
            return logits, {"state_norm": h.norm(dim=-1).mean().item()}
        return logits


class PatchFEN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, mode, meta):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.n_patches = int(meta["n_patches"])

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.vault_proj = nn.Linear(hidden_dim, hidden_dim)
        self.vault_recur = nn.Linear(hidden_dim, hidden_dim)

        if "spatial" in mode:
            head_in = hidden_dim + self.n_patches * hidden_dim
        else:
            head_in = hidden_dim * 2

        self.head = make_head(head_in, output_dim)

    def forward(self, x, return_stats=False):
        batch = x.shape[0]
        h = torch.zeros(batch, self.hidden_dim, device=x.device)

        is_spatial = "spatial" in self.mode

        if is_spatial:
            S = torch.zeros(batch, self.n_patches, self.hidden_dim, device=x.device)
        else:
            S = torch.zeros(batch, self.hidden_dim, device=x.device)

        gate_means = []
        diffuse_norms = []
        raw_norms = []

        no_subtraction = self.mode.endswith("_no_subtraction")
        no_archive = self.mode.endswith("_no_archive")
        no_skip = self.mode.endswith("_no_skip")

        for i in range(x.shape[1]):
            xt = self.x_proj(x[:, i])
            z = h + xt
            h_raw = torch.tanh(self.core(z) + z)

            g = torch.sigmoid(self.gate(h_raw))
            D = g * h_raw

            if no_subtraction:
                h = h_raw
            else:
                h = h_raw - D

            update = self.vault_proj(D)

            if no_archive:
                pass
            elif is_spatial:
                if no_skip:
                    S[:, i, :] = torch.tanh(self.vault_recur(S[:, i, :]) + update)
                else:
                    S[:, i, :] = S[:, i, :] + update
            else:
                if no_skip:
                    S = torch.tanh(self.vault_recur(S) + update)
                else:
                    S = S + update

            if return_stats:
                gate_means.append(g.mean().detach())
                diffuse_norms.append(D.norm(dim=-1).mean().detach())
                raw_norms.append(h_raw.norm(dim=-1).mean().detach())

        if is_spatial:
            S_flat = S.reshape(batch, self.n_patches * self.hidden_dim)
            combined = torch.cat([h, S_flat], dim=-1)
            vault_norm = S_flat.norm(dim=-1).mean().item()
        else:
            combined = torch.cat([h, S], dim=-1)
            vault_norm = S.norm(dim=-1).mean().item()

        logits = self.head(combined)

        if return_stats:
            stats = {
                "gate_mean": torch.stack(gate_means).mean().item(),
                "diffuse_norm": torch.stack(diffuse_norms).mean().item(),
                "raw_norm": torch.stack(raw_norms).mean().item(),
                "pipe_norm": h.norm(dim=-1).mean().item(),
                "vault_norm": vault_norm,
            }
            return logits, stats

        return logits


def build_model(mode, input_dim, output_dim, hidden_dim, meta):
    if mode in ["patch_rnn", "patch_gru", "patch_lstm"]:
        return PatchBaseline(input_dim, output_dim, hidden_dim, mode)
    if mode.startswith("patch_FEN"):
        return PatchFEN(input_dim, output_dim, hidden_dim, mode, meta)
    raise ValueError(f"Unknown RUN_MODE: {mode}")


def choose_hidden(mode, input_dim, output_dim, meta):
    if not AUTO_MATCH_PARAMS:
        return 64

    best_h, best_diff = None, None

    for h in range(MIN_HIDDEN, MAX_HIDDEN + 1):
        model = build_model(mode, input_dim, output_dim, h, meta)
        params = count_params(model)
        diff = abs(params - TARGET_PARAMS)

        if best_diff is None or diff < best_diff:
            best_h, best_diff = h, diff

    return best_h

# ============================================================
# 6. LOSS / METRICS
# ============================================================

def compute_loss_and_metrics(logits, y, meta):
    if meta["task_type"] == "single":
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        return loss, {"acc": acc}

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

        return loss, {
            "acc": exact_acc,
            "exact_acc": exact_acc,
            "token_acc": token_acc,
        }

    raise ValueError(meta["task_type"])

# ============================================================
# 7. TRAIN / EVAL
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

        if stats:
            stat_count += 1
            for k, v in stats.items():
                stat_sums[k] = stat_sums.get(k, 0.0) + v

    out = {"loss": total_loss / total_n}
    for k, v in metric_sums.items():
        out[k] = v / total_n

    if stat_count:
        for k, v in stat_sums.items():
            out[k] = v / stat_count

    return out


def train_one(seed, mode):
    seed_everything(seed)

    X_train, y_train, meta = make_dataset(TASK, TRAIN_N, seed=1000 + seed)
    X_test, y_test, _ = make_dataset(TASK, TEST_N, seed=2000 + seed)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    hidden = choose_hidden(mode, meta["input_dim"], meta["output_dim"], meta)
    model = build_model(mode, meta["input_dim"], meta["output_dim"], hidden, meta).to(device)

    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("\n" + "=" * 80)
    print(f"TASK={TASK} | MODE={mode} | SEED={seed}")
    print(f"hidden={hidden} | params={params:,} | target_params={TARGET_PARAMS:,}")
    print("=" * 80)

    best = {"epoch": 0, "acc": -1.0, "metrics": None}
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
            best = {"epoch": epoch, "acc": val["acc"], "metrics": dict(val)}

        if PRINT_EVERY_EPOCH:
            if meta["task_type"] == "single":
                extra = f"acc={val['acc']:.3f}"
            else:
                extra = f"exact={val['exact_acc']:.3f} token={val['token_acc']:.3f}"

            FEN_extra = ""
            if "gate_mean" in val:
                FEN_extra = (
                    f" gate={val['gate_mean']:.3f}"
                    f" pipe={val['pipe_norm']:.2f}"
                    f" vault={val['vault_norm']:.2f}"
                )

            print(
                f"ep {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val['loss']:.4f} | "
                f"{extra} | "
                f"best={best['acc']:.3f}@{best['epoch']}"
                f"{FEN_extra}"
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
# 8. RUN
# ============================================================

if RUN_MODE == "all_quick":
    modes = [
        "patch_rnn",
        "patch_gru",
        "patch_lstm",
        "patch_FEN_global",
        "patch_FEN_global_no_subtraction",
        "patch_FEN_global_no_skip",
        "patch_FEN_spatial",
        "patch_FEN_spatial_no_subtraction",
        "patch_FEN_spatial_no_archive",
    ]
    run_seeds = [SEEDS[0]]
else:
    modes = [RUN_MODE]
    run_seeds = SEEDS

all_results = []

for mode in modes:
    for seed in run_seeds:
        all_results.append(train_one(seed, mode))

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

    if rows[0]["best_metrics"] is not None:
        for k in sorted(rows[0]["best_metrics"].keys()):
            if k in ["loss", "gate_mean", "diffuse_norm", "raw_norm", "pipe_norm", "vault_norm"]:
                continue
            vals = np.array([r["best_metrics"].get(k, np.nan) for r in rows])
            print(f"best_{k}:  mean={np.nanmean(vals):.4f}")

