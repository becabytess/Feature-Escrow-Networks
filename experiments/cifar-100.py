# ============================================================
# Multi-Scale Feature-Escrow Network (FEN) on CIFAR-100
# Reproducing the 4-way Abstractive Bottleneck Table
# ============================================================

import os, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# ============================================================
# 1. EDIT CONFIG
# ============================================================

RUN_MODE = "all_quick"
# Options:
#   "plain_cnn"        -> No residuals in baseline (59.58%)
#   "resnet_baseline"  -> Residuals in baseline (55.56%)
#   "fen_copy_only"    -> FEN without subtraction (57.11%)
#   "fen_full"         -> FEN with subtraction (61.50%)
#   "all_quick"        -> Runs all 4 sequentially

SEEDS = [1, 2]
EPOCHS = 35
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4

TARGET_PARAMS = 250000  # Hard bottleneck regime for CIFAR-100
AUTO_MATCH_PARAMS = True

# ============================================================
# 2. SETUP & DATA
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# CIFAR-100 exact stats
C100_MEAN = (0.5071, 0.4865, 0.4409)
C100_STD = (0.2673, 0.2564, 0.2762)

def make_loaders(seed):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(C100_MEAN, C100_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(C100_MEAN, C100_STD),
    ])

    train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ============================================================
# 3. MODELS
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        if self.use_residual:
            return self.net(x) + x
        return self.net(x)

class CNNBaseline(nn.Module):
    def __init__(self, hidden_dim, use_residual=False):
        super().__init__()
        self.stem = nn.Conv2d(3, hidden_dim, 3, padding=1, bias=False)
        
        self.stage1 = nn.Sequential(ConvBNAct(hidden_dim, hidden_dim, use_residual), ConvBNAct(hidden_dim, hidden_dim, use_residual))
        self.pool1 = nn.MaxPool2d(2) 
        
        self.stage2 = nn.Sequential(ConvBNAct(hidden_dim, hidden_dim, use_residual), ConvBNAct(hidden_dim, hidden_dim, use_residual))
        self.pool2 = nn.MaxPool2d(2) 
        
        self.stage3 = nn.Sequential(ConvBNAct(hidden_dim, hidden_dim, use_residual), ConvBNAct(hidden_dim, hidden_dim, use_residual))
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Linear(hidden_dim, 100)

    def forward(self, x, return_stats=False):
        h = self.stem(x)
        h = self.pool1(self.stage1(h))
        h = self.pool2(self.stage2(h))
        h = self.pool3(self.stage3(h)).flatten(1)
        
        logits = self.head(h)
        if return_stats: return logits, {}
        return logits

class PeristalticConv2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )
        self.gate = nn.Conv2d(dim, dim, 1)
        self.vault_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, pipe, mode):
        # Active transformation + internal residual
        f_raw = self.conv(pipe) + pipe
        
        # Escrow Gate
        g = torch.sigmoid(self.gate(f_raw))
        D = g * f_raw
        
        # Subtractive Routing
        if mode == "fen_copy_only":
            pipe_next = f_raw
        else:
            pipe_next = f_raw - D
            
        v = self.vault_proj(D)
        
        return pipe_next, v, g.mean().detach()

class FeatureEscrowCNN(nn.Module):
    def __init__(self, hidden_dim, mode):
        super().__init__()
        self.mode = mode
        self.stem = nn.Conv2d(3, hidden_dim, 3, padding=1, bias=False)
        
        # Stage 1: 32x32
        self.p1_a = PeristalticConv2d(hidden_dim)
        self.p1_b = PeristalticConv2d(hidden_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        # Stage 2: 16x16
        self.p2_a = PeristalticConv2d(hidden_dim)
        self.p2_b = PeristalticConv2d(hidden_dim)
        self.pool2 = nn.MaxPool2d(2)
        
        # Stage 3: 8x8
        self.p3_a = PeristalticConv2d(hidden_dim)
        self.p3_b = PeristalticConv2d(hidden_dim)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(hidden_dim * 4, 100)

    def forward(self, x, return_stats=False):
        B = x.shape[0]
        pipe = self.stem(x)
        
        pipe, v1a, g1a = self.p1_a(pipe, self.mode)
        pipe, v1b, g1b = self.p1_b(pipe, self.mode)
        vault_32 = v1a + v1b
        pipe = self.pool1(pipe)
        
        pipe, v2a, g2a = self.p2_a(pipe, self.mode)
        pipe, v2b, g2b = self.p2_b(pipe, self.mode)
        vault_16 = v2a + v2b
        pipe = self.pool2(pipe)
        
        pipe, v3a, g3a = self.p3_a(pipe, self.mode)
        pipe, v3b, g3b = self.p3_b(pipe, self.mode)
        vault_8 = v3a + v3b
        
        v32_pooled = self.global_pool(vault_32).flatten(1)
        v16_pooled = self.global_pool(vault_16).flatten(1)
        v8_pooled  = self.global_pool(vault_8).flatten(1)
        pipe_pooled = self.global_pool(pipe).flatten(1)
        
        combined = torch.cat([v32_pooled, v16_pooled, v8_pooled, pipe_pooled], dim=-1)
        logits = self.head(combined)
        
        if return_stats:
            gate_mean = (g1a + g1b + g2a + g2b + g3a + g3b) / 6.0
            return logits, {"gate_mean": gate_mean, "pipe_norm": pipe_pooled.norm(dim=-1).mean().item()}
            
        return logits

def build_model(mode, h):
    if mode == "plain_cnn": return CNNBaseline(h, use_residual=False)
    if mode == "resnet_baseline": return CNNBaseline(h, use_residual=True)
    if mode in ["fen_copy_only", "fen_full"]: return FeatureEscrowCNN(h, mode)
    raise ValueError(mode)

def choose_hidden(mode):
    best_h, best_diff = 16, float('inf')
    for h in range(16, 128):
        model = build_model(mode, h)
        diff = abs(count_params(model) - TARGET_PARAMS)
        if diff < best_diff:
            best_h, best_diff = h, diff
    return best_h

# ============================================================
# 4. TRAINING LOOP
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, acc_sum, total = 0.0, 0.0, 0
    gate_sum, pipe_sum, stat_n = 0.0, 0.0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, stats = model(xb, return_stats=True)
        loss = criterion(logits, yb)
        
        bs = xb.shape[0]
        loss_sum += loss.item() * bs
        acc_sum += (logits.argmax(-1) == yb).sum().item()
        total += bs
        
        if stats:
            gate_sum += stats.get("gate_mean", 0)
            pipe_sum += stats.get("pipe_norm", 0)
            stat_n += 1

    out = {"loss": loss_sum / total, "acc": acc_sum / total}
    if stat_n:
        out["gate"] = gate_sum / stat_n
        out["pipe"] = pipe_sum / stat_n
    return out

def train_one(seed, mode):
    seed_everything(seed)
    train_loader, test_loader = make_loaders(seed)
    
    hidden = choose_hidden(mode)
    model = build_model(mode, hidden).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"MODE={mode} | SEED={seed} | hidden={hidden} | params={params:,}")
    print("=" * 80)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(xb)
                loss = criterion(logits, yb)
                
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
        scheduler.step()
        val = evaluate(model, test_loader, criterion)
        
        if val["acc"] > best_acc: best_acc = val["acc"]
        
        gate_str = f" gate={val['gate']:.3f} pipe={val['pipe']:.2f}" if "gate" in val else ""
        print(f"ep {epoch:02d} | val_loss={val['loss']:.4f} | val_acc={val['acc']:.4f} | best={best_acc:.4f}{gate_str}")

    print(f"FINAL {mode}: Best Acc {best_acc:.4f} | Time: {time.time()-start:.1f}s")
    return {"mode": mode, "params": params, "best_acc": best_acc}

# ============================================================
# 5. EXECUTION
# ============================================================

if RUN_MODE == "all_quick":
    modes = ["plain_cnn", "resnet_baseline", "fen_copy_only", "fen_full"]
else:
    modes = [RUN_MODE]

results = []
for mode in modes:
    for seed in SEEDS:
        results.append(train_one(seed, mode))

print("\n\n" + "#" * 80)
print("SUMMARY")
print("#" * 80)
for mode in sorted(set(r["mode"] for r in results)):
    rows = [r for r in results if r["mode"] == mode]
    print(f"\nMODE: {mode}")
    print(f"params:       mean={np.mean([r['params'] for r in rows]):.0f}")
    print(f"best_acc:     mean={np.mean([r['best_acc'] for r in rows]):.4f}")

print("\nDone.")