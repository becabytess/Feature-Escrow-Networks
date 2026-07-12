# ==============================================================================
# Multi-Scale 2D Feature-Escrow Network (FEN) on CIFAR-100
# Task: Image classification under 250k parameter budget
# Sweep: 2-Seed comparison of Plain CNN, ResNet, Subtractive FEN, and Rotational FEN
# Features: Dynamic Kaggle dataset paths, multi-GPU stats gathering, history logging
# ==============================================================================

import os
import time
import random
import csv
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# --- 1. CONFIGURATION ---
MODES = ["plain_cnn", "resnet_baseline", "fen_subtractive", "fen_rotational"]
SEEDS = [100, 200]
EPOCHS = 30
BATCH_SIZE = 512
LR = 2e-3  # Scaled LR for larger batch size
WEIGHT_DECAY = 1e-4

TARGET_PARAMS = 250000  # Strict parameter budget
AUTO_MATCH_PARAMS = True

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

# CIFAR-100 stats
C100_MEAN = (0.5071, 0.4865, 0.4409)
C100_STD = (0.2673, 0.2564, 0.2762)

class CustomCIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def load_cifar100_pickle(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    data = data_dict[b'data']
    labels = data_dict[b'fine_labels']
    data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return data, np.array(labels, dtype=np.int64)

def get_cifar100_paths():
    import glob
    # Recursively search for the 'train' file anywhere inside /kaggle/input/
    train_matches = glob.glob("/kaggle/input/**/train", recursive=True)
    for train_path in train_matches:
        dir_path = os.path.dirname(train_path)
        test_path = os.path.join(dir_path, "test")
        if os.path.exists(test_path):
            return train_path, test_path
            
    # Fallback to local directories
    possible_dirs = [
        "./data/cifar-100-python",
        "./data"
    ]
    for d in possible_dirs:
        train_path = os.path.join(d, "train")
        test_path = os.path.join(d, "test")
        if os.path.exists(train_path) and os.path.exists(test_path):
            return train_path, test_path
    raise FileNotFoundError("Could not find CIFAR-100 train/test pickle files in /kaggle/input/ or local paths.")

def make_loaders(seed):
    train_path, test_path = get_cifar100_paths()
    print(f"Loading CIFAR-100 directly from: {train_path}")
    
    train_data, train_labels = load_cifar100_pickle(train_path)
    test_data, test_labels = load_cifar100_pickle(test_path)
    
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Normalize(C100_MEAN, C100_STD),
    ])
    test_tf = T.Compose([
        T.Normalize(C100_MEAN, C100_STD),
    ])

    train_set = CustomCIFAR100Dataset(train_data, train_labels, transform=train_tf)
    test_set = CustomCIFAR100Dataset(test_data, test_labels, transform=test_tf)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# --- 2. MODEL DEFINITIONS ---

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
        if return_stats: 
            # Return matching tensor type for DataParallel consistency
            stats_tensor = torch.stack([torch.tensor(0.0, device=x.device), h.norm(dim=-1).mean()])
            return logits, stats_tensor
        return logits

class PeristalticConv2d(nn.Module):
    def __init__(self, dim, use_rotation=False):
        super().__init__()
        self.use_rotation = use_rotation
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )
        self.gate = nn.Conv2d(dim, dim, 1)
        self.vault_proj = nn.Conv2d(dim, dim, 1)
        
        if use_rotation:
            self.roll_gate = nn.Conv2d(dim, 1, 1)

    def forward(self, pipe, E, mode):
        f_raw = self.conv(pipe) + pipe
        
        g = torch.sigmoid(self.gate(f_raw))
        D = g * f_raw
        pipe_next = f_raw - D
        v = self.vault_proj(D)
        
        if self.use_rotation:
            gamma = torch.sigmoid(self.roll_gate(f_raw))
            E_next = (1 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=1) + v
        else:
            E_next = E + v
            
        return pipe_next, E_next, g.mean()

class FeatureEscrowCNN(nn.Module):
    def __init__(self, hidden_dim, mode):
        super().__init__()
        self.mode = mode
        self.use_rotation = (mode == "fen_rotational")
        
        self.stem = nn.Conv2d(3, hidden_dim, 3, padding=1, bias=False)
        
        # Stage 1: 32x32
        self.p1_a = PeristalticConv2d(hidden_dim, self.use_rotation)
        self.p1_b = PeristalticConv2d(hidden_dim, self.use_rotation)
        self.pool1 = nn.MaxPool2d(2)
        
        # Stage 2: 16x16
        self.p2_a = PeristalticConv2d(hidden_dim, self.use_rotation)
        self.p2_b = PeristalticConv2d(hidden_dim, self.use_rotation)
        self.pool2 = nn.MaxPool2d(2)
        
        # Stage 3: 8x8
        self.p3_a = PeristalticConv2d(hidden_dim, self.use_rotation)
        self.p3_b = PeristalticConv2d(hidden_dim, self.use_rotation)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(hidden_dim * 4, 100)

    def forward(self, x, return_stats=False):
        B = x.shape[0]
        pipe = self.stem(x)
        
        # Stage 1
        E1 = torch.zeros_like(pipe)
        pipe, E1, g1a = self.p1_a(pipe, E1, self.mode)
        pipe, E1, g1b = self.p1_b(pipe, E1, self.mode)
        vault_32 = E1
        pipe = self.pool1(pipe)
        
        # Stage 2
        E2 = torch.zeros_like(pipe)
        pipe, E2, g2a = self.p2_a(pipe, E2, self.mode)
        pipe, E2, g2b = self.p2_b(pipe, E2, self.mode)
        vault_16 = E2
        pipe = self.pool2(pipe)
        
        # Stage 3
        E3 = torch.zeros_like(pipe)
        pipe, E3, g3a = self.p3_a(pipe, E3, self.mode)
        pipe, E3, g3b = self.p3_b(pipe, E3, self.mode)
        vault_8 = E3
        
        v32_pooled = self.global_pool(vault_32).flatten(1)
        v16_pooled = self.global_pool(vault_16).flatten(1)
        v8_pooled  = self.global_pool(vault_8).flatten(1)
        pipe_pooled = self.global_pool(pipe).flatten(1)
        
        combined = torch.cat([v32_pooled, v16_pooled, v8_pooled, pipe_pooled], dim=-1)
        logits = self.head(combined)
        
        if return_stats:
            gate_mean = (g1a + g1b + g2a + g2b + g3a + g3b) / 6.0
            # Return stats as a gathered tensor to support DataParallel without OOMs
            stats_tensor = torch.stack([gate_mean, pipe_pooled.norm(dim=-1).mean()])
            return logits, stats_tensor
            
        return logits

# --- 3. PARAMETER MATCHING ---

def build_model(mode, h):
    if mode == "plain_cnn": 
        return CNNBaseline(h, use_residual=False)
    elif mode == "resnet_baseline": 
        return CNNBaseline(h, use_residual=True)
    elif mode in ["fen_subtractive", "fen_rotational"]: 
        return FeatureEscrowCNN(h, mode)
    raise ValueError(mode)

HIDDEN_SIZES_CACHE = {}

def choose_hidden(mode):
    if mode in HIDDEN_SIZES_CACHE:
        return HIDDEN_SIZES_CACHE[mode]
        
    best_h, best_diff = 16, float('inf')
    for h in range(16, 256):
        model = build_model(mode, h)
        diff = abs(count_params(model) - TARGET_PARAMS)
        if diff < best_diff:
            best_h, best_diff = h, diff
            
    HIDDEN_SIZES_CACHE[mode] = best_h
    return best_h

def save_history_to_csv(history, filepath="cifar100_history.csv"):
    file_exists = os.path.exists(filepath)
    keys = history[0].keys()
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(history)

# --- 4. TRAINING & EVALUATION ---

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
        
        if stats is not None:
            # Average out stats across GPUs if wrapped in nn.DataParallel
            if stats.dim() > 1:
                stats = stats.mean(dim=0)
            gate_sum += stats[0].item()
            pipe_sum += stats[1].item()
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
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via nn.DataParallel!")
        model = nn.DataParallel(model)
        
    print("\n" + "=" * 80)
    print(f"MODE={mode.upper()} | SEED={seed} | Hidden={hidden} | Params={params:,}")
    print("=" * 80)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    start = time.time()
    history = []

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
        
        if val["acc"] > best_acc: 
            best_acc = val["acc"]
        
        gate_str = f" gate={val['gate']:.3f} pipe={val['pipe']:.2f}" if "gate" in val else ""
        print(f"  Ep {epoch:02d} | Val Loss: {val['loss']:.4f} | Val Acc: {val['acc']*100:.2f}% | Best Acc: {best_acc*100:.2f}%{gate_str}")

        history.append({
            "seed": seed,
            "mode": mode,
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_acc": val["acc"] * 100,
            "gate": val.get("gate", 0.0),
            "pipe": val.get("pipe", 0.0)
        })

    print(f"  Finished {mode.upper()} | Best Acc: {best_acc*100:.2f}% | Time: {time.time()-start:.1f}s")
    save_history_to_csv(history)
    return {"mode": mode, "params": params, "best_acc": best_acc}

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    results = {mode: [] for mode in MODES}
    
    print("=" * 90)
    print(f"STARTING 2-SEED ALTERNATING CIFAR-100 SWEEP | TARGET PARAMS = {TARGET_PARAMS}")
    print("=" * 90)
    
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n================================================================================")
        print(f"SEED BATCH {s_idx+1}/{len(SEEDS)}: SEED = {seed}")
        print(f"================================================================================")
        
        for mode in MODES:
            res = train_one(seed, mode)
            results[mode].append(res)
            
        print(f"\n--- RUNNING RESULTS SUMMARY (Completed Seeds: {s_idx+1}/{len(SEEDS)}) ---")
        print(f"{'Model':<16} | {'Val Acc (Mean)':<16} | {'Completed Runs'}")
        print("-" * 55)
        for mode in MODES:
            accs = [r["best_acc"] * 100 for r in results[mode]]
            print(f"{mode.upper():<16} | {np.mean(accs):.2f}% (±{np.std(accs):.2f}) | {len(accs)}/2")
        print("-" * 55)
        
    print("\n\n" + "#" * 90)
    print("FINAL CIFAR-100 SWEEP COMPLETE SUMMARY")
    print("#" * 90)
    for mode in MODES:
        accs = [r["best_acc"] * 100 for r in results[mode]]
        print(f"MODE: {mode.upper():<16} | Val Acc: {np.mean(accs):.2f}% (±{np.std(accs):.2f})")
    print("#" * 90)