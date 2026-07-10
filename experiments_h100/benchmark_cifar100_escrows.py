# ==============================================================================
# CIFAR-100 spatial vs. Global Feedforward FEN Sweep (H100 Optimized)
# Sweep: Plain CNN, ResNet, Spatial FEN, Global FEN, and Rotational FEN
# Setup: Fixed width (hidden_dim = 256), 100 epochs, kagglehub download, BF16 AMP.
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
import torchvision.transforms as T
import kagglehub

# --- 1. CONFIGURATION ---
MODES = ["plain_cnn", "resnet_baseline", "fen_spatial", "fen_global", "fen_rotational"]
SEEDS = [100, 200]
EPOCHS = 100
BATCH_SIZE = 512
LR = 2e-3  
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | CIFAR-100 Escrows Sweep | Width: {HIDDEN_DIM}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 2. DATA LOADERS (KAGGLEHUB) ---
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
    data = data_dict[b'data'] if b'data' in data_dict else data_dict['data']
    labels = data_dict[b'fine_labels'] if b'fine_labels' in data_dict else data_dict['fine_labels']
    data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return data, np.array(labels, dtype=np.int64)

def make_loaders(seed):
    print("Downloading CIFAR-100 via kagglehub...")
    path = kagglehub.dataset_download("fedesoriano/cifar100")
    print(f"Dataset path: {path}")
    
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    
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
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# --- 3. MODEL DEFINITIONS ---

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
            return logits, torch.stack([torch.tensor(0.0, device=x.device), h.norm(dim=-1).mean()])
        return logits

# --- Spatial FEN Block ---
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

    def forward(self, pipe, E):
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

# --- Global Vector FEN Block ---
class GlobalPeristalticConv2d(nn.Module):
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

    def forward(self, pipe, E):
        f_raw = self.conv(pipe) + pipe
        g = torch.sigmoid(self.gate(f_raw))
        D = g * f_raw
        pipe_next = f_raw - D
        
        v = self.vault_proj(D)
        v_pooled = torch.nn.functional.adaptive_avg_pool2d(v, 1).flatten(1)
        
        if self.use_rotation:
            gamma = torch.sigmoid(self.roll_gate(f_raw).mean(dim=[-2, -1]))
            E_next = (1 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v_pooled
        else:
            E_next = E + v_pooled
        return pipe_next, E_next, g.mean()

class FeatureEscrowCNN(nn.Module):
    def __init__(self, hidden_dim, mode):
        super().__init__()
        self.mode = mode
        self.use_rotation = (mode == "fen_rotational")
        self.is_global = (mode == "fen_global")
        
        self.stem = nn.Conv2d(3, hidden_dim, 3, padding=1, bias=False)
        
        # Stage 1: 32x32
        Block = GlobalPeristalticConv2d if self.is_global else PeristalticConv2d
        self.p1_a = Block(hidden_dim, self.use_rotation)
        self.p1_b = Block(hidden_dim, self.use_rotation)
        self.pool1 = nn.MaxPool2d(2)
        
        # Stage 2: 16x16
        self.p2_a = Block(hidden_dim, self.use_rotation)
        self.p2_b = Block(hidden_dim, self.use_rotation)
        self.pool2 = nn.MaxPool2d(2)
        
        # Stage 3: 8x8
        self.p3_a = Block(hidden_dim, self.use_rotation)
        self.p3_b = Block(hidden_dim, self.use_rotation)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        if self.is_global:
            self.head = nn.Linear(hidden_dim * 2, 100)
        else:
            self.head = nn.Linear(hidden_dim * 4, 100)

    def forward(self, x, return_stats=False):
        B = x.shape[0]
        pipe = self.stem(x)
        
        if self.is_global:
            E = torch.zeros(B, pipe.size(1), device=x.device)
            pipe, E, g1a = self.p1_a(pipe, E)
            pipe, E, g1b = self.p1_b(pipe, E)
            pipe = self.pool1(pipe)
            pipe, E, g2a = self.p2_a(pipe, E)
            pipe, E, g2b = self.p2_b(pipe, E)
            pipe = self.pool2(pipe)
            pipe, E, g3a = self.p3_a(pipe, E)
            pipe, E, g3b = self.p3_b(pipe, E)
            
            pipe_pooled = self.global_pool(pipe).flatten(1)
            combined = torch.cat([E, pipe_pooled], dim=-1)
        else:
            E1 = torch.zeros_like(pipe)
            pipe, E1, g1a = self.p1_a(pipe, E1)
            pipe, E1, g1b = self.p1_b(pipe, E1)
            vault_32 = E1
            pipe = self.pool1(pipe)
            
            E2 = torch.zeros_like(pipe)
            pipe, E2, g2a = self.p2_a(pipe, E2)
            pipe, E2, g2b = self.p2_b(pipe, E2)
            vault_16 = E2
            pipe = self.pool2(pipe)
            
            E3 = torch.zeros_like(pipe)
            pipe, E3, g3a = self.p3_a(pipe, E3)
            pipe, E3, g3b = self.p3_b(pipe, E3)
            vault_8 = E3
            
            v32_pooled = self.global_pool(vault_32).flatten(1)
            v16_pooled = self.global_pool(vault_16).flatten(1)
            v8_pooled  = self.global_pool(vault_8).flatten(1)
            pipe_pooled = self.global_pool(pipe).flatten(1)
            combined = torch.cat([v32_pooled, v16_pooled, v8_pooled, pipe_pooled], dim=-1)
            
        logits = self.head(combined)
        
        if return_stats:
            gate_mean = (g1a + g1b + g2a + g2b + g3a + g3b) / 6.0
            stats_tensor = torch.stack([gate_mean, pipe_pooled.norm(dim=-1).mean()])
            return logits, stats_tensor
        return logits

# --- 4. SWEEP RUNNER ---
def build_model(mode, hidden_dim):
    if mode == "plain_cnn": 
        return CNNBaseline(hidden_dim, use_residual=False)
    elif mode == "resnet_baseline": 
        return CNNBaseline(hidden_dim, use_residual=True)
    elif mode in ["fen_spatial", "fen_global", "fen_rotational"]: 
        return FeatureEscrowCNN(hidden_dim, mode)
    raise ValueError(mode)

def save_history_to_csv(history, filepath="experiments_h100/cifar100_escrows_history.csv"):
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    file_exists = os.path.exists(filepath)
    keys = history[0].keys()
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(history)

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
            gate_sum += stats[0].item()
            pipe_sum += stats[1].item()
            stat_n += 1

    out = {"loss": loss_sum / total, "acc": acc_sum / total}
    if stat_n:
        out["gate"] = gate_sum / stat_n
        out["pipe"] = pipe_sum / stat_n
    return out

def choose_hidden(mode, target_params):
    best_h = 16
    best_diff = float('inf')
    # Sweep step size 4 to ensure division compatibility with head counts
    for h in range(16, 512, 4):
        model = build_model(mode, h)
        params = count_params(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_h = h
            best_diff = diff
    return best_h

TARGET_PARAMS = 4450000

def train_one(seed, mode, train_loader, test_loader):
    seed_everything(seed)
    h_dim = choose_hidden(mode, TARGET_PARAMS)
    model = build_model(mode, h_dim).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"CIFAR-100 | MODE={mode.upper()} | SEED={seed} | Width={h_dim} | Params={params:,}")
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
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
    return {"mode": mode, "best_acc": best_acc}

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PARAMETER COUNT SUMMARY:")
    for mode in MODES:
        h_dim = choose_hidden(mode, TARGET_PARAMS)
        model_test = build_model(mode, h_dim)
        print(f"  {mode.upper():<16} (Width={h_dim:<3}) : {count_params(model_test):,} parameters")
    print("=" * 80 + "\n")

    train_loader, test_loader = make_loaders(seed=42)
    results = {mode: [] for mode in MODES}
    
    print("\n" + "#" * 80)
    print(f"STARTING CIFAR-100 SWEEP | TARGET_PARAMS = {TARGET_PARAMS:,} | EPOCHS = {EPOCHS}")
    print("#" * 80)
    
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n================================================================================")
        print(f"SEED BATCH {s_idx+1}/{len(SEEDS)}: SEED = {seed}")
        print(f"================================================================================")
        
        for mode in MODES:
            res = train_one(seed, mode, train_loader, test_loader)
            results[mode].append(res)
