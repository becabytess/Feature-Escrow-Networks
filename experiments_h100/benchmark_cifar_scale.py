# ==============================================================================
# CIFAR-100 Scale Sweep: 12 and 18 blocks, 12M Parameters
# Dynamic parameter matching at startup.
# ==============================================================================

import os
import time
import random
import csv
import urllib.request
import tarfile
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURATION ---
DEPTHS = [12, 18]  # 12 blocks (4/stage) and 18 blocks (6/stage)
MODES = ["plain_cnn", "resnet_baseline", "fen_spatial", "fen_global"]
SEEDS = [100]
EPOCHS = 100
BATCH_SIZE = 256  # Larger batch size to saturate H100
LR = 1e-3
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | CIFAR-100 Scale Sweep | Depths: {DEPTHS}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# --- 2. DATA LOADERS ---
C100_MEAN = (0.5071, 0.4867, 0.4408)
C100_STD = (0.2675, 0.2565, 0.2761)

class CustomCIFAR100Dataset(Dataset):
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

def make_loaders(seed=42):
    import kagglehub
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
    val_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

# --- 3. MODEL ARCHITECTURES ---
class CNNBlock(nn.Module):
    def __init__(self, dim, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out + x if self.use_residual else out

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

    def forward(self, pipe, E):
        f_raw = self.conv(pipe) + pipe
        g = torch.sigmoid(self.gate(f_raw))
        D = g * f_raw
        pipe_next = f_raw - D
        v = self.vault_proj(D)
        E_next = E + v
        return pipe_next, E_next

class GlobalPeristalticConv2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )
        self.gate = nn.Conv2d(dim, dim, 1)
        self.vault_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, pipe, E):
        f_raw = self.conv(pipe) + pipe
        g = torch.sigmoid(self.gate(f_raw))
        D = g * f_raw
        pipe_next = f_raw - D
        v = self.vault_proj(D)
        v_pooled = torch.nn.functional.adaptive_avg_pool2d(v, 1).flatten(1)
        E_next = E + v_pooled
        return pipe_next, E_next

class CNNBaseline(nn.Module):
    def __init__(self, dim, num_layers, use_residual=False):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        
        # 3 stages
        layers_per_stage = num_layers // 3
        self.stage1 = nn.ModuleList([CNNBlock(dim, use_residual) for _ in range(layers_per_stage)])
        self.pool1 = nn.MaxPool2d(2)
        
        self.stage2 = nn.ModuleList([CNNBlock(dim, use_residual) for _ in range(layers_per_stage)])
        self.pool2 = nn.MaxPool2d(2)
        
        self.stage3 = nn.ModuleList([CNNBlock(dim, use_residual) for _ in range(layers_per_stage)])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, 100)

    def forward(self, x):
        h = self.stem(x)
        for layer in self.stage1:
            h = layer(h)
        h = self.pool1(h)
        for layer in self.stage2:
            h = layer(h)
        h = self.pool2(h)
        for layer in self.stage3:
            h = layer(h)
        h = self.global_pool(h).flatten(1)
        return self.head(h)

class FeatureEscrowCNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, mode):
        super().__init__()
        self.mode = mode
        self.is_global = (mode == "fen_global")
        
        self.stem = nn.Conv2d(3, hidden_dim, 3, padding=1, bias=False)
        
        layers_per_stage = num_layers // 3
        Block = GlobalPeristalticConv2d if self.is_global else PeristalticConv2d
        
        self.stage1 = nn.ModuleList([Block(hidden_dim) for _ in range(layers_per_stage)])
        self.pool1 = nn.MaxPool2d(2)
        
        self.stage2 = nn.ModuleList([Block(hidden_dim) for _ in range(layers_per_stage)])
        self.pool2 = nn.MaxPool2d(2)
        
        self.stage3 = nn.ModuleList([Block(hidden_dim) for _ in range(layers_per_stage)])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        if self.is_global:
            self.head = nn.Linear(hidden_dim * 2, 100)
        else:
            self.head = nn.Linear(hidden_dim * 4, 100)

    def forward(self, x):
        B = x.shape[0]
        pipe = self.stem(x)
        
        if self.is_global:
            E = torch.zeros(B, pipe.size(1), device=x.device)
            for layer in self.stage1:
                pipe, E = layer(pipe, E)
            pipe = self.pool1(pipe)
            for layer in self.stage2:
                pipe, E = layer(pipe, E)
            pipe = self.pool2(pipe)
            for layer in self.stage3:
                pipe, E = layer(pipe, E)
            pipe_pooled = self.global_pool(pipe).flatten(1)
            combined = torch.cat([E, pipe_pooled], dim=-1)
        else:
            E1 = torch.zeros_like(pipe)
            for layer in self.stage1:
                pipe, E1 = layer(pipe, E1)
            vault_32 = E1
            pipe = self.pool1(pipe)
            
            E2 = torch.zeros_like(pipe)
            for layer in self.stage2:
                pipe, E2 = layer(pipe, E2)
            vault_16 = E2
            pipe = self.pool2(pipe)
            
            E3 = torch.zeros_like(pipe)
            for layer in self.stage3:
                pipe, E3 = layer(pipe, E3)
            vault_8 = E3
            
            v32_pooled = self.global_pool(vault_32).flatten(1)
            v16_pooled = self.global_pool(vault_16).flatten(1)
            v8_pooled  = self.global_pool(vault_8).flatten(1)
            pipe_pooled = self.global_pool(pipe).flatten(1)
            combined = torch.cat([v32_pooled, v16_pooled, v8_pooled, pipe_pooled], dim=-1)
            
        return self.head(combined)

def build_model(mode, num_layers, hidden_dim):
    if mode == "plain_cnn": 
        return CNNBaseline(hidden_dim, num_layers, use_residual=False)
    elif mode == "resnet_baseline": 
        return CNNBaseline(hidden_dim, num_layers, use_residual=True)
    elif mode in ["fen_spatial", "fen_global"]: 
        return FeatureEscrowCNN(hidden_dim, num_layers, mode)
    raise ValueError(mode)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def choose_hidden(mode, num_layers, target_params):
    best_h = 16
    best_diff = float('inf')
    for h in range(16, 512, 4):
        model = build_model(mode, num_layers, h)
        params = count_params(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_h = h
            best_diff = diff
    return best_h

def save_history_to_csv(history, filepath="experiments_h100/cifar_scale_history.csv"):
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
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.shape[0]
        loss_sum += loss.item() * bs
        acc_sum += (logits.argmax(-1) == yb).sum().item()
        total += bs
    return {"loss": loss_sum / total, "acc": acc_sum / total}

def train_one(seed, mode, num_layers, target_params, train_loader, val_loader):
    seed_everything(seed)
    h_dim = choose_hidden(mode, num_layers, target_params)
    model = build_model(mode, num_layers, h_dim).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"CIFAR SCALE | MODE={mode.upper()} | LAYERS={num_layers} | Width={h_dim} | Params={params:,}")
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
        val = evaluate(model, val_loader, criterion)
        if val["acc"] > best_acc: 
            best_acc = val["acc"]
        
        print(f"  Ep {epoch:02d} | Val Loss: {val['loss']:.4f} | Val Acc: {val['acc']*100:.2f}% | Best Acc: {best_acc*100:.2f}%")

        history.append({
            "seed": seed,
            "mode": mode,
            "depth": num_layers,
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_acc": val["acc"] * 100
        })

    print(f"  Finished {mode.upper()} Depth={num_layers} | Best Acc: {best_acc*100:.2f}% | Time: {time.time()-start:.1f}s")
    save_history_to_csv(history)
    return best_acc

if __name__ == "__main__":
    PARAM_TARGETS = {
        12: 8000000,
        18: 12000000
    }

    print("\n" + "=" * 80)
    print("CIFAR SCALE PARAMETER MATRIX:")
    for l in DEPTHS:
        target = PARAM_TARGETS[l]
        print(f"  Depth = {l} (Target Params: {target:,}):")
        for mode in MODES:
            h_dim = choose_hidden(mode, l, target)
            model_test = build_model(mode, l, h_dim)
            print(f"    {mode.upper():<16} (Width={h_dim:<3}) : {count_params(model_test):,} parameters")
    print("=" * 80 + "\n")

    train_loader, val_loader = make_loaders(seed=42)
    
    for l in DEPTHS:
        target = PARAM_TARGETS[l]
        print(f"\n" + "#" * 80)
        print(f"STARTING SWEEP FOR DEPTH L={l} | TARGET PARAMS = {target:,}")
        print("#" * 80)
        
        for seed in SEEDS:
            for mode in MODES:
                train_one(seed, mode, l, target, train_loader, val_loader)
