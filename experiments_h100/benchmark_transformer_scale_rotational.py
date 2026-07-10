# ==============================================================================
# 12-Layer Transformer Scale Sweep with Rotational FEN on WikiText-2
# Target Params = 30M. Sweeps learning rates: [5e-4, 1e-3]
# ==============================================================================

import os
import time
import random
import csv
import math
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURATION ---
MODES = ["gpt_baseline", "fet_attn_only", "fet_attn_rotational", "fet_ffn_only", "fet_ffn_rotational"]
LRS = [5e-4, 1e-3]
SEEDS = [100]
EPOCHS = 15
BATCH_SIZE = 64
SEQ_LEN = 128
WEIGHT_DECAY = 1e-4
NUM_HEADS = 8
NUM_LAYERS = 12
VOCAB_LIMIT = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | 12-Layer Rotational Sweep | Target Params: 30M")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# --- 2. DATASET DOWNLOAD & LOAD ---
def download_wikitext2():
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    valid_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    
    os.makedirs(".data", exist_ok=True)
    train_path = ".data/train.txt"
    valid_path = ".data/valid.txt"
    
    if not os.path.exists(train_path):
        urllib.request.urlretrieve(train_url, train_path)
    if not os.path.exists(valid_path):
        urllib.request.urlretrieve(valid_url, valid_path)
    return train_path, valid_path

class WikiTextDataset(Dataset):
    def __init__(self, file_path, vocab=None, seq_len=128):
        self.seq_len = seq_len
        with open(file_path, "r", encoding="utf-8") as f:
            words = f.read().lower().replace("\n", " <eos> ").split()
            
        if vocab is None:
            from collections import Counter
            counts = Counter(words)
            for spec in ["<pad>", "<unk>", "<eos>"]:
                if spec in counts:
                    del counts[spec]
            most_common = counts.most_common(VOCAB_LIMIT - 3)
            self.vocab = {word: idx + 3 for idx, (word, _) in enumerate(most_common)}
            self.vocab["<pad>"] = 0
            self.vocab["<unk>"] = 1
            self.vocab["<eos>"] = 2
        else:
            self.vocab = vocab
            
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.tokens = [self.vocab.get(word, self.vocab["<unk>"]) for word in words]
        self.tokens = torch.tensor(self.tokens, dtype=torch.int64)
        
        self.num_seqs = (len(self.tokens) - 1) // self.seq_len
        self.inputs = self.tokens[:self.num_seqs * self.seq_len].view(self.num_seqs, self.seq_len)
        self.targets = self.tokens[1:self.num_seqs * self.seq_len + 1].view(self.num_seqs, self.seq_len)

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# --- 3. MODEL BLOCKS ---
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class TransformerFET(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, ff_dim, num_layers, mode):
        super().__init__()
        self.mode = mode
        self.num_layers = num_layers
        
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(SEQ_LEN, dim)
        
        self.attn_layers = nn.ModuleList([CausalSelfAttention(dim, num_heads) for _ in range(num_layers)])
        self.ff_layers = nn.ModuleList([PositionwiseFeedForward(dim, ff_dim) for _ in range(num_layers)])
        
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        
        self.has_escrow = mode in ["fet_attn_only", "fet_attn_rotational", "fet_ffn_only", "fet_ffn_rotational"]
        
        if mode == "fet_attn_only":
            self.gate_attn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
            self.escrow_proj_attn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
            self.head = nn.Linear(dim * 2, vocab_size)
        elif mode == "fet_attn_rotational":
            self.gate_attn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
            self.roll_gate_attn = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_layers)])
            self.head = nn.Linear(dim * 2, vocab_size)
        elif mode == "fet_ffn_only":
            self.gate_ff = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
            self.escrow_proj_ff = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
            self.head = nn.Linear(dim * 2, vocab_size)
        elif mode == "fet_ffn_rotational":
            self.gate_ff = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
            self.roll_gate_ff = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_layers)])
            self.head = nn.Linear(dim * 2, vocab_size)
        else:
            self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        device = x.device
        
        h = self.token_embed(x)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embed(positions)
        
        mask = torch.tril(torch.ones(T, T, device=device))
        
        if self.has_escrow:
            E = torch.zeros_like(h)
            
        for i in range(self.num_layers):
            # --- 1. Attention Block ---
            h_norm = self.norm1[i](h)
            a = self.attn_layers[i](h_norm, mask=mask)
            
            if self.mode == "fet_attn_only":
                f_raw = h + a
                g = torch.sigmoid(self.gate_attn[i](f_raw))
                D = g * f_raw
                h = f_raw - D
                E = E + self.escrow_proj_attn[i](D)
            elif self.mode == "fet_attn_rotational":
                f_raw = h + a
                g = torch.sigmoid(self.gate_attn[i](f_raw))
                D = g * f_raw
                h = f_raw - D
                gamma = torch.sigmoid(self.roll_gate_attn[i](f_raw))
                E = (1 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + D
            else:
                h = h + a
                
            # --- 2. MLP Block ---
            h_norm = self.norm2[i](h)
            m = self.ff_layers[i](h_norm)
            
            if self.mode == "fet_ffn_only":
                f_raw = h + m
                g = torch.sigmoid(self.gate_ff[i](f_raw))
                D = g * f_raw
                h = f_raw - D
                E = E + self.escrow_proj_ff[i](D)
            elif self.mode == "fet_ffn_rotational":
                f_raw = h + m
                g = torch.sigmoid(self.gate_ff[i](f_raw))
                D = g * f_raw
                h = f_raw - D
                gamma = torch.sigmoid(self.roll_gate_ff[i](f_raw))
                E = (1 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + D
            else:
                h = h + m
                
        if self.has_escrow:
            combined = torch.cat([h, E], dim=-1)
            logits = self.head(combined)
        else:
            logits = self.head(h)
            
        return logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def choose_hidden(mode, target_params, vocab_size):
    best_h = 16
    best_diff = float('inf')
    for h in range(64, 512, 8):
        model = TransformerFET(vocab_size, h, NUM_HEADS, h * 2, NUM_LAYERS, mode)
        params = count_params(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_h = h
            best_diff = diff
    return best_h

TARGET_PARAMS = 20200000

def save_history_to_csv(history, filepath="experiments_h100/transformer_scale_rotational_history.csv"):
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
    total_loss, total_tokens = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        total_loss += loss.item() * yb.numel()
        total_tokens += yb.numel()
    val_loss = total_loss / total_tokens
    val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
    return {"loss": val_loss, "ppl": val_ppl}

def train_one(seed, mode, lr, vocab_size, train_loader, val_loader):
    seed_everything(seed)
    h_dim = choose_hidden(mode, TARGET_PARAMS, vocab_size)
    model = TransformerFET(vocab_size, h_dim, NUM_HEADS, h_dim * 2, NUM_LAYERS, mode).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"TRANSFORMER ROTATIONAL SCALE | MODE={mode.upper()} | LR={lr} | Width={h_dim} | Params={params:,}")
    print("=" * 80)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_ppl = float('inf')
    start = time.time()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            opt.step()
            
        val = evaluate(model, val_loader, criterion)
        if val["ppl"] < best_ppl:
            best_ppl = val["ppl"]
            
        print(f"  Ep {epoch:02d} | Val Loss: {val['loss']:.4f} | Val PPL: {val['ppl']:.2f} | Best PPL: {best_ppl:.2f}")

        history.append({
            "seed": seed,
            "mode": mode,
            "lr": lr,
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_ppl": val["ppl"]
        })

    print(f"  Finished {mode.upper()} LR={lr} | Best Val PPL: {best_ppl:.2f} | Time: {time.time()-start:.1f}s")
    save_history_to_csv(history)
    return best_ppl

if __name__ == "__main__":
    train_txt, valid_txt = download_wikitext2()
    train_dataset = WikiTextDataset(train_txt, seq_len=SEQ_LEN)
    vocab = train_dataset.vocab
    val_dataset = WikiTextDataset(valid_txt, vocab=vocab, seq_len=SEQ_LEN)
    
    vocab_size = len(vocab)
    print(f"Dataset Tokenized. Vocab Size: {vocab_size} | Train seqs: {len(train_dataset)} | Val seqs: {len(val_dataset)}")
    
    print("\n" + "=" * 80)
    print("PARAMETER COUNT SUMMARY (Target: 30M):")
    for mode in MODES:
        h_dim = choose_hidden(mode, TARGET_PARAMS, vocab_size)
        model_test = TransformerFET(vocab_size, h_dim, NUM_HEADS, h_dim * 2, NUM_LAYERS, mode)
        print(f"  {mode.upper():<24} (Width={h_dim:<3}) : {count_params(model_test):,} parameters")
    print("=" * 80 + "\n")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    for lr in LRS:
        print(f"\n" + "#" * 80)
        print(f"STARTING SWEEP FOR LR={lr} | 12-LAYERS | TARGET PARAMS = {TARGET_PARAMS:,}")
        print("#" * 80)
        for seed in SEEDS:
            for mode in MODES:
                train_one(seed, mode, lr, vocab_size, train_loader, val_loader)
