# ==============================================================================
# Pushing FEN (Copy-Only) to SOTA Potential on FordA Time-Series Classification
# 100 Epochs, Non-linear readout, Dropout, and learning rate scheduler
# ==============================================================================

import os
import time
import random
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

# --- 1. CONFIGURATION ---
SEED = 2026
TARGET_PARAMS = 100000
BATCH_SIZE = 64
LR = 2e-3  # Slightly higher starting learning rate with decay scheduler
NUM_EPOCHS = 100  # More epochs to allow deep features to converge
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# --- 2. DATASET DOWNLOADING & PREPROCESSING ---
def download_dataset():
    os.makedirs('data', exist_ok=True)
    train_path = 'data/FordA_TRAIN.tsv'
    test_path = 'data/FordA_TEST.tsv'
    
    train_url = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv'
    test_url = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv'
    
    if not os.path.exists(train_path):
        print("Downloading FordA training set...")
        urllib.request.urlretrieve(train_url, train_path)
        
    if not os.path.exists(test_path):
        print("Downloading FordA testing set...")
        urllib.request.urlretrieve(test_url, test_path)
        
    return train_path, test_path

class FordADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]

def prepare_data():
    train_path, test_path = download_dataset()
    
    train_df = pd.read_csv(train_path, sep='\t', header=None)
    test_df = pd.read_csv(test_path, sep='\t', header=None)
    
    y_train = train_df[0].values
    y_train = np.where(y_train == -1, 0, 1)
    X_train = train_df.iloc[:, 1:].values
    
    y_test = test_df[0].values
    y_test = np.where(y_test == -1, 0, 1)
    X_test = test_df.iloc[:, 1:].values
    
    n = len(X_train)
    indices = np.arange(n)
    np.random.seed(SEED)
    np.random.shuffle(indices)
    
    val_size = int(n * 0.15)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    
    print(f"Dataset Loaded Successfully!")
    print(f"Sequences -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    train_dataset = FordADataset(X_train, y_train)
    val_dataset = FordADataset(X_val, y_val)
    test_dataset = FordADataset(X_test, y_test)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

# --- 3. OPTIMIZED MODEL DEFINITION ---

class OptimizedFeatureEscrowRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.core = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.escrow_proj = nn.Linear(hidden_size, hidden_size)
        
        # High-capacity non-linear readout head with dropout to prevent overfitting
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        E = torch.zeros(B, self.hidden_size, device=x.device)
        
        # Speed Optimization: Pre-project the entire sequence outside the loop
        x_proj_all = self.x_proj(x)
        
        gate_means = []
        for t in range(seq_len):
            xt = x_proj_all[:, t, :]
            z = h + xt
            f_raw = torch.tanh(self.core(z) + z)
            
            g = torch.sigmoid(self.gate(f_raw))
            D = g * f_raw
            
            # Copy-only routing (no subtraction to preserve temporal context flow)
            h = f_raw
            E = E + self.escrow_proj(D)
            
            if return_stats:
                gate_means.append(g.mean().item())
                
        combined = torch.cat([h, E], dim=-1)
        logits = self.fc(combined)
        
        if return_stats:
            stats = {
                "active_norm": h.norm(dim=-1).mean().item(),
                "gate_mean": sum(gate_means)/len(gate_means) if gate_means else 0.5
            }
            return logits, stats
        return logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def choose_hidden_dim(input_size):
    best_h = 8
    best_diff = float('inf')
    for h in range(8, 512):
        model = OptimizedFeatureEscrowRNN(input_size=input_size, hidden_size=h)
        params = count_params(model)
        diff = abs(params - TARGET_PARAMS)
        if diff < best_diff:
            best_h = h
            best_diff = diff
    return best_h

# --- 4. MAIN TRAINING & EVALUATION LOOP ---
if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = prepare_data()
        
        hidden_dim = choose_hidden_dim(input_size=1)
        model = OptimizedFeatureEscrowRNN(input_size=1, hidden_size=hidden_dim).to(DEVICE)
        params_count = count_params(model)
        
        print(f"\n================================================================================")
        print(f"TRAINING OPTIMIZED FEN (COPY) | Hidden={hidden_dim} | Params={params_count:,}")
        print(f"================================================================================")
        
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        # Learning rate scheduler: decays learning rate if validation F1 plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = -1.0
        best_model_path = "best_optimized_fen_copy.pth"
        
        start_time = time.time()
        
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            active_norm_sum = 0.0
            batches_val = 0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    
                    outputs, stats = model(x, return_stats=True)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    
                    preds = outputs.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(y.cpu().numpy())
                    active_norm_sum += stats["active_norm"]
                    batches_val += 1
                    
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_active_norm = active_norm_sum / batches_val if batches_val > 0 else 0.0
            
            val_acc = accuracy_score(val_targets, val_preds) * 100
            val_f1 = f1_score(val_targets, val_preds, average='macro') * 100
            
            # Step the scheduler based on validation F1
            scheduler.step(val_f1)
            
            # Log progress every 5 epochs or on best performance
            if epoch % 5 == 0 or val_f1 > best_val_f1:
                print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | Active Norm: {avg_active_norm:.2f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                
        training_time = time.time() - start_time
        print(f"\nFinished Training | Best Val Macro F1: {best_val_f1:.2f}% | Total Time: {training_time:.1f}s")
        
        # Test Evaluation
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        test_preds, test_targets = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                outputs = model(x)
                preds = outputs.argmax(dim=-1).cpu().numpy()
                test_preds.extend(preds)
                test_targets.extend(y.cpu().numpy())
                
        test_acc = accuracy_score(test_targets, test_preds) * 100
        test_f1 = f1_score(test_targets, test_preds, average='macro') * 100
        
        print("\n" + "#" * 80)
        print("OPTIMIZED FEN (COPY) FINAL EVALUATION")
        print("#" * 80)
        print(f"Test Accuracy:   {test_acc:.2f}%")
        print(f"Test Macro F1:   {test_f1:.2f}%")
        print(f"Total Params:    {params_count:,}")
        print("#" * 80)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
