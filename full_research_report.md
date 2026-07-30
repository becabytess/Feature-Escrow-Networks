# Feature-Escrow Networks (FEN): Full Research & Experimental Report

This document contains the complete, uncompressed research record, theoretical foundations, biological inspirations, and empirical benchmark logs for **Feature-Escrow Networks (FEN)**.

---

## 1. Biological Intuition & The Origin of Feature-Escrow

### The Small Intestine Metaphor ("Processed Features Diffuse Immediately")

In standard sequential and deep neural network architectures, intermediate features extracted at step $t$ or layer $l$ that are required for a final decision at step $T$ must be carried forward step-by-step through the active recurrent state:

$$h_t = f(h_{t-1}, x_t)$$

This forces intermediate hidden units to act as passive **"pass-through conduits"**, burning state capacity and risking feature overwriting or degradation across temporal steps.

The original biological inspiration for **Feature-Escrow Networks** comes from the **human small intestine & digestive system**:

> **The Small Intestine Analogy:**
> In the digestive tract, nutrients do not wait to travel all the way through to the end of the intestinal tube before being absorbed. The moment a food particle is digested and broken down into useful nutrients, it **diffuses immediately through the intestinal wall into the bloodstream**. It does not stay in the active digestive fluid, nor does it mix with undigested material moving down the tube.
>
> Standard neural networks treat memory like a closed pipe where everything must be dragged along until the end. **Feature-Escrow Networks** treat feature extraction like intestinal absorption: the moment a temporal feature is processed and ready, the model **deposits it into an auxiliary escrow vault ($E$)**. The active recurrent loop is freed from carrying that memory, allowing it to focus strictly on local temporal dynamics without fear of overwriting finished facts.

### Dual-Pathway Formulation

FEN formalizes this intuition by establishing a **dual-pathway architecture**:

```text
       RNN Pathway (Temporal Dynamics & Digestion)
x_t ──► [ h_t = f(h_{t-1}, x_t) ] ───────► h_T ──┐
             │                                   ├─► Head([h_T, E])
             ▼ (phi extraction / absorption)     │
       Escrow Pathway (Vault / Bloodstream)      │
        E = Σ_t φ(h_t) ──────────────────────────┘
```

1. **Temporal Processing Pathway (Active Loop):** The recurrent network operates unconstrained to compute step-by-step temporal state transformations $h_t = f(h_{t-1}, x_t)$.
2. **Escrow Accumulation Pathway (Auxiliary Vault):** An extraction transformation $\phi(h_t)$ absorbs ready features as they appear and deposits them into the escrow vault $E = \sum_{t=1}^T \phi(h_t)$.
3. **Joint Decision Head:** The final prediction evaluates both active state and escrow vault: $\hat{y} = \text{Head}([h_T, E])$.

---

## 2. Complementary Lens: Speculative Write-Time Attention

Alongside the biological absorption perspective, FEN provides a clean mathematical bridge between Recurrent Networks and Transformers:

| Paradigm | Core Strategy | Mechanism / Analogy | Computational Cost |
|----------|---------------|---------------------|-------------------|
| **RNN / LSTM** | Carry memories forward through recurrence | *"Carry important memories step-by-step until needed."* (Gating inside single active loop) | $O(T)$ time, $O(1)$ memory |
| **Attention (Transformer)** | Look backward dynamically | *"Look back later at decision time and retrieve past states."* (QKV dot-product search) | $O(T^2)$ time/space |
| **FEN (Feature Escrow)** | Speculative forward accumulation | *"Extract features the moment they are ready, and deposit them into escrow."* (Write-time absorption into $E$) | **$O(T)$ time, $O(1)$ memory** |

---

## 3. Metrics that Matter

Results are reported with two complementary views:

| Metric | What it shows |
|--------|----------------|
| **Peak / best accuracy** | Whether the model can solve the task under the budget |
| **Early accuracy (epoch 1–2)** | How directly useful signal reaches parameters: gradient flow, stability, sample efficiency |

---

## 4. Architectural Switches & Topologies

| Switch | Options | Role |
|--------|---------|------|
| **Write Mode** | **Bag** | Commutative summation $\sum \phi(h_t)$ for static facts & dual-role tasks |
| | **Slots / Hard Tape** | Addressable ordered cells for exact ordered token recovery |
| | **Channel-Roll** | Non-commutative circular shift vault for long ordered scans (sMNIST, pMNIST, audio) |
| | **Hybrid** | Concurrent Bag + Roll vaults for pure 2D raster image peak accuracy |
| **Depletion** | **`roll_nodep` / `fen_copy`** | **Deplete OFF (Universal Default)**: Pure accumulation. Lets RNN operate unconstrained. |
| | `fen_bag` / `*_dep` | Deplete ON ($h_t \leftarrow h_t - D_t$): Optional pipe hygiene switch. |

---

## 5. Synthetic Foundation Probes ($T=96$, ~15k params)

### Distracted Counting (Dual-Role State Test)
Static **ID** at $t=0$, then noisy **count** events. Label = ID × count bin.

| Model | Joint Peak Acc | ID Accuracy | Count Accuracy | Status |
|-------|---------------:|------------:|---------------:|--------|
| Residual RNN | 9.0% | 11.0% | 95.0% | State Overwrite Failure |
| LSTM Baseline | 10.0% | 10.0% | 95.0% | State Overwrite Failure |
| FEN Slot (Hard Tape) | 19.0% | 19.0% | 100.0% | Wrong Topology |
| FEN Bag (+ Deplete) | 77.5% | 78.1% | 97.5% | Solved |
| **FEN Bag (No Deplete / `fen_copy`)** | **95.1%** | **97.5%** | **97.2%** | **Optimal** |

### Ordered Recall (Recall5: Exact Token Recovery)
Recover 5 symbols in exact order. Primary metric: **exact** full-sequence accuracy.

| Model | Exact Acc | Token Acc | Status |
|-------|----------:|----------:|--------|
| Residual RNN | 0.0% | ~10.0% | Floor Failure |
| LSTM Baseline | 0.0% | ~10.0% | Floor Failure |
| FEN Bag / Roll (Pooled) | ~0.0% | ~35.0% | Cannot Address Cells |
| **FEN Slot (Hard Tape)** | **96.0% – 100.0%** | **~100.0%** | **Solved** |

---

## 6. Hard-Bench: Sequential MNIST (sMNIST, $T=400$, 100k params)

| Model | best acc | ep1 | ep2 | pipe norm |
|-------|---------:|----:|----:|----------:|
| Residual RNN | 0.102 | 0.10 | 0.10 | 15.7 |
| LSTM (1-Layer Baseline) | 0.110 | 0.10 | 0.10 | low |
| FEN 2-Pass Cold | 0.465 | 0.15 | 0.29 | ~9 |
| FEN Bag | 0.661 | 0.24 | 0.36 | ~9 |
| FEN Hard Bag | 0.719 | 0.28 | 0.34 | ~5 |
| FEN Copy (Bag, No Deplete) | 0.776 | 0.35 | 0.49 | ~11 |
| FEN Reinject | 0.823 | 0.23 | 0.39 | ~18 |
| FEN Roll (+ Deplete) | 0.881 | 0.64 | 0.80 | ~10 |
| **FEN Roll (No Deplete / `roll_nodep`)** | **0.887** | **0.691** | **0.815** | ~11.5 |
| FEN Hybrid (Bag + Roll) | **0.906** | 0.67 | 0.71 | ~10 |

### Dedicated 30-Epoch LSTM Deep Sweep (`exp08b`)

| Variant | best acc | ep1 | ep2 | Epoch to 50% |
|---------|---------:|----:|----:|-------------:|
| LSTM 1L Baseline | 0.477 | 0.10 | 0.10 | Never |
| LSTM 2L Standard | 0.719 | 0.10 | 0.12 | Ep 16 |
| LSTM 2L Wide | 0.720 | 0.10 | 0.26 | Ep 23 |
| LSTM 2L + Dropout | 0.774 | 0.10 | 0.15 | Ep 13 |
| **LSTM 3L Best-Tuned** | **0.802** | **0.10** | **0.23** | **Ep 15** |

*Takeaway:* FEN Roll reaches **80.0% accuracy at Epoch 2**, whereas the best 3-Layer LSTM requires **30 full epochs** just to hit 80.2% (15x speedup in sample efficiency).

---

## 7. Permuted MNIST (pMNIST: Ordered Escrow vs Spatial Locality)

Fixed random permutation of $T=400$ pixel axis (`PERM_SEED=123`). Spatial neighborhoods destroyed.

| Model | pMNIST Peak | pMNIST Ep 1 | pMNIST Ep 2 | sMNIST Peak |
|-------|------------:|------------:|------------:|------------:|
| Residual RNN | 0.616 | 0.403 | 0.547 | 0.102 |
| FEN Bag | 0.402 | 0.194 | 0.231 | 0.661 |
| FEN Copy | 0.589 | 0.218 | 0.330 | 0.776 |
| **FEN Roll (No Deplete)** | **0.875** | **0.604** | **0.671** | **0.887** |
| FEN Hybrid | 0.840 | 0.327 | 0.514 | 0.906 |
| LSTM 1-Layer | 0.799 | 0.218 | 0.488 | 0.110 |

---

## 8. Sequential CIFAR-100 & Hierarchical FEN (`exp10`, `exp11`, `exp13`)

### Patch Regime Map ($T=16 \to 64 \to 256$, ~100k params)

| Regime | Sequence Length | FEN Roll | FEN Hybrid | FEN Bag | LSTM 1L | Residual RNN |
|--------|----------------|---------:|-----------:|--------:|--------:|-------------:|
| **P8 (Low Stress)** | $T=16, C=192$ | **19.9%** (Ep1: 8.5%) | 19.9% (Ep1: 8.4%) | 18.1% (Ep1: 7.5%) | 16.9% | 13.2% |
| **P4 (Mid Stress)** | $T=64, C=48$ | 21.8% (Ep1: **10.5%**) | **21.9%** (Ep1: 8.5%) | 19.7% (Ep1: 5.5%) | 17.7% | 7.6% |
| **P2 (High Stress)** | $T=256, C=12$ | **14.9%** (Ep1: **7.5%**) | **14.9%** (Ep1: 3.7%) | 3.1% (Floor) | 10.4% | 5.8% |

### Pixel-Level $T=1024$ Hierarchical FEN (`exp13`)
Dividing $T=1024$ pixel stream into $K=32$ chunks of length 32:

| Model | Best (15 ep) | Best (40 ep) | Epoch 1 Acc | Params |
|-------|-------------:|-------------:|------------:|-------:|
| Standard Hierarchical RNN | 5.36% | 5.36% | 2.18% | 99,746 |
| Standard Hierarchical Residual RNN | 5.36% | 5.36% | 2.73% | 99,746 |
| **Hierarchical FEN Roll (Single-Pass)** | 21.88% | **23.38% (Ep 17)** | **10.38%** | 100,339 |
| Hierarchical FEN Sandwich (Double-Pass) | **23.66%** | **23.94% (Ep 21)** | 8.71% | 99,166 |

---

## 9. Deplete Law & Real Waveforms (`exp05`, `exp05b`, `exp12`, `exp12b`)

- **Deplete Law:** Subtraction $h_t \leftarrow h_t - D_t$ trims pipe norm (hygiene), but is **not required for peak or early accuracy**. `roll_nodep` and `fen_copy` (pure accumulation) are the simplest and most effective.
- **MIT-BIH ECG:** Residual collapses to majority class; FEN Roll & FEN Bag achieve high classification (~92%+).
- **FordA Acoustic Noise:** Residual near chance; FEN Roll dominates performance over raw sensor waveforms.

---

## 10. Complete Experiment Inventory (`fen_lab/`)

| Script | Title / Role |
|--------|--------------|
| `exp01_baseline_dual_task.py` | FEN family on synthetic foundation probes |
| `exp01b_lstm_baseline.py` | LSTM & Residual baselines on foundation probes |
| `exp02_ode_fen_order_ablation.py` | Soft-tape order & cell-aligned readout ablations |
| `exp03_write_vs_readout.py` | Write topology x readout mechanism cross-grid |
| `exp04_mid_deliver.py` | Decision-time read vs continuous reinjection |
| `exp05_real_data.py` | MIT-BIH ECG 1D waveform classification |
| `exp05_forda.py` | FordA acoustic sensor noise classification |
| `exp06_multipass_read.py` | Multi-pass discrete readout evaluation |
| `exp07_shared_board.py` | Dual experts with shared escrow blackboard |
| `exp08_smnist.py` | sMNIST sequential digit hard benchmark |
| `exp08b_lstm_smnist_sweep.py` | 30-epoch LSTM hyperparameter sweep |
| `exp09_pmnist.py` | pMNIST permuted stream: locality vs ordered escrow probe |
| `exp10_cifar100.py` | Sequential CIFAR-100 patch benchmark (P4 & P2) |
| `exp11_stress_curve.py` | CIFAR regime map (P8 -> P4 -> P2) |
| `exp12_deplete_law.py` | Deplete ON/OFF grid ablation on Distracted Counting |
| `exp12b_roll_nodep_smnist.py` | sMNIST Channel-Roll without depletion (`roll_nodep`) |
| `exp13_hierarchical_cifar.py` | Hierarchical FEN on pixel-level $T=1024$ CIFAR-100 |
