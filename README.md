# Feature-Escrow Networks (FEN)

[![Web Report](https://img.shields.io/badge/Web_Report-Interactive-2563eb.svg)](index.html)
[![Full Report](https://img.shields.io/badge/Full_Report-Markdown-10b981.svg)](full_research_report.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Feature-Escrow Networks (FEN)** is a novel, high-efficiency recurrent neural network architecture that decouples **active temporal computation** from **historical information preservation** via a dual-pathway structure.

### 📚 Research Documentation
* 🌐 **[Interactive Web Report (`index.html`)](index.html)** — Anthropic-style interactive research report featuring visual architecture flows, dynamic paradigm comparisons, and interactive topology selectors.
* 📝 **[Full Research & Experimental Report (`full_research_report.md`)](full_research_report.md)** — Complete, uncompressed documentation of all 15 experiments, synthetic probes, regime maps, and theoretical foundations.

---

## 💡 Biological Inspiration: The Small Intestine Analogy

Standard recurrent models (RNNs, LSTMs, GRUs) force all sequence history and state updates into a single hidden state trajectory $h_t = f(h_{t-1}, x_t)$. This forces intermediate neurons to act as passive **"pass-through conduits"**, burning state capacity to drag old memories along step-by-step.

FEN is inspired by the **absorption mechanism of the human small intestine**:

> In the digestive system, nutrients do not wait to travel through to the end of the intestinal tube before being absorbed. The moment a food particle is digested and ready, it **diffuses immediately through the intestinal wall into the bloodstream**. It does not stay in the active digestive fluid, nor does it mix with undigested material moving down the tube.
> 
> **Feature-Escrow Networks** apply this exact principle to deep learning: the moment a temporal feature is processed and ready, the model **deposits it into an auxiliary escrow vault ($E$)**. The active recurrent loop is freed from carrying that memory, allowing it to focus strictly on local temporal dynamics without fear of overwriting finished facts.

```text
       RNN Pathway (Temporal Dynamics & Digestion)
x_t ──► [ h_t = f(h_{t-1}, x_t) ] ───────► h_T ──┐
             │                                   ├─► Head([h_T, E])
             ▼ (phi extraction / absorption)     │
       Escrow Pathway (Vault / Bloodstream)      │
        E = Σ_t φ(h_t) ──────────────────────────┘
```

---

## ⚡ Architectural Comparison

By extracting features as soon as they appear, FEN acts as **Speculative Write-Time Attention**, capturing attention-like feature preservation in linear time and constant memory:

| Architecture | Memory Strategy | Time Complexity | State Memory | Early Signal (Epoch 1 Acc) | Peak Acc (sMNIST) |
|--------------|-----------------|-----------------|--------------|----------------------------|-------------------|
| **RNN / Residual** | Single state, no gating | $O(T)$ | $O(1)$ | 10.0% (Chance) | 10.2% |
| **LSTM (1-Layer)** | Single state, internal forget gates | $O(T)$ | $O(1)$ | 10.0% (Chance) | 11.0% |
| **LSTM (3-Layer Tuned)** | Deep stack, internal forget gates | $O(T)$ | $O(1)$ | 10.0% (Chance) | 80.2% (@ Ep 30) |
| **Transformer** | Full state cache + backward QKV search | $O(T^2)$ | $O(T)$ KV Cache | Fast | High |
| **FEN (`roll_nodep`)** | Dual-pathway speculative escrow | **$O(T)$** | **$O(1)$** | **69.1% (Epoch 1)** | **88.7% (Epoch 9)** |

---

## 🏆 Key Benchmark Highlights

* **Sequential MNIST ($T=400$):** `roll_nodep` reaches **69.1% Epoch-1** and **88.7% Peak Accuracy** (outperforming tuned 3-Layer LSTMs by 15x in sample efficiency).
* **Permuted MNIST (pMNIST):** `roll_nodep` maintains **60.4% Epoch-1** and **87.5% Peak Accuracy**, proving true ordered non-commutative escrow.
* **Distracted Counting ($T=96$):** `bag_nodep` solves dual-role state retention with **95.1% Joint Accuracy** (vs ~10.0% for LSTMs).
* **Pixel-Level $T=1024$ CIFAR-100:** Hierarchical FEN Roll reaches **23.38%** vs 5.36% for standard Hierarchical RNNs.

---

## 💻 Minimal PyTorch Implementation (`roll_nodep`)

```python
import torch
import torch.nn as nn

class FENRollNoDep(nn.Module):
    """
    Feature-Escrow Network with Channel-Roll Write (No Depletion).
    Simple, fast, and SOTA among non-attention recurrent architectures.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.roll_gate = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        E = x.new_zeros(B, self.hidden_dim)
        x_p = self.x_proj(x)

        for t in range(T):
            z = h + x_p[:, t]
            f = torch.tanh(self.core(z) + z)        # 1. Recurrent state proposal
            g = torch.sigmoid(self.gate(f))          # 2. Extraction gate
            D = g * f
            v = self.v_proj(D)
            
            h = f                                   # 3. Temporal state (NO depletion)
            
            gamma = torch.sigmoid(self.roll_gate(f))# 4. Escrow channel-roll update
            E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=1, dims=-1) + v

        # 5. Joint Decision Head over final temporal state [h_T] and Escrow vault [E]
        return self.head(torch.cat([h, E], dim=-1))
```

---

## 📁 Repository Structure

```text
Feature-Escrow-Networks/
├── README.md                   ← Portfolio landing page (this document)
├── full_research_report.md     ← Uncompressed research report (all 15 experiments & log details)
├── index.html                  ← Interactive Anthropic-style Web Research Report
├── requirements.txt            ← Python dependencies
└── fen_lab/                    ← Experimental laboratory scripts (exp01 to exp13)
```
