# fen_lab — active FEN experiments

Clean-slate, one-question-per-script track for Feature-Escrow Networks.

**Full journey, proofs, and next steps:** see the repo root [`README.md`](../README.md).

## Run (Colab)

1. Runtime → **GPU**  
2. Paste an entire `expXX_*.py` file into one cell → Run  
3. Toggle `FAST_MODE` at the top when present  

Deps: `torch`, `numpy` (exp01–04); add `pandas` for exp05 (auto-downloads data).

| Exp | File | Question |
|-----|------|----------|
| **01** | `exp01_baseline_dual_task.py` | Dual-task: bag vs roll vs slot vs residual |
| **02** | `exp02_ode_fen_order_ablation.py` | Soft-tape order: which hypothesis fixes recall? |
| **03** | `exp03_write_vs_readout.py` | Write × readout grid; hybrid both tasks |
| **04** | `exp04_mid_deliver.py` | Mid-read of archive vs every-step reinject |
| **05** | `exp05_real_data.py` | Frozen FEN on MIT-BIH (+ optional FordA) |
| **05b** | `exp05_forda.py` | FordA only (longer budget / new seed) |

## Not here

Archived early explorations: [`../history/`](../history/). They are **not** part of this path’s evidence base.
