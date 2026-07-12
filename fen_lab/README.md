# fen_lab — active FEN experiments

Clean-slate, one-question-per-script track.

**Full journey (read this first):** repo root [`README.md`](../README.md) — problem → synthetic foundation (incl. LSTM floors) → ablations → real data.

## Run (Colab)

1. Runtime → **GPU**  
2. Paste an entire `expXX_*.py` file into one cell → Run  
3. Toggle `FAST_MODE` when present  

Deps: `torch`, `numpy` (01–04, 01b); add `pandas` for exp05.

| Exp | File | Role in the journey |
|-----|------|---------------------|
| **01** | `exp01_baseline_dual_task.py` | FEN family on recall5 + distracted |
| **01b** | `exp01b_lstm_baseline.py` | **LSTM** + residual / bag / slot on the same foundation probes |
| **02** | `exp02_ode_fen_order_ablation.py` | Soft-tape order ablations |
| **03** | `exp03_write_vs_readout.py` | Write × readout; hybrid both tasks |
| **04** | `exp04_mid_deliver.py` | Mid-read vs reinject |
| **05** | `exp05_real_data.py` | MIT-BIH |
| **05b** | `exp05_forda.py` | FordA |

## Not here

Archived early work: [`../history/`](../history/). Not part of this path’s evidence base.
