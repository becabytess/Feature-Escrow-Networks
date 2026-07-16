# fen_lab

Active FEN experiments. Full write-up and all result tables: [`../README.md`](../README.md).

## Run

1. Colab / Kaggle → GPU  
2. Paste an entire `expXX_*.py` file → Run  
3. Toggle `FAST_MODE` / `TASKS` / `PATCH_SIZES` when present  

Deps: `torch`, `numpy`; `pandas` for some loaders (CIFAR/MNIST CSV).

| Exp | File | Notes |
|-----|------|--------|
| 01 | `exp01_baseline_dual_task.py` | Foundation probes (FEN family) |
| 01b | `exp01b_lstm_baseline.py` | LSTM + residual on foundation |
| 02 | `exp02_ode_fen_order_ablation.py` | Soft-tape order ablations |
| 03 | `exp03_write_vs_readout.py` | Write × readout grid |
| 04 | `exp04_mid_deliver.py` | Mid-read vs reinject |
| 05 | `exp05_real_data.py` | MIT-BIH |
| 05b | `exp05_forda.py` | FordA |
| 06 | `exp06_multipass_read.py` | Multi-pass discrete read |
| 07 | `exp07_shared_board.py` | Dual experts + shared board |
| 08 | `exp08_smnist.py` | sMNIST hard-bench |
| 08b | `exp08b_lstm_smnist_sweep.py` | LSTM honesty sweep on sMNIST |
| 09 | `exp09_pmnist.py` | pMNIST (ordered escrow vs local raster) |
| 10 | `exp10_cifar100.py` | Sequential CIFAR-100; set `PATCH_SIZE` to 4 or 2 |
| 11 | `exp11_stress_curve.py` | Regime map: CIFAR patches 8→4→2 |
| 12 | `exp12_deplete_law.py` | bag/roll × deplete (distracted; optional sMNIST) |
| 12b | `exp12b_roll_nodep_smnist.py` | sMNIST roll **without** deplete (early signal) |
| 13 | `exp13_hierarchical_cifar.py` | Hierarchical vanilla RNNs vs Hierarchical FEN (roll & sandwich) on T=1024 pixel CIFAR-100 |

