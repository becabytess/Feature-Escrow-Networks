# fen_lab

Active FEN experiments. Full write-up: [`../README.md`](../README.md).

## Run

1. Colab / Kaggle → GPU  
2. Paste an entire `expXX_*.py` file → Run  
3. Toggle `FAST_MODE` when present  

Deps: `torch`, `numpy`; `pandas` for some loaders (CIFAR/MNIST CSV).

| Exp | File |
|-----|------|
| 01 | `exp01_baseline_dual_task.py` |
| 01b | `exp01b_lstm_baseline.py` |
| 02 | `exp02_ode_fen_order_ablation.py` |
| 03 | `exp03_write_vs_readout.py` |
| 04 | `exp04_mid_deliver.py` |
| 05 | `exp05_real_data.py` |
| 05b | `exp05_forda.py` |
| 06 | `exp06_cifar100.py` |
| 07 | `exp07_multipass_read.py` |
| 08 | `exp08_shared_board.py` |
| 09 | `exp09_smnist.py` |
| 09b | `exp09b_lstm_smnist_sweep.py` |
| 10 | `exp10_pmnist.py` | pMNIST (perm) — ordered escrow vs local-raster hypothesis |

Archived early work: [`../history/`](../history/).
