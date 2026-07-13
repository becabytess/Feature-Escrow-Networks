# fen_lab

Active FEN experiments. Full write-up and all result tables: [../README.md](../README.md).

## Run

1. Colab / Kaggle → GPU  
2. Paste an entire xpXX_*.py file → Run  
3. Toggle FAST_MODE / INPUT_MODE / PATCH_SIZE when present  

Deps: 	orch, 
umpy; pandas for some loaders (CIFAR/MNIST CSV).

| Exp | File | Notes |
|-----|------|--------|
| 01 | xp01_baseline_dual_task.py | Foundation probes (FEN family) |
| 01b | xp01b_lstm_baseline.py | LSTM + residual on foundation |
| 02 | xp02_ode_fen_order_ablation.py | Soft-tape order ablations |
| 03 | xp03_write_vs_readout.py | Write × readout grid |
| 04 | xp04_mid_deliver.py | Mid-read vs reinject |
| 05 | xp05_real_data.py | MIT-BIH |
| 05b | xp05_forda.py | FordA |
| 06 | xp06_multipass_read.py | Multi-pass discrete read |
| 07 | xp07_shared_board.py | Dual experts + shared board |
| 08 | xp08_smnist.py | sMNIST hard-bench |
| 08b | xp08b_lstm_smnist_sweep.py | LSTM honesty sweep on sMNIST |
| 09 | xp09_pmnist.py | pMNIST (ordered escrow vs local raster) |
| 10 | xp10_cifar100.py | Sequential CIFAR-100; set PATCH_SIZE to 4 or 2 |

Archived early work: [../history/](../history/) — not part of this track’s evidence base.
