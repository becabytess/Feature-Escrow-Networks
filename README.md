# Feature-Escrow Networks (FEN)

Clean-slate experimental track: [`fen_lab/`](fen_lab/).  
Earlier explorations are archived under [`history/`](history/) and are not part of this path’s evidence base.

```text
Feature-Escrow-Networks/
  README.md           ← this document
  fen_lab/            ← experiments (Colab / Kaggle one-cells)
  history/            ← archived early work
  requirements.txt
  LICENSE
```

---

## Metrics that matter

Results are reported with **two** complementary views:

| Metric | What it shows |
|--------|----------------|
| **Peak / best accuracy** | Whether the model can solve the task under the budget |
| **Early accuracy (epoch 1–2, and climb)** | How directly useful signal reaches parameters — gradient flow, stability, sample efficiency |

A model that is weak early and strong late is not “equal” to one that is strong early, even if final numbers are close. Early accuracy is treated as evidence of **how well the architecture propagates learning signal**, not as a minor training detail. On harder or longer data, poor early dynamics usually get worse, not better.

**Floor vs lag:** on foundation probes, failures are often **chance / exact ≈ 0**, not a few points behind. Those are capability gaps.

---

## 1. The problem

Sequential models often force **one** hidden state to do two jobs:

1. **Active computation** — update on the current step, count, react to noise  
2. **Long-lived memory** — keep facts that must survive many steps of that activity  

When both live in the same tensor, activity overwrites memory. Residuals help gradients but can **bloat** the stream over long sequences (high norm, context drift, dual-role collapse).

**Claim.** Working memory and archive should not be the same place. Resolved features should leave the active path and be stored outside it.

---

## 2. Biological inspiration

Metaphor: digestion in the **small intestine** (not biophysics).

| Biology | FEN |
|---------|-----|
| Lumen (active tract) | **Pipe** \(h\) |
| Ready to absorb? | **Gate** \(g\) |
| Nutrients to bloodstream | **Escrow write** \(E\) |
| Mass removed from tract | **Deplete** \(h \leftarrow f - D\) |
| Later use of nutrients | **Read** archive at query / final time |

Removal matters, not only copy: a vault that never frees the pipe still leaves the active stream crowded.

```text
h  →  transform  →  f
                 →  gate  →  D = g ⊙ f
                 →  E ← write(E, D)
                 →  h ← f − D
                 →  later: head([h, E])
```

---

## 3. Architecture

One skeleton; variants change **write** and **when/how** \(E\) is read.

| Step | Always |
|------|--------|
| 1 | Propose on the pipe |
| 2 | Gate → commit \(D\) |
| 3 | Write \(D\) into external archive \(E\) |
| 4 | Deplete (when enabled): \(h \leftarrow f - D\) |
| 5 | Deliver via \(\mathrm{head}([h, \mathrm{arch}])\) at final / query time — not continuous dump of \(E\) into \(h\) |

| Write | Role |
|-------|------|
| **Bag** | Commutative “set of facts”; dual-role / static context |
| **Hard pointer / slots** | Ordered cells; exact sequence outputs |
| **Channel-roll** | Non-commutative vault; ordered **scans** (pixels, sensors) |
| **Hybrid (bag + roll)** | Two vaults; strong on long ordered classification |

| Control | Role |
|---------|------|
| Residual RNN | No escrow |
| LSTM | Classical recurrent baseline |

---

## 4. Foundation tasks (prove the failure modes)

Synthetic probes (\(T = 96\), ~15k params) force dual-role overwrite and exact order into the open. These define **what FEN is for**, not which write wins on every dataset.

### Distracted counting

Static **ID** at \(t=0\), then noisy **count** events. Label = ID × count bin.  
Success: high joint accuracy **and** high ID accuracy.

### Ordered recall (recall5)

Recover five symbols in order. Primary metric: **exact** full-sequence accuracy (token accuracy alone is not success).

### Foundation results (incl. LSTM)

Parameter-matched residual, LSTM, and FEN modes ([`exp01`](fen_lab/exp01_baseline_dual_task.py), [`exp01b`](fen_lab/exp01b_lstm_baseline.py)).

**recall5 exact**

| Model | exact | token |
|-------|------:|------:|
| residual | 0.000 | ~0.10 |
| lstm | 0.000 | ~0.10 |
| fen_bag / fen_roll (pool) | ~0.00 | higher token only |
| fen_slot | **0.96–1.0** | ~1.0 |

**Distracted**

| Model | acc | id | count |
|-------|----:|---:|------:|
| residual | ~0.09 | ~0.11 | high |
| lstm | ~0.10 | ~0.10 | high |
| fen_bag | **~0.99** | **~1.0** | **~1.0** |
| fen_slot | ~0.19 | ~0.19 | ~1.0 |

```text
LSTM and residual sit at the floor on both foundation probes.
Bag solves dual-role; slots solve exact order.
Wrong topology is chance-level failure, not a small lag.
```

---

## 5. Operators (what write / read / deliver do)

### Soft tape and readout ([`exp02`](fen_lab/exp02_ode_fen_order_ablation.py), [`exp03`](fen_lab/exp03_write_vs_readout.py))

Soft address under a pooled head stays near floor on exact recall. **Cell-aligned (slot) readout** yields exact → 1.0. Hard write can order under pool. Bag is required for dual-role; hybrid **soft_bag_slot** solves both foundation tasks.

### Delivery ([`exp04`](fen_lab/exp04_mid_deliver.py))

| Pattern | Result |
|---------|--------|
| Explicit mid/final read \(\mathrm{head}([h,E])\) | Strong dual-role, lean pipe |
| Every-step reinject of \(E\) into \(h\) | Dual-role degraded, **pipe bloat** |

```text
Read the archive. Do not pour it into the pipe every step.
```

### Multi-pass discrete read ([`exp07`](fen_lab/exp07_multipass_read.py))

On distracted (already solved by 1-pass bag), a second full scan after a one-shot read of \(E\) stays lean but **does not beat** 1-pass bag. Residual two-pass stays at the floor. Reinject again shows fat pipes. **Default remains single-pass + final/mid read.**

### Shared board ([`exp08`](fen_lab/exp08_shared_board.py))

Partitioned streams (ID-only vs count-only experts): dual residual fails ID; dual FEN private or shared both solve (~0.93–0.95). Shared bag slightly edges private notebooks; communication can happen via **joint head over archives**, not only a live shared bus. Reinject shared board → fat pipe again.

---

## 6. Synthetic operator freeze

```text
CANONICAL
  propose → gate → D → write E → (usually) deplete → head([h, arch])
  deliver by read, not every-step reinject

BY TASK FAMILY
  dual-role / static facts     → bag + deplete
  exact ordered outputs        → hard / slot write and/or slot read
  long ordered classification  → roll (hybrid strong on raster sMNIST; see §8–10)
  multi-worker facts           → escrow outside each pipe; head or shared E
```

---

## 7. Real 1D waveforms ([`exp05`](fen_lab/exp05_real_data.py), [`exp05b`](fen_lab/exp05_forda.py))

~75k params. Residual fails (majority MIT-BIH / chance FordA, fat pipe). FEN learns. Roll is strong on both; bag needs longer FordA budgets to approach roll. LSTM is task-dependent (competitive late on MIT-BIH in some runs; weak on FordA). These support escrow outside pure toys; they are not the main place to rank every write (foundation + sMNIST do that more cleanly).

---

## 8. Hard-bench: sequential MNIST (sMNIST)

Foundation tasks hit **ceiling** for many FEN variants, so they are the wrong place to rank “which FEN is better.”  
**sMNIST** (pixel stream) is used as a hard sequential task with headroom.

**Protocol** ([`exp09`](fen_lab/exp09_smnist.py)): 28×28 → 20×20, \(T=400\), \(C=1\), 1500 train / 200 test per digit, ~100k params, seed 1, 10 epochs for the FEN sweep.

### Peak accuracy (FEN sweep)

| Model | best acc | @ep | pipe |
|-------|---------:|----:|-----:|
| residual | 0.102 | 1 | 15.7 |
| lstm (1-layer, same recipe) | 0.110 | 6 | low |
| fen_2pass_cold | 0.465 | 9 | ~9 |
| fen_bag | 0.661 | 10 | ~9 |
| fen_hard_bag | 0.719 | 9 | **~5** |
| fen_copy (bag write, **no** deplete) | 0.776 | 9 | ~11 |
| fen_reinject | 0.823 | 10 | **~18** |
| fen_roll | **0.881** | 8 | ~10 |
| fen_hybrid (bag + roll) | **0.906** | 10 | ~10 |

```text
hybrid 0.91  ≳  roll 0.88  >  reinject 0.82  >  copy 0.78  >  hard 0.72  >  bag 0.66
  ≫  2pass 0.47  ≫  1-layer LSTM ≈ residual ≈ chance (@10 ep)
```

### Early accuracy (epoch 1–2) — primary ranking signal

| Model | ep1 | ep2 | Note |
|-------|----:|----:|------|
| residual | 0.10 | 0.10 | no learning |
| lstm 1L @10ep | 0.10 | 0.10 | no learning |
| fen_2pass | 0.15 | 0.29 | weak |
| fen_bag | 0.24 | 0.36 | slow start |
| fen_hard_bag | 0.28 | 0.34 | slow start |
| fen_copy | 0.35 | 0.49 | better than bag early |
| fen_reinject | 0.23 | 0.39 | slow; later pipe pathology |
| **fen_roll** | **0.64** | **0.80** | already past bag’s final best by ep2 |
| **fen_hybrid** | **0.67** | 0.71 | strongest ep1 |

```text
After 2 epochs:
  roll 0.80  ≈  bag’s best after 10 epochs (0.66) and ≈ best LSTM after 30 epochs (0.80)
```

Early accuracy shows **how direct the learning signal is**. Roll and hybrid put useful structure in the escrow immediately; bag is a slower integrator; residual/1L-LSTM do not move. That gap is as important as the final leaderboard.

### Interpretation of sMNIST variants

| Variant | Lesson |
|---------|--------|
| **roll / hybrid** | Best for long **ordered scans**; hybrid best peak; roll best pure early climb |
| **bag** | Works but **slow and weaker** on pixel streams (unlike dual-role toys) |
| **copy (no deplete)** | Beats bag here → deplete is **task-dependent**, not always mandatory for classification scans |
| **reinject** | Competitive peak, **worst pipe** (~18) → reject as architecture |
| **2-pass** | Fails ranking; not the hard-task upgrade |
| **hard tape** | Mid pack; lean pipe; not the sMNIST default |

---

## 9. LSTM honesty sweep ([`exp09b`](fen_lab/exp09b_lstm_smnist_sweep.py))

exp09’s 1-layer LSTM @10 ep sat near chance. A dedicated sweep tests whether that was only under-training.

**Same data protocol**, 30 epochs, variants: 1/2/3 layers, wider nets (~150k), higher LR, dropout.

| Variant | best | ep1 | ep2 | to 50% | to 80% |
|---------|-----:|----:|----:|-------:|-------:|
| lstm_2L_hiLR | 0.102 | 0.10 | 0.10 | — | — |
| lstm_1L_wide | 0.170 | 0.10 | 0.13 | — | — |
| lstm_1L | 0.477 | 0.10 | 0.10 | — | — |
| lstm_2L | 0.719 | 0.10 | 0.12 | ep16 | — |
| lstm_2L_wide | 0.720 | 0.10 | 0.26 | ep23 | — |
| lstm_2L_drop | 0.774 | 0.10 | 0.15 | ep13 | — |
| **lstm_3L (best)** | **0.802** | 0.10 | 0.23 | ep15 | **ep30** |

### FEN vs best LSTM (honest comparison)

| | fen_roll (exp09) | fen_hybrid (exp09) | best LSTM 3L (exp09b) |
|--|-----------------:|-------------------:|----------------------:|
| Best acc | **0.881** | **0.906** | 0.802 |
| Epochs to that best | **8–10** | **10** | **30** |
| ep1 | **0.64** | **0.67** | 0.10 |
| ep2 | **0.80** | 0.71 | 0.23 |

**Conclusions:**

1. **FEN beats LSTM** on this hard sequential protocol: higher peak with **far fewer** epochs.  
2. LSTM is **not** permanently stuck at chance if given depth and time — 3L reaches ~0.80 at ep30.  
3. That does **not** erase the architectural gap: even if an LSTM later reached a high final score, **early accuracy** shows it struggles to move useful signal in the first epochs. Roll is already at **0.80 by epoch 2**, where the best LSTM is still ~0.23 after a full honesty sweep.  
4. On more complex or longer data, architectures that only “eventually” learn with deep stacks and long schedules tend to degrade more; early dynamics are a leading indicator of that stress.

```text
Peak:     hybrid/roll > best LSTM (with 3× epochs for LSTM)
Early:    roll/hybrid ≫ any LSTM in the sweep
Efficiency: FEN reaches high accuracy while LSTM is still near floor
```

---

## 10. Permuted MNIST (pMNIST) — is roll a “weak CNN” of local pixels?

### Hypothesis

On **raster** sMNIST, early escrow commits might store **local spatial** structure along the scan (neighboring timesteps ≈ nearby pixels), i.e. a weak CNN-like path: local decisions → board → final head. That would explain roll/hybrid’s huge **ep1–ep2** lead.

**Test** ([`exp10`](fen_lab/exp10_pmnist.py)): same protocol as exp09, but a **fixed random permutation** of the \(T=400\) axis (`PERM_SEED=123`) is applied to every sample. Spatial neighborhoods along the sequence are destroyed; a **consistent** (scrambled) order remains across train/test.

**If spatial locality is the main story:** roll/hybrid peak and **especially ep1–ep2** should collapse toward bag; `roll − bag` gaps should shrink.  
**If ordered non-commutative escrow is the main story:** roll should still dominate bag on peak **and** early accuracy.

### Full results (10 ep, ~100k, seed 1)

| Model | best | ep1 | ep2 | last | pipe |
|-------|-----:|----:|----:|-----:|-----:|
| residual | 0.616 | **0.403** | **0.547** | 0.615 | 14.4 |
| fen_bag | 0.402 | 0.194 | 0.231 | 0.402 | 7.0 |
| fen_copy | 0.589 | 0.218 | 0.330 | 0.589 | 10.6 |
| **fen_roll** | **0.875** | **0.604** | **0.671** | 0.873 | 6.3 |
| fen_hybrid | 0.840 | 0.327 | 0.514 | 0.840 | 6.0 |
| lstm (1L) | 0.799 | 0.218 | **0.488** | 0.799 | 5.4 |
| lstm_3L | 0.634 | 0.174 | 0.302 | 0.634 | 4.9 |

```text
pMNIST peak:  roll 0.88  >  hybrid 0.84  >  lstm1L 0.80  >  residual 0.62  ≈ copy 0.59  >  lstm3L 0.63  >  bag 0.40
pMNIST ep1:   roll 0.60  >  residual 0.40  >  hybrid 0.33  >  lstm 0.22 ≈ copy  >  bag 0.19
pMNIST ep2:   roll 0.67  >  residual 0.55  >  hybrid 0.51  >  lstm 0.49  >  copy 0.33  >  bag 0.23
```

### Side-by-side with sMNIST (exp09) — peak **and** early

| Model | sMNIST best / ep1 / ep2 | pMNIST best / ep1 / ep2 |
|--------|-------------------------|-------------------------|
| residual | 0.10 / 0.10 / 0.10 | **0.62 / 0.40 / 0.55** |
| fen_bag | 0.66 / 0.24 / 0.36 | **0.40 / 0.19 / 0.23** |
| fen_copy | 0.78 / 0.35 / 0.49 | **0.59 / 0.22 / 0.33** |
| fen_roll | 0.88 / **0.64** / **0.80** | **0.88 / 0.60 / 0.67** |
| fen_hybrid | **0.91** / **0.67** / 0.71 | **0.84 / 0.33 / 0.51** |
| lstm 1L | 0.11 / 0.10 / 0.10 | **0.80 / 0.22 / 0.49** |

**Gaps `roll − bag` (early is primary for the locality probe)**

| Dataset | peak gap | **ep1 gap** | **ep2 gap** |
|---------|---------:|------------:|------------:|
| sMNIST | +0.22 | **+0.40** | **+0.44** |
| pMNIST | **+0.47** | **+0.41** | **+0.44** |

Peak gap **grows** (bag falls more than roll). **Ep1 and ep2 gaps do not shrink** — roll’s early lead over bag is essentially unchanged.

### Verdict

| Claim | Supported? |
|-------|------------|
| Roll needs **raster spatial locality** as its main advantage | **No** — peak almost unchanged (0.881 → 0.875); ep1 still ~0.60 |
| Roll = **ordered non-commutative escrow** over a *fixed* sequence order | **Yes** — still crushes bag on peak **and** ep1–ep2 under permutation |
| Hybrid early boost is partly tied to raster-friendly structure | **Plausible** — hybrid ep1 **0.67 → 0.33** (large early hit); peak only 0.91 → 0.84 |
| Bag is a poor write for long arbitrary-order scans | **Yes** — worst FEN on pMNIST (0.40), weak ep1–ep2 |
| Residual always chance on \(T=400\) | **No** — residual **learns** on pMNIST (0.62) with fat pipe; raster sMNIST was especially hostile |

```text
sMNIST roll/hybrid success
  ≠ mainly “local CNN deposits from raster neighbors”
  ≈ “structured ordered vault + final read”
     works for raster order and for a fixed random order

Early accuracy still ranks: roll moves useful signal immediately;
bag and 1L LSTM (on raster) do not.
```

Permutation destroys **spatial adjacency** but keeps a **consistent temporal layout** across samples. Roll exploits that consistency; it does not require 2D neighborhoods.

**vs LSTM on pMNIST (same 10 ep):** roll still wins peak (0.88 vs 0.80) and especially early (ep1 **0.60 vs 0.22**, ep2 **0.67 vs 0.49**). lstm_3L at 10 ep (0.63) is weaker than 1L here; depth needs longer schedules (as in exp09b).

---

## 11. Conclusions

### Established

1. **Dual-state + escrow** fixes residual dual-load on foundation dual-role; residual fat-pipe failure also appears on long scans when the task is hostile (raster sMNIST).  
2. **LSTM fails foundation probes** (exact recall 0; distracted joint ~0.10).  
3. **Topology must match the task:** bag for dual-role; slots for exact lists; **roll (or hybrid) for long ordered classification streams**.  
4. **Delivery is read, not continuous reinject** (pipe norms).  
5. On **sMNIST**, FEN **beats LSTM** on peak and—more importantly—on **ep1–ep2**; roll/hybrid dominate.  
6. On **pMNIST**, roll’s peak and **early** lead over bag **survive** permutation → advantage is **ordered escrow**, not primarily local spatial CNN-like deposits.  
7. **Early accuracy is first-class evidence** of gradient usefulness and architectural stability. A late catch-up does not make two models equal.

### Task-dependent notes

| Setting | Prefer |
|---------|--------|
| Dual-role / static facts | `fen_bag` + deplete |
| Exact ordered multi-token out | hard / slot |
| Long ordered classification (sMNIST / pMNIST) | **`fen_roll`**; hybrid strong on raster, softer early under perm |
| Deplete always? | Strong for dual-role; on sMNIST **copy** beat bag — task-dependent |
| Multi-pass / reinject as default | **No** |

### Not claimed

- Universal SOTA on vision or language  
- That bag is best on every domain  
- That roll is a substitute for real CNNs  
- That LSTM can never match a final number with unlimited tuning — **early-learning and efficiency** gaps remain the architectural point  

---

## 12. Experiments

Run on Colab/Kaggle GPU: paste a full file from [`fen_lab/`](fen_lab/).  
Deps: `torch`, `numpy`; `pandas` for some data paths (see `requirements.txt`).

| Exp | File | Role |
|-----|------|------|
| 01 | [`exp01_baseline_dual_task.py`](fen_lab/exp01_baseline_dual_task.py) | FEN family on foundation probes |
| 01b | [`exp01b_lstm_baseline.py`](fen_lab/exp01b_lstm_baseline.py) | LSTM + residual + bag + slot on foundation probes |
| 02 | [`exp02_ode_fen_order_ablation.py`](fen_lab/exp02_ode_fen_order_ablation.py) | Soft-tape order ablations |
| 03 | [`exp03_write_vs_readout.py`](fen_lab/exp03_write_vs_readout.py) | Write × readout grid |
| 04 | [`exp04_mid_deliver.py`](fen_lab/exp04_mid_deliver.py) | Mid-read vs reinject |
| 05 | [`exp05_real_data.py`](fen_lab/exp05_real_data.py) | MIT-BIH |
| 05b | [`exp05_forda.py`](fen_lab/exp05_forda.py) | FordA |
| 06 | [`exp06_cifar100.py`](fen_lab/exp06_cifar100.py) | Sequential CIFAR-100 (patch/pixel) |
| 07 | [`exp07_multipass_read.py`](fen_lab/exp07_multipass_read.py) | Multi-pass discrete read |
| 08 | [`exp08_shared_board.py`](fen_lab/exp08_shared_board.py) | Dual experts + shared board |
| 09 | [`exp09_smnist.py`](fen_lab/exp09_smnist.py) | sMNIST hard-bench FEN variants |
| 09b | [`exp09b_lstm_smnist_sweep.py`](fen_lab/exp09b_lstm_smnist_sweep.py) | Best-effort LSTM sweep on sMNIST |
| 10 | [`exp10_pmnist.py`](fen_lab/exp10_pmnist.py) | pMNIST: locality vs ordered-escrow test |

---

## 13. Summary

Feature-Escrow Networks keep an active residual **pipe** and an external **escrow**: resolved features are gated into the archive and (when appropriate) removed from the pipe, then read when needed—like clearing nutrients from the intestinal lumen into the bloodstream.

On synthetic probes that isolate dual-role retention and exact ordered memory, residual networks and LSTMs remain near chance while topology-matched FEN modes reach high accuracy. On long sequential digit streams, **channel-roll FEN leads on both peak and epoch-1/2 accuracy**, beating multi-layer LSTMs in sample efficiency; **permuted MNIST shows that this lead is not mainly “local CNN deposits from raster neighbors,”** but **ordered non-commutative escrow over a consistent sequence**, with early accuracy remaining the sharpest ranking signal.
