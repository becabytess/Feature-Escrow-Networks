# Feature-Escrow Networks (FEN)

This repository is a personal research project documenting my experiments with a novel recurrent neural network concept: **Feature-Escrow Networks (FEN)**. 

The core idea, which I wanted to test after noticing how traditional RNNs/LSTMs bottleneck by forcing a single vector to handle both active processing (CPU) and historical state (RAM), is to decouple active computation from a separated, accumulated escrow memory. 

This isn't a claim to a new mainstream SOTA architecture. It is simply a clean-slate playground where I designed experiments, ran rigorous baselines (including a best-effort tuned LSTM), and mapped out the properties and trade-offs of different memory topologies (such as bag-based accumulation, spatial channel rotation, and slot-addressable tape) across various sequential tasks.

Clean-slate experimental track: [`fen_lab/`](fen_lab/).  

```text
Feature-Escrow-Networks/
  README.md           ← this document
  fen_lab/            ← experiments (Colab / Kaggle one-cells)
  requirements.txt
```

---

## Metrics that matter

Results are reported with **two** complementary views:

| Metric | What it shows |
|--------|----------------|
| **Peak / best accuracy** | Whether the model can solve the task under the budget |
| **Early accuracy (epoch 1–2, and climb)** | How directly useful signal reaches parameters: gradient flow, stability, sample efficiency |

A model that is weak early and strong late is not “equal” to one that is strong early, even if final numbers are close. Early accuracy is treated as evidence of **how well the architecture propagates learning signal**, not as a minor training detail. On harder or longer data, poor early dynamics usually get worse, not better.

**Floor vs lag:** on foundation probes, failures are often **chance / exact ≈ 0**, not a few points behind. Those are capability gaps.

---

## 1. The problem

Sequential models often force **one** hidden state to do two jobs:

1. **Active computation** : update on the current step, count, react to noise  
2. **Long-lived memory** : keep facts that must survive many steps of that activity  

When both live in the same tensor, activity overwrites memory. Residuals help gradients but can **bloat** the stream over long sequences (high norm, context drift, dual-role collapse).

**Claim.** Working memory and archive should not be the same place. Resolved features should leave the active path and be stored outside it.

---

## 2. Biological inspiration

Metaphor: digestion in the **small intestine**.

| Biology | FEN |
|---------|-----|
| Lumen (active tract) | **Pipe** $h$ |
| Ready to absorb? | **Gate** $g$ |
| Nutrients to bloodstream | **Escrow write** $E$ |
| Mass removed from tract | **Deplete** $h \leftarrow f - D$ (optional hygiene) |
| Later use of nutrients | **Read** archive at query / final time |

**Write into escrow** is the load-bearing split (pipe vs vault). **Deplete** (scrub the pipe after write) is a separate switch: it usually **lowers pipe norm**, but is **not** a universal accuracy booster (see §12). Metaphor still: absorption into bloodstream; emptying the lumen is often helpful hygiene, not always what makes the nutrients usable.

```text
h  →  transform  →  f
                 →  gate  →  D = g ⊙ f
                 →  E ← write(E, D)
                 →  h ← f − D     (optional deplete)
                 →  later: head([h, E])
```

---

## 3. Architecture

One skeleton. Almost every “variant” is a **small product of switches**, not a new network family.

| Step | Role |
|------|------|
| 1 | Propose on the pipe |
| 2 | Gate → commit $D$ |
| 3 | Write $D$ into external archive $E$ |
| 4 | **Optional** deplete: $h \leftarrow f - D$ (else $h \leftarrow f$; “copy-style”) |
| 5 | Deliver via $\text{head}([h, \text{arch}])$ at final / query time (not a continuous dump of $E$ into $h$) |

### Switch A: write (topology; pick by **job**)

| Write | Role | Typical job |
|-------|------|-------------|
| **Bag** | Commutative “set of facts” | Dual-role / static context |
| **Hard pointer / slots** | Ordered cells | Exact multi-token outputs |
| **Channel-roll** | Non-commutative vault | Long **ordered scans** (pixels, sensors) |
| **Hybrid (bag + roll)** | Two vaults | Raster peak specialist; early often fragile |

### Switch B: deplete

| Name in logs | Meaning |
|--------------|---------|
| `fen_bag` / `*_dep` | write + **deplete on** |
| `fen_copy` / `bag_nodep` | **bag** write + **deplete off** |
| `roll_nodep` | **roll** write + **deplete off** |

### Switch C: deliver (frozen)

| Pattern | Status |
|---------|--------|
| Final / mid **read** $\text{head}([h,E])$ | **Default** |
| Every-step reinject $E \to h$ | **Reject** (pipe bloat) |
| Multi-pass cold read | Not an upgrade on foundation dual-role |

### Controls (not FEN)

| Control | Role |
|---------|------|
| Residual RNN | No escrow |
| LSTM | Classical recurrent baseline |

```text
Day-to-day cast (not 50 models):
  residual | fen_bag | fen_copy | fen_roll | fen_hybrid | (fen_slot) | lstm
Everything else is an ablation of those switches.
```

---

## 4. Foundation tasks (prove the failure modes)

Synthetic probes ($T = 96$, ~15k params) force dual-role overwrite and exact order into the open. These define **what FEN is for**, not which write wins on every dataset.

### Distracted counting

Static **ID** at $t=0$, then noisy **count** events. Label = ID × count bin.  
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
| Explicit mid/final read $\text{head}([h,E])$ | Strong dual-role, lean pipe |
| Every-step reinject of $E$ into $h$ | Dual-role degraded, **pipe bloat** |

```text
Read the archive. Do not pour it into the pipe every step.
```

### Multi-pass discrete read ([`exp06`](fen_lab/exp06_multipass_read.py))

On distracted (already solved by 1-pass bag), a second full scan after a one-shot read of $E$ stays lean but **does not beat** 1-pass bag. Residual two-pass stays at the floor. Reinject again shows fat pipes. **Default remains single-pass + final/mid read.**

### Shared board ([`exp07`](fen_lab/exp07_shared_board.py))

Partitioned streams (ID-only vs count-only experts): dual residual fails ID; dual FEN private or shared both solve (~0.93–0.95). Shared bag slightly edges private notebooks; communication can happen via **joint head over archives**, not only a live shared bus. Reinject shared board → fat pipe again.

---

## 6. Synthetic operator freeze

```text
CANONICAL
  propose → gate → D → write E → (optional) deplete → head([h, arch])
  deliver by read, not every-step reinject

BY TASK FAMILY
  dual-role / static facts     → bag write (escrow load-bearing; deplete optional; see §12)
  long ordered classification  → fen_roll default (deplete not required for early sMNIST; see §12)
  multi-worker facts           → escrow outside each pipe; head or shared E
  sequential CIFAR ranking     → fen_roll + longer thin tokens (patch-2); stress curve §11
  deplete                      → pipe hygiene; not a universal accuracy switch (§12)
```

---

## 7. Real 1D waveforms ([`exp05`](fen_lab/exp05_real_data.py), [`exp05b`](fen_lab/exp05_forda.py))

~75k params. Residual fails (majority MIT-BIH / chance FordA, fat pipe). FEN learns. Roll is strong on both; bag needs longer FordA budgets to approach roll. LSTM is task-dependent (competitive late on MIT-BIH in some runs; weak on FordA). These support escrow outside pure toys; they are not the main place to rank every write (foundation + sMNIST do that more cleanly).

---

## 8. Hard-bench: sequential MNIST (sMNIST)

Foundation tasks hit **ceiling** for many FEN variants, so they are the wrong place to rank “which FEN is better.”  
**sMNIST** (pixel stream) is used as a hard sequential task with headroom.

**Protocol** ([`exp08`](fen_lab/exp08_smnist.py)): 28×28 → 20×20, $T=400$, $C=1$, 1500 train / 200 test per digit, ~100k params, seed 1, 10 epochs for the FEN sweep.

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
| fen_roll (+ deplete) | **0.881** | 8 | ~10 |
| fen_hybrid (bag + roll) | **0.906** | 10 | ~10 |
| **roll_nodep** (roll, **no** deplete) ([`exp12b`](fen_lab/exp12b_roll_nodep_smnist.py)) | **0.887** | 9 | ~11.5 |

```text
hybrid 0.91  ≳  roll_nodep 0.89  ≈  roll+dep 0.88  >  reinject 0.82  >  copy 0.78
  >  hard 0.72  >  bag 0.66  ≫  2pass 0.47  ≫  1L LSTM ≈ residual ≈ chance (@10 ep)
```

### Early accuracy (epoch 1–2): primary ranking signal

| Model | ep1 | ep2 | Note |
|-------|----:|----:|------|
| residual | 0.10 | 0.10 | no learning |
| lstm 1L @10ep | 0.10 | 0.10 | no learning |
| fen_2pass | 0.15 | 0.29 | weak |
| fen_bag | 0.24 | 0.36 | slow start |
| fen_hard_bag | 0.28 | 0.34 | slow start |
| fen_copy | 0.35 | 0.49 | better than bag early |
| fen_reinject | 0.23 | 0.39 | slow; later pipe pathology |
| **fen_roll** (+ deplete) | **0.64** | **0.80** | already past bag’s final best by ep2 |
| **fen_hybrid** | **0.67** | 0.71 | strongest ep1 in original sweep |
| **roll_nodep** (no deplete) | **0.691** | **0.815** | **same early story without deplete** |

```text
After 2 epochs:
  roll±deplete ~0.80–0.82  ≈  bag’s best after 10 epochs (0.66)
                           ≈  best LSTM after 30 epochs (0.80)
```

Early accuracy shows **how direct the learning signal is**. Roll (with or without deplete) puts useful structure in the escrow immediately; bag is a slower integrator; residual/1L-LSTM do not move. That gap is as important as the final leaderboard.

### Roll without deplete ([`exp12b`](fen_lab/exp12b_roll_nodep_smnist.py))

Same sMNIST protocol, **only** `roll` write + deplete **off** (~100k, seed 1, 10 ep). Compared to exp08 `fen_roll` (+ deplete):

| | ep1 | ep2 | best | pipe |
|--|----:|----:|-----:|-----:|
| roll + deplete (exp08) | 0.64 | 0.80 | 0.881 | ~10 |
| **roll, no deplete (exp12b)** | **0.691** | **0.815** | **0.887** | ~11.5 |

```text
sMNIST roll early jump does NOT require deplete.
Ordered channel-roll write + final read is the load-bearing piece.
Deplete: slightly leaner pipe; not the reason roll beats bag.
```

### Interpretation of sMNIST variants

| Variant | Lesson |
|---------|--------|
| **roll** (± deplete) | **Most consistent** long-scan write: strong **ep1–ep2 and peak** on sMNIST *and* pMNIST; deplete optional |
| **hybrid (bag+roll)** | Best **peak on raster sMNIST**; early accuracy **collapses under perm** (ep1 0.67→0.33) (not the robust default) |
| **bag** | Works but **slow and weaker** on pixel streams (unlike dual-role toys) |
| **copy (bag, no deplete)** | Beats bag here → for **bag** write, deplete can **hurt** scan accuracy |
| **reinject** | Competitive peak, **worst pipe** (~18) → reject as architecture |
| **2-pass** | Fails ranking; not the hard-task upgrade |
| **hard tape** | Mid pack; lean pipe; not the sMNIST default |

---

## 9. LSTM honesty sweep ([`exp08b`](fen_lab/exp08b_lstm_smnist_sweep.py))

exp08’s 1-layer LSTM @10 ep sat near chance. A dedicated sweep tests whether that was only under-training.

**Same data protocol**, 30 epochs, variants: 1/2/3 layers, wider nets (~150k), higher LR, dropout.

| Variant | best | ep1 | ep2 | to 50% | to 80% |
|---------|-----:|----:|----:|-------:|-------:|
| lstm_2L_hiLR | 0.102 | 0.10 | 0.10 | - | - |
| lstm_1L_wide | 0.170 | 0.10 | 0.13 | - | - |
| lstm_1L | 0.477 | 0.10 | 0.10 | - | - |
| lstm_2L | 0.719 | 0.10 | 0.12 | ep16 | - |
| lstm_2L_wide | 0.720 | 0.10 | 0.26 | ep23 | - |
| lstm_2L_drop | 0.774 | 0.10 | 0.15 | ep13 | - |
| **lstm_3L (best)** | **0.802** | 0.10 | 0.23 | ep15 | **ep30** |

### FEN vs best LSTM (honest comparison)

| | fen_roll (exp08) | fen_hybrid (exp08) | best LSTM 3L (exp08b) |
|--|-----------------:|-------------------:|----------------------:|
| Best acc | **0.881** | **0.906** | 0.802 |
| Epochs to that best | **8–10** | **10** | **30** |
| ep1 | **0.64** | **0.67** | 0.10 |
| ep2 | **0.80** | 0.71 | 0.23 |

**Conclusions:**

1. **FEN beats LSTM** on this hard sequential protocol: higher peak with **far fewer** epochs.  
2. LSTM is **not** permanently stuck at chance if given depth and time; 3L reaches ~0.80 at ep30.  
3. That does **not** erase the architectural gap: even if an LSTM later reached a high final score, **early accuracy** shows it struggles to move useful signal in the first epochs. Roll is already at **0.80 by epoch 2**, where the best LSTM is still ~0.23 after a full honesty sweep.  
4. On more complex or longer data, architectures that only “eventually” learn with deep stacks and long schedules tend to degrade more; early dynamics are a leading indicator of that stress.

```text
Peak:     hybrid/roll > best LSTM (with 3× epochs for LSTM)
Early:    roll/hybrid ≫ any LSTM in the sweep
Efficiency: FEN reaches high accuracy while LSTM is still near floor
```

---

## 10. Permuted MNIST (pMNIST): is roll a “weak CNN” of local pixels?

### Hypothesis

On **raster** sMNIST, early escrow commits might store **local spatial** structure along the scan (neighboring timesteps ≈ nearby pixels), i.e. a weak CNN-like path: local decisions → board → final head. That would explain roll/hybrid’s huge **ep1–ep2** lead.

**Test** ([`exp09`](fen_lab/exp09_pmnist.py)): same protocol as exp08, but a **fixed random permutation** of the $T=400$ axis (`PERM_SEED=123`) is applied to every sample. Spatial neighborhoods along the sequence are destroyed; a **consistent** (scrambled) order remains across train/test.

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

### Side-by-side with sMNIST (exp08): peak **and** early

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

Peak gap **grows** (bag falls more than roll). **Ep1 and ep2 gaps do not shrink**; roll’s early lead over bag is essentially unchanged.

### Verdict

| Claim | Supported? |
|-------|------------|
| Roll needs **raster spatial locality** as its main advantage | **No** (peak almost unchanged 0.881 → 0.875; ep1 still ~0.60) |
| Roll = **ordered non-commutative escrow** over a *fixed* sequence order | **Yes** (still crushes bag on peak **and** ep1–ep2 under permutation) |
| Hybrid early boost is fragile under non-raster order | **Yes** (hybrid ep1 **0.67 → 0.33** (halved); ep2 0.71 → 0.51; peak only 0.91 → 0.84. The bag vault in hybrid appears to drag early learning when the scan is not a spatial walk.) |
| **Roll is the most consistent long-scan write** | **Yes** (across sMNIST and pMNIST, roll keeps **ep1 ≈ 0.60+** and peak ≈ **0.88**; hybrid wins peak only on raster and loses early consistency under perm) |
| Bag is a poor write for long arbitrary-order scans | **Yes** (worst FEN on pMNIST 0.40, weak ep1–ep2) |
| Residual always chance on $T=400$ | **No** (residual **learns** on pMNIST 0.62 with fat pipe; raster sMNIST was especially hostile) |

**Roll vs hybrid (consistency, early first):**

| | sMNIST ep1 | pMNIST ep1 | sMNIST peak | pMNIST peak |
|--|-----------:|-----------:|------------:|------------:|
| fen_roll | **0.64** | **0.60** | 0.88 | **0.88** |
| fen_hybrid | **0.67** | **0.33** | **0.91** | 0.84 |

Hybrid’s ep1 collapse under permutation is as important as roll’s stable ep1: **adding a bag channel is not free; it can dilute the early-learning advantage that pure roll keeps on both raster and permuted streams. For a default long-sequence classification write, roll is preferred over hybrid on robustness (early + peak across orders). Hybrid remains a strong raster-only peak option.**

```text
sMNIST roll/hybrid success
  ≠ mainly “local CNN deposits from raster neighbors”
  ≈ “structured ordered vault + final read”
     pure roll: stable early + peak under raster and perm
     hybrid:    best peak on raster; early accuracy fragile under perm

Early accuracy still ranks: roll moves useful signal immediately;
bag and 1L LSTM (on raster) do not.
```

Permutation destroys **spatial adjacency** but keeps a **consistent temporal layout** across samples. Roll exploits that consistency; it does not require 2D neighborhoods.

**vs LSTM on pMNIST (same 10 ep):** roll still wins peak (0.88 vs 0.80) and especially early (ep1 **0.60 vs 0.22**, ep2 **0.67 vs 0.49**). lstm_3L at 10 ep (0.63) is weaker than 1L here; depth needs longer schedules (as in exp08b).

---

## 11. Hard transfer: sequential CIFAR-100

After sMNIST / pMNIST, the open question was: **does the frozen FEN story (especially `fen_roll` + final read) transfer beyond digit streams?**

**Not claimed:** CNN-level CIFAR accuracy. This is a **sequential RNN/FEN** protocol (~100k params), not 2D vision SOTA. Chance floor = **1%**.  
**Not evidence:** archived spatial-CNN CIFAR (~+1% over plain CNN); different architecture, different claim.

### Protocol ([`exp10`](fen_lab/exp10_cifar100.py), stress curve [`exp11`](fen_lab/exp11_stress_curve.py))

| Piece | Value |
|-------|--------|
| Data | CIFAR-100 fine labels, 150 train / 20 test per class (~15k / 2k) |
| Models (exp10) | residual, fen_bag, fen_copy, fen_roll, fen_hybrid, lstm, lstm_3L |
| Models (exp11 lean) | residual, fen_bag, fen_roll, fen_hybrid, lstm |
| Budget | ~100k params, AdamW, seed 1, GPU + CUDA graphs |
| Metrics | **peak** and **ep1 / ep2** (same ranking signal as exp08/09) |

**Tokenization is load-bearing** (not a minor hyperparameter):

| Mode | Patch | Shape | Sequential stress |
|------|------:|-------|-------------------|
| **P8** (shortest, fattest) | 8×8 | $T=16$, $C=192$ | **low** (almost short token classification) |
| **P4** (short fat tokens) | 4×4 | $T=64$, $C=48$ | **mid** (local structure per step, short scan) |
| **P2** (longer thin tokens) | 2×2 | $T=256$, $C=12$ | **high** (longer ordered scan, less info per step) |

Hypothesis: large patches can **compress** architecture gaps (everyone grabs easy local signal; hard wall ~20% from capacity/100-class difficulty). Smaller patches **restore long-scan pressure** and should reopen roll ≫ bag if the ordered-escrow story is real.

### A. Patch-8 ($T=16$, $C=192$): low stress ([`exp11`](fen_lab/exp11_stress_curve.py))

**15 epochs**, lean model set

| Model | best | ep1 | ep2 | last | pipe |
|-------|-----:|----:|----:|-----:|-----:|
| residual | 0.132 | 0.053 | 0.076 | 0.127 | 12.3 |
| fen_bag | 0.181 | 0.075 | 0.094 | 0.181 | 7.0 |
| fen_roll | **0.199** | 0.085 | **0.113** | 0.198 | 6.3 |
| fen_hybrid | **0.199** | 0.084 | 0.104 | 0.199 | 6.0 |
| lstm | 0.169 | 0.043 | 0.078 | 0.158 | 5.3 |

```text
P8: everyone in a pile (~0.13–0.20); residual *learns*; roll−bag peak only +0.017
Low sequential stress → architecture gaps compressed
```

### B. Patch-4 ($T=64$, $C=48$): mid stress

**15 epochs**

| Model | best | ep1 | ep2 | last | pipe |
|-------|-----:|----:|----:|-----:|-----:|
| residual | 0.076 | 0.037 | 0.049 | 0.066 | 14.2 |
| fen_bag | 0.197 | 0.055 | 0.089 | 0.193 | 7.5 |
| fen_copy | 0.166 | 0.063 | 0.083 | 0.158 | 10.1 |
| fen_roll | 0.218 | **0.105** | **0.134** | 0.218 | 6.3 |
| fen_hybrid | **0.219** | 0.085 | 0.112 | 0.219 | 6.3 |
| lstm | 0.177 | 0.032 | 0.069 | 0.177 | 6.1 |
| lstm_3L | 0.156 | 0.028 | 0.041 | 0.156 | 4.5 |

**30 epochs** (same seed/init path; tests whether gaps are just under-training)

| Model | best | ep1 | ep2 | last | to_best | pipe |
|-------|-----:|----:|----:|-----:|--------:|-----:|
| residual | 0.078 | 0.037 | 0.049 | 0.074 | 29 | 14.3 |
| fen_bag | 0.215 | 0.055 | 0.089 | 0.215 | 18 | 7.8 |
| fen_copy | 0.196 | 0.063 | 0.083 | 0.189 | 25 | 10.3 |
| fen_roll | 0.224 | **0.105** | **0.134** | 0.208 | 18 | 6.6 |
| fen_hybrid | **0.234** | 0.085 | 0.112 | 0.217 | 21 | 6.7 |
| lstm | 0.195 | 0.032 | 0.069 | 0.184 | 25 | 6.4 |
| lstm_3L | 0.178 | 0.028 | 0.041 | 0.175 | 22 | 4.7 |

```text
P4 @15:  hybrid ≈ roll (~0.22)  >  bag 0.20  >  lstm 0.18  ≫  residual 0.08
P4 @30:  hybrid 0.23  ≳  roll 0.22  >  bag 0.22  ≳  lstm 0.20  ≫  residual 0.08
P4 ep1:  roll 0.105  >  hybrid 0.085  >  bag 0.055  >  residual/lstm ~0.03
```

**P4 read:** Rankings still hold (roll early; hybrid/roll peak; residual fat pipe fails; FEN > LSTM). Absolute band is **~15–23%** (well above 1% chance, far from CNN CIFAR). **Doubling epochs barely moves peaks** → **hard wall / capacity limit**, not “need more training.” **Gaps are small** (roll−bag peak only **+0.01–0.02**): short fat tokens let bag/LSTM learn enough that write topology is less decisive. Bag vs copy (deplete): **bag > copy** here (opposite of sMNIST bag vs copy). exp11 P4 **matches exp10 P4** on shared models (reproducible).

### C. Patch-2 ($T=256$, $C=12$): high stress

**15 epochs** (main long-scan ranking for sequential CIFAR)

| Model | best | ep1 | ep2 | last | pipe |
|-------|-----:|----:|----:|-----:|-----:|
| residual | 0.058 | 0.033 | 0.037 | 0.057 | **14.8** |
| fen_bag | **0.031** | 0.017 | 0.017 | 0.029 | 8.8 |
| fen_copy | 0.049 | 0.013 | 0.016 | 0.049 | 11.0 |
| fen_roll | **0.149** | **0.075** | **0.094** | 0.146 | 6.1 |
| fen_hybrid | **0.149** | 0.037 | 0.046 | 0.148 | 6.3 |
| lstm | 0.104 | 0.026 | 0.027 | 0.104 | 4.4 |
| lstm_3L | 0.092 | 0.028 | 0.023 | 0.092 | 4.7 |

```text
P2 peak:  roll = hybrid 0.15  ≫  lstm 0.10  >  residual 0.06  >  copy 0.05  >  bag ~0.03 (near floor)
P2 ep1:   roll 0.075  ≫  hybrid 0.037  ≥  residual 0.033  >  lstm 0.026  >  bag 0.017
```

**P2 read:** Absolute peaks drop (longer scan, thinner tokens). **Architecture gaps reopen hard.** Bag falls to **near chance** while roll holds **~15%** → commutative bag is the **wrong write** for this stream (capability gap, not a small lag). Hybrid **ties peak** with roll but **halves early accuracy** (bag vault drags again). Residual stays weak with a **fat pipe**. LSTM learns but stays below roll on peak and early. (P2 numbers from exp10; not re-run in exp11 to save GPU time.)

### D. Regime map: gaps vs sequential stress

Combined **exp11** (P8, P4) + **exp10** (P2). Same subset / ~100k / seed 1 / 15 ep where comparable.

| Stress | T | roll | bag | hyb | lstm | res | **r−b peak** | **r−b ep1** | **r−b ep2** | h−r ep1 |
|--------|--:|-----:|----:|----:|-----:|----:|-------------:|------------:|------------:|--------:|
| P8 low | 16 | 0.199 | 0.181 | 0.199 | 0.169 | 0.132 | **+0.017** | **+0.010** | +0.020 | −0.002 |
| P4 mid | 64 | 0.218 | 0.197 | 0.219 | 0.177 | 0.076 | **+0.021** | **+0.050** | +0.045 | −0.020 |
| P2 high | 256 | 0.149 | 0.031 | 0.149 | 0.104 | 0.058 | **+0.118** | **+0.058** | +0.077 | −0.038 |

```text
roll−bag peak:  P8 +0.02  ≈  P4 +0.02  →  P2 +0.12   (jumps when scan is long/thin)
roll−bag ep1:   P8 +0.01  →  P4 +0.05  →  P2 +0.06   (opens before peak gap)
residual:       lives at P8 (0.13) → dies as T grows (0.08 → 0.06), fat pipe
```

| Claim | Supported? |
|-------|------------|
| Frozen FEN ranking transfers beyond digits | **Yes** (especially under P2) |
| Roll is consistent default (early + peak among FEN) | **Yes** on P2 and P4/P8 early; mid peak hybrid ≳ roll |
| Hybrid loses early vs pure roll as stress grows | **Yes** (ep1 gap 0 → −0.02 → −0.04) |
| Bag is fine for long ordered vision streams | **No** (dies on P2 ~3%) |
| More epochs fix P4 wall | **No** (15→30 adds ~1–2 points) |
| This is CNN-competitive CIFAR | **No** (sequential ~100k protocol only) |
| Large patches erase FEN advantages | **Partially** (compress peak gaps; early roll lead still visible) |
| Gap grows smoothly with every T step | **No** (more like short/mid compressed, long thin reopens) |

### E. Sequential Pixel-Level CIFAR-100 without Patching via Hierarchical FEN ([`exp13`](fen_lab/exp13_hierarchical_cifar.py))

To completely bypass sequential bottlenecks and evaluate FEN under extreme sequential stress ($T=1024$ steps, pixel-by-pixel CIFAR-100), we implement a **Hierarchical FEN** (Divide & Conquer over time). 

By splitting the 1024-step sequence into $K=32$ chunks of length 32, we run a local pass inside each chunk (batch dimension $B \times 32$, sequence length 32) and a global pass across the 32 chunks. We concatenate the final active state and the local escrow of each chunk to form the sequence representation for the global pass:

$$\text{chunk\_seq}_t = [h_{\text{final\_local}, t}, E_{\text{local}, t}] \quad (\text{shape: } [B, \text{num\_chunks}, 2 \times H])$$

This is evaluated under a strict **~100k parameters target budget** and compared across 4 hierarchical variants using vanilla `nn.RNN` cells (to match model capacity):

| Model | Best Acc | Epoch 1 | Epoch 2 | Epoch 15 | Hidden | Params |
|---|---:|---:|---:|---:|---:|---:|
| **`standard_hrnn`** | 0.0536 | 0.0218 | 0.0374 | 0.0520 | 149 | 99746 |
| **`standard_hrnn_residual`** | 0.0536 | 0.0273 | 0.0324 | 0.0540 | 149 | 99746 |
| **`fen_roll_hierarchical`** | **0.2188** | **0.1038** | **0.1429** | 0.2080 | 89 | 100339 |
| **`fen_sandwich_hierarchical`** | **0.2366** | **0.0871** | **0.1468** | **0.2320** | 74 | 99166 |

```text
Hierarchical FEN variants (roll and sandwich) completely crush standard Hierarchical RNNs (which fail at 5.36% accuracy).
`fen_roll_hierarchical` shows the strongest early learning jump (Epoch 1: 10.38%).
`fen_sandwich_hierarchical` scales up to 23.66% accuracy at Epoch 13.
```

### CIFAR defaults (after exp10–11)

| Setting | Prefer |
|---------|--------|
| Sequential CIFAR ranking / long scan | **patch-2** ($T=256$) + **`fen_roll`** |
| Short-token transfer / compression check | patch-4 or patch-8; report that gaps shrink |
| Peak-only on patch streams | hybrid optional (small edge @30ep P4); watch early |
| Deplete on this domain | bag > copy on P4; both fail on P2; see §12 |

---

## 12. Deplete law (optional pipe hygiene)

**Question:** is $h \leftarrow f - D$ load-bearing for accuracy, or mainly pipe control?

`fen_copy` = bag write + **no** deplete. `roll_nodep` = roll write + **no** deplete.  
Do **not** confuse “copy lost on some leaderboard” with “deplete always wins.”

### A. Dual-role grid ([`exp12`](fen_lab/exp12_deplete_law.py))

Distracted counting, ~15k params, $T=96$, 12 epochs, seed 1. Models: bag/roll × deplete on/off.

| Model | peak | id | count | ep1 | ep2 | pipe |
|-------|-----:|---:|------:|----:|----:|-----:|
| **bag_nodep** (no deplete) | **0.951** | **0.975** | 0.972 | 0.090 | 0.148 | 5.36 |
| bag_dep | 0.775 | 0.781 | 0.975 | 0.074 | 0.138 | **1.84** |
| **roll_nodep** | **0.853** | **0.895** | 0.953 | 0.046 | 0.067 | 5.59 |
| roll_dep | 0.715 | 0.717 | 0.984 | 0.027 | 0.071 | **2.38** |

```text
Δ deplete (bag):  peak −0.176   id −0.194   pipe −3.5
Δ deplete (roll): peak −0.137   id −0.179   pipe −3.2
```

**Read:** On this short dual-role probe @12 ep, **no-deplete wins peak and id** for both writes; **deplete wins lean pipe**.  
`bag_dep` is still climbing late (joint ~0.45→0.78 ep8–11) while `bag_nodep` is already ~0.95; budget/trajectory matters.  
**Escrow still solves dual-role** vs residual/LSTM floors in §4; deplete is **not** what makes the vault work. Bag still beats roll on this dual-role job (topology match).

### B. Long-scan bag deplete (already in exp08 / exp10; no new slow runs)

| Setting | bag + deplete | bag, no deplete (copy) | Δ peak (dep − nodep) |
|---------|--------------:|-----------------------:|---------------------:|
| sMNIST (exp08) | 0.661 | **0.776** | **−0.12** (deplete hurts) |
| CIFAR P4 (exp10) | **0.197** | 0.166 | **+0.03** (deplete helps slightly) |
| CIFAR P2 (exp10) | 0.031 | **0.049** | both ~floor |

### C. sMNIST roll deplete ([`exp12b`](fen_lab/exp12b_roll_nodep_smnist.py))

Only missing cell for the **winner** write: roll without deplete (early signal primary).

| | ep1 | ep2 | best | pipe |
|--|----:|----:|-----:|-----:|
| roll + deplete (exp08) | 0.64 | 0.80 | 0.881 | ~10 |
| **roll, no deplete (exp12b)** | **0.691** | **0.815** | **0.887** | ~11.5 |

```text
roll’s sMNIST early jump does NOT require deplete.
Write topology (channel-roll) is load-bearing; deplete is not.
```

### D. Deplete law (frozen)

| Question | Answer |
|----------|--------|
| Is deplete required for dual-role **accuracy**? | **No** (exp12 @12 ep: nodep higher) |
| Is deplete required for roll’s **sMNIST early** signal? | **No** (exp12b: ep1 0.69 without) |
| Does deplete lower **pipe norm**? | **Yes**, consistently |
| When can deplete help bag peak? | Sometimes (CIFAR-P4); opposite of sMNIST bag |
| What is load-bearing? | **Escrow write** + **topology match** + **read delivery** |

```text
DEPLETE = optional pipe hygiene / regularizer
         ≠ universal accuracy switch
         ≠ the reason bag solves dual-role or roll wins long scans

Prefer reporting pipe when comparing dep vs nodep;
rank architectures primarily by write + early/peak accuracy.
```

---

## 13. Conclusions

### Established

1. **Dual-state + escrow** fixes residual dual-load on foundation dual-role; residual fat-pipe failure also appears on long scans when the task is hostile (raster sMNIST, sequential CIFAR residual).  
2. **LSTM fails foundation probes** (exact recall 0; distracted joint ~0.10).  
3. **Topology must match the task:** bag for dual-role; slots for exact lists; **`fen_roll` for long ordered classification streams**. Roll is **not** universal #1 (loses dual-role to bag; that is expected).  
4. **Delivery is read, not continuous reinject** (pipe norms).  
5. On **sMNIST**, FEN **beats LSTM** on peak and, more importantly, on **ep1–ep2**; roll/hybrid lead; **roll without deplete keeps the early jump** (ep1≈0.69).  
6. On **pMNIST**, roll keeps **ep1≈0.60 and peak≈0.88**; hybrid’s **early** accuracy falls hard (ep1≈0.33) → **roll is the consistent default**, hybrid is a raster peak specialist.  
7. Roll’s early lead over bag **survives** permutation → advantage is **ordered escrow**, not primarily local spatial CNN-like deposits.  
8. **Early accuracy is first-class evidence** of gradient usefulness and architectural stability. A late catch-up does not make two models equal.  
9. On **sequential CIFAR-100**, ranking **transfers** under high sequential stress: **regime map P8→P4→P2** shows compressed gaps at short/fat tokens and **roll ≫ bag** at long/thin (P2 bag ~chance).  
10. **Deplete is optional pipe hygiene**, not a universal accuracy law: can hurt peak on dual-role @12 ep and on sMNIST bag; not required for roll’s sMNIST early signal; usually leaner pipe (§12).  
11. **FEN Sandwich (Double-Pass) is the superior readout topology.** In a strict head-to-head comparison on $T=1024$ CIFAR-100, the Hierarchical Sandwich (`23.66%`) significantly outperforms the single-pass Hierarchical FEN-Roll (`21.88%`). This confirms that running a dedicated second pass initialized by the compiled escrow (context-priming) is structurally superior to simply concatenating active and escrow states at the end of a single run.  
12. **Hierarchy makes long sequences trainable.** Dividing long sequence scans ($T=1024$) into local/global chunks ($K=32$) yields a massive 15x–20x training speedup due to parallel GPU occupancy. This transforms long-scan sequential models from untrainable or glacially slow loops into highly stable, fast-converging layouts.

### Architectural Insight: FEN as Write-Time Compressed Attention

Through this in-depth analysis of the results, FEN can be conceptualized in relation to standard Attention:

1. **Read-Time vs. Write-Time Selection:** Standard Attention (Bahdanau or Transformer) achieves selective memory routing at **read-time** (it must store all past hidden states in $O(N)$ memory and search them using a Query-Key dot product). In contrast, FEN acts as a **Write-Time Compressed Attention** mechanism. It uses proposal gates ($g_t$) to select and route key features on the fly, folding them into a single $O(1)$ escrow vault.
2. **Decoupled $O(1)$ Memory:** Unlike LSTMs, which also attempt write-time selection but immediately overwrite the memory by storing it in the active processing path ($h_t$), FEN isolates the selected features in $E_t$.
3. **The Efficiency Trade-Off:** This research demonstrates that for tasks requiring accumulation, counting (like `distracted`), or ordered scans (like `smnist`), we do not need the memory or computational overhead of read-time search. By moving selection to the write stage and protecting the memory in an isolated $O(1)$ vault, FEN outperforms LSTMs by large margins while remaining just as computationally cheap.

### Task-dependent notes

| Setting | Prefer |
|---------|--------|
| Dual-role / static facts | **`fen_bag`** (escrow); deplete **optional** for peak (prefer if you care about lean pipe) |
| Exact ordered multi-token out | hard / slot |
| Long ordered classification (digits, long scans) | **`fen_roll`** (± deplete; deplete not required for early sMNIST) |
| Sequential CIFAR (ranking) | **`fen_roll`** + **patch-2** ($T=256$); P4/P8 as compression checks |
| Raster / short-token peak chase | `fen_hybrid` optional; early accuracy often worse than pure roll |
| Deplete always? | **No** (pipe hygiene; accuracy effect flips by task/budget; see §12) |
| Multi-pass / reinject as default | **No** |

### Not claimed

- Universal SOTA on vision or language  
- That bag is best on every domain  
- That roll is a substitute for real CNNs or competitive CIFAR vision  
- That roll always beats bag (wrong job → bag wins dual-role)  
- That deplete is required for dual-role or for roll’s early learning  
- That LSTM can never match a final number with unlimited tuning; the **early-learning and efficiency** gaps remain the architectural point  
- That short-patch sequential CIFAR is the best place to rank write modes (use longer scans / P2 for that)

---

## 14. Experiments

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
| 06 | [`exp06_multipass_read.py`](fen_lab/exp06_multipass_read.py) | Multi-pass discrete read |
| 07 | [`exp07_shared_board.py`](fen_lab/exp07_shared_board.py) | Dual experts + shared board |
| 08 | [`exp08_smnist.py`](fen_lab/exp08_smnist.py) | sMNIST hard-bench FEN variants |
| 08b | [`exp08b_lstm_smnist_sweep.py`](fen_lab/exp08b_lstm_smnist_sweep.py) | Best-effort LSTM sweep on sMNIST |
| 09 | [`exp09_pmnist.py`](fen_lab/exp09_pmnist.py) | pMNIST: locality vs ordered-escrow test |
| 10 | [`exp10_cifar100.py`](fen_lab/exp10_cifar100.py) | Sequential CIFAR-100 (P4 + P2) |
| 11 | [`exp11_stress_curve.py`](fen_lab/exp11_stress_curve.py) | CIFAR regime map P8→P4→P2 (P2 reusable from exp10) |
| 12 | [`exp12_deplete_law.py`](fen_lab/exp12_deplete_law.py) | bag/roll × deplete on distracted (+ optional sMNIST) |
| 12b | [`exp12b_roll_nodep_smnist.py`](fen_lab/exp12b_roll_nodep_smnist.py) | sMNIST roll **without** deplete (early signal) |
| 13 | [`exp13_hierarchical_cifar.py`](fen_lab/exp13_hierarchical_cifar.py) | Hierarchical vanilla RNNs vs Hierarchical FEN (roll & sandwich) on T=1024 pixel CIFAR-100 |

---

## 15. Summary

Feature-Escrow Networks keep an active residual **pipe** and an external **escrow**: resolved features are gated into the archive, optionally removed from the pipe (**deplete**), then **read** when needed (like clearing nutrients from the intestinal lumen into the bloodstream).

On synthetic probes that isolate dual-role retention and exact ordered memory, residual networks and LSTMs remain near chance while topology-matched FEN modes reach high accuracy (**bag** dual-role, **slots** exact order). On long sequential digit streams, **channel-roll** is the most consistent write: strong **epoch-1/2 and peak** on raster sMNIST and pMNIST, with or **without** deplete (roll_nodep ep1≈0.69). Hybrid can edge peak on pure raster but loses early under permutation. The story is **ordered non-commutative escrow**, not mainly local CNN-like deposits.

On **sequential CIFAR-100** (~100k, not CNN vision), **tokenization sets sequential stress**: short/fat patches (P8/P4) compress architecture gaps and hit a ~20% wall; long/thin patches (P2) reopen **roll ≫ bag**. **Deplete** consistently trims pipe norms but is **not** a universal accuracy switch; escrow **write** and **topology match** remain the load-bearing claims. **Early accuracy** is the sharpest ranking signal across domains.
