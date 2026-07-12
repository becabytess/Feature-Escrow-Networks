# Feature-Escrow Networks (FEN)

Clean-slate experimental track: [`fen_lab/`](fen_lab/).  
Earlier explorations are archived under [`history/`](history/) and are not part of this path’s evidence base.

```text
Feature-Escrow-Networks/
  README.md           ← this document
  fen_lab/            ← experiments (Colab one-cells)
  history/            ← archived early work
  requirements.txt
  LICENSE
```

---

## 1. The problem

Sequential models often force **one** hidden state to do two jobs at once:

1. **Active computation** — absorb the current step, update, count, react to noise  
2. **Long-lived memory** — keep facts that must survive many steps of that activity  

When both live in the same tensor, ongoing activity tends to **overwrite** static context. Residual connections help gradients flow, but over long sequences they can also accumulate unresolved features until the active stream becomes **bloated** (high norm, context drift, dual-role collapse).

**Claim.** Working memory and archive should not be the same place. Features that are resolved should leave the active path and be stored outside it.

---

## 2. Biological inspiration

The design is inspired by digestion in the **small intestine** (metaphor, not a biophysical model).

| Biology | Role | FEN analogue |
|---------|------|----------------|
| Lumen (gut tract) | Active processing of material still being broken down | **Pipe** \(h\) — active residual stream |
| Absorptive decision | Whether a nutrient is ready to leave the tract | **Gate** \(g\) |
| Absorption into bloodstream | Move resolved material out of the lumen | **Escrow write** into archive \(E\) |
| Physical removal from the tract | Volume leaves the gut | **Depletion** \(h \leftarrow f - D\) |
| Bloodstream → later use | Nutrients travel insulated from digestive chaos | **Archive read** at query or final time |

The important mechanical point is **removal**, not only copying. Storing a copy while leaving the pipe crowded does not free the active path.

```text
h  →  transform  →  f
                 →  gate  →  D = g ⊙ f
                 →  E ← write(E, D)
                 →  h ← f − D
                 →  later: head([h, E])
```

---

## 3. Architecture

All models in this track share one skeleton. Variants differ in **how** \(E\) is written and **how / when** it is read.

| Step | Operation |
|------|-----------|
| 1 | Propose on the pipe (residual-style transform) |
| 2 | Gate → commit \(D\) |
| 3 | Write \(D\) into external archive \(E\) |
| 4 | Deplete the pipe: \(h \leftarrow f - D\) |
| 5 | Deliver archive at the appropriate time, typically \(\mathrm{head}([h, \mathrm{arch}])\) |

| Write mode | Description | Suited to |
|------------|-------------|-----------|
| **Bag** | Additive, commutative escrow | Static facts, dual-role retention |
| **Hard pointer / slots** | Ordered cells; pointer advances on write | Exact ordered sequences |
| **Soft γ-tape** | Soft address + learned shift | Order when paired with cell-aligned readout |
| **Channel-roll** | Non-commutative update of the escrow vector | Long 1D sensor / waveform structure |

| Control | Description |
|---------|-------------|
| **Residual RNN** | Residual pipe only; no escrow |
| **LSTM** | Standard single-state recurrent baseline |

---

## 4. Foundation tasks

To test the claim, two synthetic tasks (\(T = 96\), parameter-matched at ~15k) were designed so that dual-role overwrite and ordered memory cannot be faked by a generic accuracy bump.

### Distracted counting

- At \(t = 0\), a static **ID** is presented.  
- Over the rest of the sequence the model must track noisy **count** events amid distractors.  
- Label: joint **ID × count bin** (30-way).  

**Metrics:** joint accuracy, plus separate ID and count accuracy. Count alone is not success if ID is lost.

### Ordered recall (recall5)

- Five symbols are presented early; distractors follow.  
- The model must recover the **full ordered sequence**.  
- **Primary metric: exact** sequence accuracy (all five correct). Token accuracy without exact match is not treated as solving the task.

---

## 5. Foundation results

Parameter-matched residual RNN, LSTM, and FEN variants on both tasks  
([`exp01`](fen_lab/exp01_baseline_dual_task.py), [`exp01b`](fen_lab/exp01b_lstm_baseline.py); seed 1).

### recall5 (exact sequence)

| Model | exact | token |
|-------|------:|------:|
| residual_rnn | 0.000 | 0.099 |
| lstm | 0.000 | 0.098 |
| fen_bag | 0.001 | ~0.19–0.34 |
| fen_roll | ~0.00 | ~0.3 |
| fen_slot | **0.96–1.0** | **~0.99–1.0** |

Residual and LSTM remain at exact zero. Bag and roll do not solve exact order under this protocol. Hard slot write reaches near-perfect exact accuracy.

### Distracted counting

| Model | acc | id | count |
|-------|----:|---:|------:|
| residual_rnn | 0.088 | 0.114 | 0.768 |
| lstm | 0.103 | 0.105 | 0.973 |
| fen_bag | **0.988** | **0.996** | **0.992** |
| fen_roll | ~0.82–0.94 | high | high |
| fen_slot | 0.185 | 0.185 | ~1.0 |

Residual and LSTM stay near the joint-label floor (count may improve while ID does not). Bag escrow solves dual-role retention. Slot write preserves count but loses the static ID.

### Summary

| Model | recall5 exact | distracted joint |
|-------|:-------------:|:----------------:|
| residual | floor | floor |
| lstm | floor | floor |
| fen_bag | floor | solved |
| fen_slot | solved | floor (ID) |

Matching the escrow topology to the task is required: bag for dual-role static facts, ordered slots for exact sequences. A single write algebra does not cover both probes.

---

## 6. Write and readout structure

### Soft-tape order ablations ([`exp02`](fen_lab/exp02_ode_fen_order_ablation.py))

Soft addressable tape under a pooled head stays near floor on exact recall. Interventions:

| Hypothesis | Change | recall5 exact |
|------------|--------|---------------|
| H1 | Remove bag channel | still fails |
| H2 | Sharpen address | still fails |
| H3 | Event-gated shift | still fails |
| H4 | Cell-aligned (slot) readout | **exact → 1.0** |

On distracted counting, models with a bag channel remain strong; pure order models without bag remain weak on ID.

### Write × readout grid ([`exp03`](fen_lab/exp03_write_vs_readout.py))

| Model | recall5 exact | distracted acc |
|-------|---------------:|---------------:|
| bag_pool | 0.003 | 0.973 |
| hard_pool | 0.955 | 0.203 |
| hard_slot | 1.000 | 0.203 |
| soft_pool | 0.061 | 0.232 |
| soft_slot | 1.000 | 0.232 |
| soft_bag_pool | 0.084 | 0.909 |
| soft_bag_slot | **1.000** | **0.909** |

Hard write can order under pool readout. Soft write needs cell-aligned readout. Bag is required for dual-role and does not destroy slot order. The hybrid **soft_bag_slot** is the first configuration that solves both foundation tasks.

### Mid-sequence delivery ([`exp04`](fen_lab/exp04_mid_deliver.py))

Interrupted distracted counting: the model must report ID at a mid-sequence query and still solve the final dual-role label.

| Model | Mid-ID | Final dual-role | Pipe |
|-------|--------|-----------------|------|
| residual | weak long-term | poor | high |
| bag, head sees only \(h\) | slow / indirect | good | rises |
| bag, head sees \([h, E]\) | strong | strong | lean |
| bag with every-step reinject of \(E\) into \(h\) | strong mid | degraded final | very high (~35) |

Explicit mid-read of the archive works. Continuously reinjecting escrow into the pipe recreates residual bloat and harms the final dual-role objective.

---

## 7. Operator freeze (synthetic track)

```text
1. Residual propose on pipe h
2. Gate → commit D → write external archive E
3. Deplete: h ← f − D
4. Deliver: head([h, arch]) at query or final time
   (not every-step reinject)

Write
  bag              → dual-role / static facts
  hard (+ slot read) → ordered multi-token outputs
  soft + slot read → ordered outputs (soft alone under pool is insufficient)
  bag + ordered path → when both foundation tasks matter

Read
  pool / concat    → classification and dual-role heads
  slot-aligned     → ordered multi-token outputs
  mid-read of E    → mid-sequence queries
```

---

## 8. Real 1D data

Transfer check on waveform classification with the same skeleton (~75k params):  
[`exp05`](fen_lab/exp05_real_data.py) (MIT-BIH), [`exp05b`](fen_lab/exp05_forda.py) (FordA).

### MIT-BIH (seed 1, 12 epochs)

| Model | best acc |
|-------|---------:|
| residual | 0.863 (flat majority) |
| fen_bag | 0.941 |
| fen_hard_bag | 0.931 |
| fen_roll | **0.963** |
| lstm | 0.935 |

Residual remains stuck at majority-class accuracy. FEN variants and LSTM all exceed that floor; roll is strongest under this budget.

### FordA

| Setting | residual | fen_bag | fen_hard_bag | fen_roll | lstm |
|---------|---------:|--------:|-------------:|---------:|-----:|
| seed 1, 20 ep | 0.519 | 0.590 | 0.600 | **0.803** | 0.540 |
| seed 2, 40 ep | 0.520 | **0.854** | **0.859** | **0.890** | 0.549 |

At 20 epochs, only roll shows a clear climb. At 40 epochs, bag and hard_bag also learn; residual and LSTM remain near chance. Escrow remains necessary; which write is best depends on budget and remains topology-sensitive.

---

## 9. Conclusions

**Established in this track**

1. Residual dual-load fails both foundation tasks (exact recall 0; distracted joint ~0.09).  
2. LSTM fails both foundation tasks under the same budget (exact 0; distracted joint ~0.10).  
3. Topology-matched FEN closes those gaps: bag solves dual-role; hard/slot solves exact order.  
4. Soft ordered write requires cell-aligned readout; bag is required for dual-role.  
5. Archive delivery should be explicit read, not continuous reinject.  
6. On real 1D data, residual fails where FEN learns; write algebra remains task- and budget-dependent.

**Open**

- Multi-seed ranking of bag vs hard vs roll on real 1D  
- Whether roll can solve exact recall under longer or different training (not observed in this lab’s synthetic budgets)  
- Cross-domain claims beyond the tasks run here  

**Practical defaults**

| Setting | Prefer |
|---------|--------|
| Dual-role / static facts | `fen_bag` |
| Exact ordered lists | `fen_slot` / hard write |
| Both foundation tasks | hybrid bag + ordered path |
| Long 1D sensors | `fen_roll` often strongest; bag/hard with sufficient training |

---

## 10. Experiments

Run on Colab (GPU): paste an entire file from [`fen_lab/`](fen_lab/).  
Dependencies: `torch`, `numpy`; add `pandas` for exp05 (see `requirements.txt`).

| Exp | File | Question |
|-----|------|----------|
| 01 | [`exp01_baseline_dual_task.py`](fen_lab/exp01_baseline_dual_task.py) | FEN variants on foundation tasks |
| 01b | [`exp01b_lstm_baseline.py`](fen_lab/exp01b_lstm_baseline.py) | Residual, bag, slot, and LSTM on the same tasks |
| 02 | [`exp02_ode_fen_order_ablation.py`](fen_lab/exp02_ode_fen_order_ablation.py) | Soft-tape order ablations |
| 03 | [`exp03_write_vs_readout.py`](fen_lab/exp03_write_vs_readout.py) | Write × readout grid |
| 04 | [`exp04_mid_deliver.py`](fen_lab/exp04_mid_deliver.py) | Mid-read vs reinject |
| 05 | [`exp05_real_data.py`](fen_lab/exp05_real_data.py) | MIT-BIH |
| 05b | [`exp05_forda.py`](fen_lab/exp05_forda.py) | FordA |

---

## 11. Summary

Feature-Escrow Networks maintain an active residual stream and an external archive: resolved features are gated into escrow and **subtracted** from the pipe, then read when needed—analogous to absorbing nutrients out of the intestinal lumen into the bloodstream.

On synthetic tasks that isolate dual-role retention and exact ordered memory, residual networks and LSTMs remain near chance, while topology-matched FEN writes reach high accuracy. Follow-up experiments fix readout, hybrid bag+order, and delivery rules. Real 1D classification supports the value of external escrow, with the preferred write mode still depending on task structure and training budget.
