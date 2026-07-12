# Feature-Escrow Networks (FEN)

**Active science track:** [`fen_lab/`](fen_lab/) — clean-slate experiments, one question per script.  
**This README** is the rigorous narrative: why FEN exists, which failures it targets, what we measured, and what we claim.

Earlier explorations live under [`history/`](history/) for reference only. They are **not** part of this path’s evidence base.

```text
Feature-Escrow-Networks/
  README.md                 ← you are here (canonical narrative)
  fen_lab/                  ← run these experiments (Colab one-cells)
  history/                  ← archived early work (not the rigorous path)
  requirements.txt
  LICENSE
```

---

## How to read this document

The order of sections matches the **reasoning journey**, not “latest exp number wins.”

1. **Problem** we claim standard nets have  
2. **Biological idea** and the core FEN operators  
3. **Synthetic probes** built to force that problem into the open  
4. **Foundation results** — residual / LSTM / FEN on those probes (**capability cliffs**, not −10% tables)  
5. **What write/read/delivery actually need** (ablations)  
6. **Real 1D data** (support + limits)  
7. **What is frozen** and what is still open  

**Floor vs lag:** on the hard probes, wrong models often land at **chance / exact ≈ 0**, not “slightly worse.” That distinction is load-bearing. Do not read a final-acc table on ECG the same way as **exact recall** or **distracted ID×count**.

---

## 1. The problem

Most sequential models force **one** hidden state to do two jobs at once:

1. **Active computation** — absorb the current step, update, count, react to noise  
2. **Long-lived memory** — keep facts that must survive many steps of that activity  

When both live in the same tensor, activity tends to **overwrite** memory. Residuals help gradients, but on long sequences they also pile unresolved features into the stream until it becomes **bloated** (high norm, context drift, dual-role collapse).

That is the target claim:

> Working memory and archive should **not** be the same place. Resolved features should **leave** the active path.

This is not “invent a better LSTM cell.” It is a **routing** claim: commit, store outside, deplete the pipe.

---

## 2. Biological inspiration — the small intestine

Metaphor, not biophysics.

| Biology | Role | FEN analogue |
|---------|------|----------------|
| **Lumen (gut tract)** | Active processing of material still being broken down | **Pipe** \(h\) — active residual stream |
| **Absorptive decision** | Is this nutrient ready to leave the tract? | **Gate** \(g\) — what is “resolved” |
| **Absorption into bloodstream** | Move resolved nutrients *out* of the lumen | **Escrow write** — deposit into archive \(E\) |
| **Physical removal from the tract** | Volume leaves the gut | **Depletion** — \(h \leftarrow f - D\) |
| **Bloodstream → later use** | Nutrients travel insulated from digestive chaos | **Archive read** at query / final time |
| **No continuous re-pour into the gut** | Blood is not dumped back every inch | **No every-step reinject** |

Important: **removal**, not only copy. A vault that never depletes the pipe still leaves the lumen crowded.

```text
h  →  transform  →  f
                 →  gate  →  D = g ⊙ f
                 →  E ← write(E, D)     # bloodstream
                 →  h ← f − D           # gut depleted
                 →  later: head([h, E]) # use archive without living in it
```

---

## 3. Core operators (one skeleton)

Everything in `fen_lab` is the same machine with **swappable write/read**:

| Step | Always |
|------|--------|
| 1 | Propose on the pipe (residual-style transform) |
| 2 | Gate → commit \(D\) |
| 3 | Write \(D\) into external archive \(E\) |
| 4 | Deplete: \(h \leftarrow f - D\) |
| 5 | Deliver \(E\) at the right time (final / query), usually \(\mathrm{head}([h, \mathrm{arch}])\) |

| Write mode | Intuition | Built for |
|------------|-----------|-----------|
| **Bag** | Additive, commutative “set of facts” | Static ID / dual-role |
| **Hard pointer / slots** | Ordered cells; pointer advances | Exact ordered lists |
| **Soft γ-tape** | Soft address + shift | Order only with **cell-aligned** read |
| **Channel-roll** | Non-commutative vault update | Long 1D sensors (real data track) |

| Control | Role |
|---------|------|
| **residual** | Same pipe, **no** escrow |
| **lstm** | Classical single-state recurrent baseline |

---

## 4. Why synthetic probes first

If the claim is “single-state nets fail dual-role and ordered archive,” we should not start on ECG accuracy tables. Those can look “close” while the hard mechanisms never fire.

We built two **foundation probes** (\(T=96\), ~15k params, matched width) that force the claimed failure modes into the open:

### 4.1 Distracted counting (dual-role / overwrite)

- **t = 0:** a static **ID** pulse  
- **later:** noisy **+ / −** count events + distractors  
- **label:** joint **ID × count bin** (30-way)

**What it tests:** can the model keep a committed fact while the pipe stays busy?  
**Success:** high **joint acc** and high **id_acc** (count alone is not enough).  
**Failure floor:** joint acc ~ chance / ~0.1, ID ~ chance, even if count looks fine.

This is the probe closest to the **original FEN idea**.

### 4.2 recall5 (exact ordered memory)

- Five symbols appear early (with distractors later)  
- Model must emit the **full ordered sequence**  
- **Primary metric: exact** = all five correct (not token accuracy alone)

**What it tests:** can the archive preserve **order / non-commutativity**?  
**Success:** exact → high (→ 1.0).  
**Failure floor:** exact **≈ 0** (token may still climb a little — that is not success).

### 4.3 How to score these tables

| Language to avoid | What we mean instead |
|-------------------|----------------------|
| “LSTM competes” | On **which** task? Foundation probes ≠ real classification |
| “loses by 5%” | Often **exact 0 vs 0.96** or **acc 0.10 vs 0.99** |
| Winner-only rows | Always show **floors** for residual / LSTM / wrong write |

---

## 5. Foundation results — do the claimed failures exist?

Same generators, ~15k params, seed 1. Controls and matching FEN modes on **both** probes  
([`fen_lab/exp01_baseline_dual_task.py`](fen_lab/exp01_baseline_dual_task.py), [`fen_lab/exp01b_lstm_baseline.py`](fen_lab/exp01b_lstm_baseline.py)).

### 5.1 recall5 — exact ordered sequence

| Model | exact | token | Verdict |
|-------|------:|------:|---------|
| residual_rnn | **0.000** | 0.099 | Floor. Fat pipe. |
| **lstm** | **0.000** | 0.098 | **Floor. Flat zero exact.** |
| fen_bag | **0.001** | ~0.19–0.34 | Token can rise; **exact still floor** (commutative bag). |
| fen_roll | **~0.00** | ~0.3 | Did **not** unlock exact-5 in this budget. |
| **fen_slot** | **0.96 → 1.0** | ~0.99–1.0 | **Only family that nails order** here. |

```text
exact:  fen_slot ~1   |   residual / lstm / bag / roll ~ 0
```

LSTM is not “slightly behind slot.” It is in the **same failure class as residual** on exact recall.

### 5.2 Distracted counting — dual-role

| Model | acc | id | count | Verdict |
|-------|----:|---:|------:|---------|
| residual_rnn | **0.088** | 0.114 | 0.768 | Floor joint label; ID dies; pipe ~8.5. |
| **lstm** | **0.103** | 0.105 | 0.973 | **Same pattern:** count can look fine; **ID / joint floor.** |
| **fen_bag** | **0.988** | **0.996** | **0.992** | **Nails dual-role.** Lean pipe. |
| fen_roll | **~0.82–0.94** | high | high | Learns dual-role (weaker/slower than bag). |
| fen_slot | **0.185** | 0.185 | **~1.0** | Count OK; **ID destroyed** (wrong topology for dual-role). |

```text
joint acc:  fen_bag ~0.99  |  lstm ~0.10  |  residual ~0.09  |  slot ~0.19
```

Again: **capability cliff**, not a leaderboard nudge.

### 5.3 Capability matrix (the foundation claim)

| Model | recall5 exact | distracted joint | Role in the story |
|-------|:-------------:|:----------------:|-------------------|
| residual | ✗ floor | ✗ floor | Dual-load without escrow fails |
| **lstm** | ✗ **floor** | ✗ **floor** | Classical single-state baseline fails **both** hard probes |
| fen_bag | ✗ floor | ✓ **~1** | Correct algebra for **dual-role** |
| fen_slot | ✓ **~1** | ✗ floor ID | Correct algebra for **order** |
| one write for all tasks | — | — | **False** — topology must match |

```text
LSTM solves neither foundation probe.
fen_bag  solves dual-role, not exact order.
fen_slot solves exact order, not dual-role.
```

**This is the scientific start of the project:** the failure modes we named are real under controls people respect (residual + LSTM), and FEN modes that match the topology **close the gap to ceiling**, not by a few points.

*(LSTM was always part of this foundation comparison in spirit; the explicit matched LSTM run is `exp01b`. Treat it as part of the opening evidence, not a late side quest.)*

---

## 6. Digging into write / read (why not one FEN forever)

Once floors are established, the next question is **which operators** buy which capability.

### 6.1 Soft tape order ablations ([`exp02`](fen_lab/exp02_ode_fen_order_ablation.py))

Soft addressable tape looked like a “general ordered escrow.” Under a **pooled** head it stayed near floor on exact. Hypotheses:

| H | Idea | Result on recall5 exact |
|---|------|-------------------------|
| H1 | Remove bag escape | **No** — still fails order |
| H2 | Sharpen address entropy | **No** |
| H3 | Event-gated shift | **No** |
| H4 | **Cell-aligned / slot readout** | **Yes → exact 1.0** |

On distracted: bag / ODE-with-bag stay high; pure order models without bag stay on the **ID floor**.

**Lesson:** soft write is not enough; **readout topology** decides order. Bag remains required for dual-role.

### 6.2 Write × readout grid ([`exp03`](fen_lab/exp03_write_vs_readout.py))

Full numbers (seed 1) — winners **and** floors:

| Model | recall5 exact | distracted acc | Story |
|-------|---------------:|---------------:|-------|
| bag_pool | **0.003** | **0.973** | Dual-role only |
| hard_pool | **0.955** | **0.203** | Order only (hard write enough under pool) |
| hard_slot | **1.000** | **0.203** | Order only |
| soft_pool | **0.061** | **0.232** | Floor both |
| soft_slot | **1.000** | **0.232** | Order only |
| soft_bag_pool | **0.084** | **0.909** | Dual-role only (bag escape) |
| **soft_bag_slot** | **1.000** | **0.909** | **Both** |

**Lessons:**

- Hard write alone can order; soft write needs **slot readout**.  
- Bag is required for dual-role; bag does **not** break slot order.  
- First hybrid that wins **both** foundation probes: **ordered path + bag + task-appropriate head**.

### 6.3 Mid-sequence delivery ([`exp04`](fen_lab/exp04_mid_deliver.py))

Classic FEN often only reads \(E\) at the end. Can the archive be used **mid-sequence** without re-poisoning the pipe?

Interrupted distracted task: query flag mid-run for mid-ID; final label still dual-role.

| Model | Mid-ID | Final dual-role | Pipe | Verdict |
|-------|--------|-----------------|------|---------|
| residual | early OK, then weak | poor | fat | dual load fails |
| bag_h_only | slow / cheats via pipe | good | rises | ID re-stuffed into \(h\) |
| **bag_read** | **fast + high** | **excellent** | **lean** | **correct: read \(E\)** |
| bag_gated (reinject every step) | high mid | **hurts final** | **~35** | reinject = residual disease |
| soft_bag_read | good | good | lean | same lesson |
| soft_bag_gated | mid ok | hurts final | bloated | reinject killed |

```text
Deliver = query/final read of archive
        ≠ every-step dump of E back into h
```

---

## 7. Synthetic freeze (operators)

After the foundation + ablations:

```text
CANONICAL FEN
1. Residual propose on pipe h
2. Gate → commit D → write external archive
3. Deplete: h ← f − D
4. Deliver: head([h, arch]) at query/final
   — NOT continuous reinject

WRITE  (match task topology)
  bag           → dual-role / static / set facts
  hard (+ slot read) → ordered multi-token labels
  soft          → only with cell-aligned readout
  bag + ordered path → when BOTH foundation probes matter

READ
  pool/concat   → classification / dual-role head
  slot-aligned  → ordered multi-token outputs
  mid-read      → when the task queries E mid-sequence
```

| Keep | Kill / demote |
|------|----------------|
| Depletion + external escrow | Always-on gated reinject |
| Bag for dual-role | Soft γ-tape as default order fix under pool |
| Hard write / slot read for lists | “One write algebra for every task” |
| Explicit mid-read of archive | Treating residual/LSTM as strong on the **foundation** probes |

---

## 8. Real 1D data — does escrow still matter?

Different regime: real classification can look “close” on final acc even when foundation probes showed floors. Still useful as a **transfer** check, not as a replacement for §5.

Same skeleton, ~**75k** params ([`exp05`](fen_lab/exp05_real_data.py), [`exp05b`](fen_lab/exp05_forda.py)).

### 8.1 MIT-BIH (seed 1, 12 ep)

| Model | best acc | Story |
|-------|---------:|-------|
| residual | **0.863** flat | Majority class; not learning morphology |
| fen_bag | **0.941** | Escrow works |
| fen_hard_bag | **0.931** | Good; leaner pipe |
| **fen_roll** | **0.963** | Best under this budget |
| lstm | **0.935** | Late climb — **can look competitive on this easier regime** |

```text
MIT-BIH final acc is not the foundation story.
LSTM can approach FEN here; it could not on exact-5 / distracted dual-role.
```

### 8.2 FordA

**Pilot (seed 1, 20 ep):** roll **0.80**; bag/hard ~0.59–0.60; residual/lstm ~chance.  
**Longer (seed 2, 40 ep):**

| Model | best acc | Story |
|-------|---------:|-------|
| residual | 0.520 | still dead |
| fen_bag | **0.854** | late takeoff (under-trained at 20 ep) |
| fen_hard_bag | **0.859** | same tier as bag; lean pipe |
| **fen_roll** | **0.890** | highest peak |
| lstm | 0.549 | still dead |

FordA is **noisy / seed-sensitive** for timing of takeoff; use it as a long-1D support check, not as the sole ranking oracle.

**Unchanged across both real sets:** residual dual-load fails (majority or chance + fat pipe). Escrow is not cosmetic.

---

## 9. What has been proven (honest strength)

### Strong (foundation + operators)

1. **Residual dual-load fails** the foundation probes (joint distracted floor; exact recall 0).  
2. **LSTM fails both foundation probes** under matched budget (exact **0.000**; distracted joint **~0.10**).  
3. **Matching FEN write closes the cliff:** bag → dual-role ~1; slots → exact order ~1.  
4. **Wrong topology is a floor, not a lag** (bag on exact; slot on distracted ID; soft+pool on exact).  
5. **Delivery is read, not reinject** (exp04).  
6. On real 1D, residual stays majority/chance where FEN learns — escrow still matters outside pure toys.

### Medium

7. Channel-roll is a strong **waveform** write under fixed budgets; bag/hard catch up on FordA with more epochs.  
8. Hybrid bag+ordered path can hold **both** synthetic probes.  
9. On MIT-BIH classification, LSTM can look late-competitive — **do not** import that language into foundation claims.

### Not proven

- Universal SOTA across domains  
- Multi-seed sealed ranking of bag vs roll on all UCR sets  
- Roll as a proven exact-5 solver **in this lab’s budgets** (stayed ~0 exact here)  
- Soft tape as a general replacement for hard slots  

---

## 10. Practical defaults after this path

```text
ALWAYS
  propose → gate → D → deplete → write E
  deliver head([h, arch]) at query/final   # no every-step reinject

BY TASK
  dual-role / set facts     → fen_bag
  exact ordered lists       → fen_slot / hard write (+ slot read if soft)
  both foundation probes    → hybrid bag + ordered path
  long 1D sensors (real)    → fen_roll often best; bag/hard valid with budget
```

**Do not say:** “LSTM competes with FEN.”  
**Say:** “LSTM fails the foundation dual-role and exact-order probes; on some real classification tasks final acc can look closer — that is a different regime.”

---

## 11. Experiment index (how to run)

1. Colab → **GPU**  
2. Paste entire file from [`fen_lab/`](fen_lab/) → Run  
3. Toggle `FAST_MODE` when present  

Deps: torch + numpy (01–04, 01b); + pandas for 05 (auto-download).

| Exp | File | Role in the journey |
|-----|------|---------------------|
| **01** | [`exp01_baseline_dual_task.py`](fen_lab/exp01_baseline_dual_task.py) | FEN family on foundation probes |
| **01b** | [`exp01b_lstm_baseline.py`](fen_lab/exp01b_lstm_baseline.py) | **LSTM + residual + bag + slot** on the same probes |
| **02** | [`exp02_ode_fen_order_ablation.py`](fen_lab/exp02_ode_fen_order_ablation.py) | Soft-tape order: readout is the fix |
| **03** | [`exp03_write_vs_readout.py`](fen_lab/exp03_write_vs_readout.py) | Write × read grid; hybrid both tasks |
| **04** | [`exp04_mid_deliver.py`](fen_lab/exp04_mid_deliver.py) | Mid-read vs reinject |
| **05** | [`exp05_real_data.py`](fen_lab/exp05_real_data.py) | MIT-BIH (real transfer) |
| **05b** | [`exp05_forda.py`](fen_lab/exp05_forda.py) | FordA (long 1D transfer) |

### Next (optional)

Multi-seed write bake-off on real 1D (`bag` / `hard` / `roll` only), then freeze a small `fen_core` API.  
No new dataset is required to keep the **foundation** claim honest — that claim already rests on §5.

---

## 12. File map

```text
fen_lab/
  exp01_baseline_dual_task.py
  exp01b_lstm_baseline.py         ← foundation controls incl. LSTM
  exp02_ode_fen_order_ablation.py
  exp03_write_vs_readout.py
  exp04_mid_deliver.py
  exp05_real_data.py
  exp05_forda.py
  README.md                       ← short pointer here

history/                          ← archived early work only
```

---

## 13. One-line summary

**FEN absorbs resolved features into an external escrow and subtracts them from the active pipe — like the intestine clearing nutrients into the bloodstream — so long active computation does not have to carry every committed fact in one state.**

On the synthetic probes built for that claim, **residual and LSTM sit at the floor**; **bag FEN** solves dual-role distraction; **slot FEN** solves exact ordered recall; hybrids and correct delivery rules extend the operator set. Real 1D data supports that escrow matters, with write algebra still topology- and budget-sensitive.

---

*Document status: fen_lab through foundation LSTM comparison (01b), synthetic freeze (01–04), real pilots (05 / 05b). Narrative ordered by reasoning journey, not by “latest exp number.”*
