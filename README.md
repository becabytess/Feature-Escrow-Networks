# Feature-Escrow Networks (FEN)

**Active science track:** [`fen_lab/`](fen_lab/) — clean-slate experiments, one question per script.  
**This README** documents the full rigorous journey: inspiration, operators, results, and what is actually proven.

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

## 1. Where the idea comes from

### The problem FEN is trying to solve

Most sequential models force **one** hidden state to do two jobs at once:

1. **Active computation** — absorb the current token, update, count, react to noise  
2. **Long-lived memory** — keep facts that must survive many steps of that activity  

When both live in the same tensor, activity tends to **overwrite** memory. Residual connections help keep gradients alive, but on long sequences they also pile unresolved features into the pipe until the stream becomes **bloated** (high norm, hard to train, context drift).

That dual-role conflict is the target: *not* “invent a better LSTM cell,” but **route information so working memory and archive are not the same place**.

### Biological inspiration — the small intestine

The design metaphor is **digestion in the small intestine**, not a literal biophysical model.

| Biology | Role | FEN analogue |
|---------|------|----------------|
| **Lumen (gut tract)** | Active processing of material still being broken down | **Pipe** \(h\) — active residual stream |
| **Absorptive decision** | Is this nutrient ready to leave the tract? | **Gate** \(g\) — what is “resolved” |
| **Absorption into bloodstream** | Move resolved nutrients *out* of the lumen | **Escrow write** — deposit into archive \(E\) |
| **Physical removal from the tract** | Volume leaves the gut; tract stays free for the rest | **Depletion** — \(h \leftarrow f - D\) |
| **Bloodstream → later use** | Nutrients travel insulated from digestive chaos | **Archive read** at query / final time |
| **No continuous dumping back into the gut** | Blood is not re-poured into the lumen every inch | **No every-step reinject** (exp04) |

The important mechanical point is **removal**, not only copying. A copy-only “vault” that never depletes the pipe still leaves the lumen crowded. FEN’s claim is stronger: **resolved features should leave the active stream**.

```text
food in lumen  →  absorb when ready  →  bloodstream holds nutrients
                      ↓
              lumen mass decreases
                      ↓
              remaining material can still be processed
```

Mapped to a step:

```text
h  →  transform  →  f
                 →  gate  →  D = g ⊙ f
                 →  E ← write(E, D)     # bloodstream
                 →  h ← f − D           # gut depleted
                 →  later: head([h, E]) # use archive without living in it
```

That is the entire scientific bet of this lab: **decouple active stream from protected memory, and deplete the active stream when you commit.**

---

## 2. Operators under study

All `fen_lab` models are built from the same skeleton. What changes between variants is **how the escrow is written** and **how / when it is read**.

### Always (canonical skeleton)

1. **Propose** on the pipe (residual-style transform)  
2. **Gate** → commit \(D\)  
3. **Write** \(D\) into an external archive \(E\)  
4. **Deplete** the pipe: \(h \leftarrow f - D\)  
5. **Deliver** archive to the head at the right time (final and/or query), usually via **concat** \([h, \mathrm{arch}]\)

### Write algebras tested

| Write | Intuition | Topology match |
|-------|-----------|----------------|
| **Bag** | Additive / commutative escrow (“set of facts”) | Static IDs, dual-role, unordered context |
| **Hard pointer tape** | Ordered slots, write head advances | Ordered lists / position-sensitive labels |
| **Soft γ-tape** | Continuous address + soft shift | Order *if* readout is cell-aligned |
| **Channel-roll** | Non-commutative update into escrow | Long 1D sensors where structure in the vault helps |

### Read algebras tested

| Read | Intuition |
|------|-----------|
| **Pool / concat** | Classification from summary of \(E\) + final \(h\) |
| **Slot / cell-aligned** | Head aligned to tape cells (order tasks) |
| **Mid-read** | Head sees \([h, E]\) at a query timestep |
| **Gated reinject** | Dump escrow back into \(h\) every step (almost always bad) |

### Controls

| Control | Role |
|---------|------|
| **residual** | Same pipe, **no** escrow — dual load on \(h\) alone |
| **lstm** | Strong classical baseline (real data only) |

---

## 3. Why `fen_lab` (and why `history/`)

The active track is deliberately separate from earlier FEN explorations (now archived under `history/`).

| Principle | Practice |
|-----------|----------|
| One question per script | exp01…exp05 each freeze a claim or kill a bad idea |
| Synthetic first | Prove operators on **distracted** / **recall** toys before real waveforms |
| Parameter-matched | Hidden width chosen so models sit near a shared param budget |
| Honest demotion | Ideas that fail (always-on reinject, soft tape + pool as order fix) are marked demoted, not kept in the story |
| Real data last | exp05 only after the synthetic freeze |

If a result is not produced by a script under `fen_lab/`, it is **out of scope** for this document’s claims.

---

## 4. Experimental journey

### How to run (any exp)

1. Colab → **GPU** runtime  
2. Paste the **entire** file from [`fen_lab/`](fen_lab/) into one cell → Run  
3. Toggle `FAST_MODE` at the top when present (`True` = quick pilot, `False` = multi-seed / fuller)

Deps: **torch + numpy** for exp01–04; **+ pandas** for exp05 (auto-download). See `requirements.txt`.

| Exp | File | Question |
|-----|------|----------|
| **01** | [`fen_lab/exp01_baseline_dual_task.py`](fen_lab/exp01_baseline_dual_task.py) | Dual-task baseline: which FEN write wins **distracted** vs **recall5**? |
| **02** | [`fen_lab/exp02_ode_fen_order_ablation.py`](fen_lab/exp02_ode_fen_order_ablation.py) | Why soft tape fails order — bag escape, sharpness, event shift, or **readout**? |
| **03** | [`fen_lab/exp03_write_vs_readout.py`](fen_lab/exp03_write_vs_readout.py) | Write × readout grid; first hybrid that wins **both** tasks |
| **04** | [`fen_lab/exp04_mid_deliver.py`](fen_lab/exp04_mid_deliver.py) | Is delivery **mid-read of \(E\)** or **every-step reinject**? |
| **05** | [`fen_lab/exp05_real_data.py`](fen_lab/exp05_real_data.py) | Does the freeze hold on **MIT-BIH**? |
| **05b** | [`fen_lab/exp05_forda.py`](fen_lab/exp05_forda.py) | Same models on **FordA** (incl. longer re-run) |

---

### exp01 — Baseline dual-task family

**Tasks (synthetic, \(T=96\), ~15k params)**

- **distracted:** static ID at \(t=0\), then noisy count events → dual-role + depletion stress  
- **recall5:** ordered delayed recall → structure / non-commutativity stress  

**Models:** residual · fen_bag · fen_roll · fen_slot · ode_fen (soft tape + bag)

**Findings (FAST)**

| Task | Winner | Losers / notes |
|------|--------|----------------|
| distracted | **fen_bag** (~99%) | residual fails ID; pure slot fails dual-role |
| recall5 | **fen_slot** climbs | bag / roll / soft under **pool** readout ≈ fail order |

**Interpretation:**  
Depletion + **bag** is the right algebra for dual-role static facts.  
Ordered lists need **ordered write and/or cell-aligned read**, not bag alone under a pooled head.

---

### exp02 — Soft-tape order ablations

**Question:** Soft addressable tape almost solved order in some stories — *why* did it fail under pool readout? Four hypotheses:

| H | Intervention | Result |
|---|--------------|--------|
| H1 | No bag (remove dual-role escape) | **Rejected** as order fix |
| H2 | Entropy-sharpen address | **Rejected** |
| H3 | Event-gated shift | **Rejected** |
| H4 | **Cell-aligned / slot readout** | **Accepted** — recall → perfect |

**Also:** on distracted, bag / full ODE-with-bag stay strong; pure order-only models fail ID.

**Interpretation:** Soft write is not enough; **readout topology** decides order. Soft tape needs a slot head. Bag remains required for dual-role.

---

### exp03 — Write × readout grid

Systematic pairing of write (bag / hard / soft) and read (pool / slot), including hybrids.

| Model | recall5 exact | distracted acc |
|-------|---------------:|---------------:|
| bag_pool | ~0 | ~0.97 |
| hard_pool | **~0.96** | ~0.20 |
| hard_slot | 1.0 | ~0.20 |
| soft_pool | ~0.06 | ~0.23 |
| soft_slot | 1.0 | ~0.23 |
| soft_bag_pool | ~0.08 | ~0.91 |
| **soft_bag_slot** | **1.0** | **~0.91** |

**Findings**

- **Hard write alone** can order under pool (slots already discrete).  
- **Soft write** needs **slot readout** to order.  
- **Bag** is required for dual-role; bag does **not** break slot order.  
- First hybrid that wins **both** tasks: tape (hard or soft) + bag + **task-appropriate head**.

---

### exp04 — Mid-sequence delivery (T4)

**Question:** Classic FEN only reads \(E\) at the end. Can the archive be **used mid-sequence** without re-poisoning the pipe?

**Task:** interrupted distracted counting — query flag at \(t=40\) for mid-ID; final label still dual-role.

| Model | mid_id | final / count | pipe | Verdict |
|-------|--------|---------------|------|---------|
| residual | high early, weak final | poor | fat | dual load fails long-term |
| bag_h_only | eventually mid (slow) | good | rises | cheats by re-stuffing ID into \(h\) |
| **bag_read** | **fast + high** | **excellent** | **lean** | **correct delivery** |
| bag_gated | high mid | **hurts final** | **bloated (~35)** | reinject recreates residual disease |
| soft_bag_read | good | good | lean | same lesson with tape+bag |
| soft_bag_gated | mid ok | hurts final | bloated | reinject killed |

**Interpretation (freeze for delivery)**

```text
T4 = explicit mid/query read of archive  (e.g. head([h, E]))
   ≠ every-step gated reinject into the pipe
```

---

### Synthetic freeze (after exp01–04)

```text
CANONICAL FEN (lab freeze)
1. Residual propose on pipe h
2. Gate → commit D → write external archive
3. Deplete: h ← f − D
4. Deliver: head([h, arch]) at query/final time
   — NOT continuous reinject

WRITE  = match task topology
  bag          → dual-role / static / set facts
  hard (+slot) → ordered multi-token labels
  soft         → only with cell-aligned readout

READ   = pool/concat for classification;
         slot-aligned when labels are ordered lists
```

| Keep | Demote / kill |
|------|----------------|
| Depletion + external escrow | Always-on gated reinject |
| Bag for dual-role | Soft γ-tape as **default** order fix under pool |
| Hard write / slot read for lists | “One write algebra for all tasks” |
| Explicit mid-read of archive | Treating channel-roll as mandatory core (reopened on real data) |

---

### exp05 — Real data (does the freeze survive waveforms?)

Same skeleton, ~**75k** params, CUDA-graph speed path (GPU preload, full batches, no per-step `.item()`).

| Model | Role |
|-------|------|
| residual | no-escrow control |
| fen_bag | deplete + bag + concat |
| fen_hard_bag | hard tape + bag + pool/concat |
| fen_roll | deplete + channel-roll escrow + concat |
| lstm | classical baseline |

**Datasets**

| Dataset | Shape | Role |
|---------|-------|------|
| **MIT-BIH** | \([N, 187, 1]\), 5 classes | short ECG morphology |
| **FordA** | \([N, 500, 1]\), 2 classes | long 1D sensor / late features |

#### MIT-BIH — pilot (seed 1, 12 ep, batch 1000)

| Model | best acc | pipe (approx) | Story |
|-------|---------:|--------------:|-------|
| residual | **0.863** flat | ~14.5 | majority class; not learning morphology |
| fen_bag | **0.941** climbing | ~10.5 | escrow works |
| fen_hard_bag | **0.931** | ~6.1 | good, leaner pipe |
| **fen_roll** | **0.963** | ~8.3 | best under this budget |
| lstm | **0.935** late climb | ~4.1 | competitive, no escrow story |

```text
MIT-BIH:  roll ≫ bag ≥ lstm ≥ hard_bag ≫ residual (majority)
```

#### FordA — pilot (seed 1, 20 ep)

| Model | best acc | Story |
|-------|---------:|-------|
| residual | 0.519 | chance + fat pipe |
| fen_bag | 0.590 | slight help, no solve |
| fen_hard_bag | 0.600 | same tier as bag |
| **fen_roll** | **0.803** | only clear learner |
| lstm | 0.540 | near chance |

At **20 epochs**, bag/hard looked almost useless. That reading was **premature**.

#### FordA — longer re-run (seed 2, 40 ep)

| Model | best acc | best @ | Story |
|-------|---------:|-------:|-------|
| residual | 0.520 | ep6 | still dead |
| fen_bag | **0.854** | **ep40** | late takeoff; still rising at end |
| fen_hard_bag | **0.859** | ep39 | same tier as bag; leanest FEN pipe |
| **fen_roll** | **0.890** | ep34 | highest peak; better sample efficiency |
| lstm | 0.549 | ep19 | still dead |

```text
FordA @ 40 ep:  roll 0.89  ≳  hard ≈ bag ~0.85–0.86  ≫  lstm ≈ residual ~chance
```

**What the longer run changed**

| After pilot (20 ep) | After longer (40 ep) |
|---------------------|----------------------|
| “Only roll works on FordA” | **Too strong** — bag/hard become real learners with time |
| Bag insufficient | **Under-trained**, not architecturally dead |
| Roll sole solution | **Fastest + slightly best**, not exclusive |

**Unchanged**

- Residual never learns FordA (pipe ~14).  
- LSTM in this matched setup never learns FordA.  
- **Escrow + deplete** remains necessary for non-trivial accuracy on these waveforms.

---

## 5. What has been proven (in this lab)

Claims below are **supported by fen_lab runs**. Strength is stated honestly.

### Strong

1. **Dual-state + depletion beats residual dual-load** on dual-role synthetic tasks and on both real 1D sets (residual majority/chance + fat pipe).  
2. **Archive must be readable without living in the pipe** — mid/query **concat read** works; **every-step reinject** bloats pipe and hurts final dual-role (exp04).  
3. **Write/read topology must match the task**  
   - bag ↔ dual-role / set facts  
   - hard write and/or slot readout ↔ ordered lists  
   - soft tape under pool is **not** an order fix (exp02–03).  
4. **On real 1D classification, FEN-style escrow is not a cosmetic residual** — residual fails both MIT-BIH (majority) and FordA (chance) under matched budgets.

### Medium (directionally clear, not multi-seed sealed)

5. **Channel-roll is a strong 1D write** for waveforms under fixed budgets (best on MIT-BIH pilot and both FordA runs).  
6. **Bag is competitive on FordA if trained long enough** (0.59 @20 ep → 0.85 @40 ep, seed 2) — pilot epoch budget was misleading.  
7. **Hard+bag** is a lean middle on real data (not the synthetic “hard solves order under pool” story when the head is a class pool).  
8. **LSTM is task-dependent** — useful late on MIT-BIH; near useless on FordA here. Treat as speed/control baseline, not scientific ceiling.

### Not proven (do not overclaim)

- Universal SOTA across vision / language / all UCR sets  
- Multi-seed statistical ranking of bag vs hard vs roll  
- That roll is the **only** viable write on long 1D (false after 40 ep FordA)  
- That bag alone is always enough on long 1D (needs budget; roll still edged it)  
- Soft tape as a general replacement for hard slots  

---

## 6. Current architecture default

```text
ALWAYS
  residual propose → gate → D → deplete (h ← f − D) → write escrow
  deliver: head([h, arch]) at query/final   # no every-step reinject

WRITE (topology-conditioned)
  bag          — dual-role, static facts, set-like context
  hard pointer — ordered multi-token / list labels (esp. + slot read)
  channel-roll — long 1D sensors / waveforms when sample efficiency matters
                 (best practical 1D default after exp05 pilots)

READ
  pool / concat  — classification
  slot-aligned   — ordered multi-token outputs
  mid-read       — when the task queries archive mid-sequence
```

**Practical 1D classification default after exp05:**  
prefer **`fen_roll`** for speed-to-good-acc; **`fen_bag` / `fen_hard_bag`** remain valid with longer training and for dual-role / lean-pipe stories.

**Practical dual-role / synthetic default:**  
**`fen_bag`** (+ hard/slot when order is the task).

---

## 7. Next direction

The **science claim** (escrow + deplete + correct delivery) is strong enough to freeze as the core.

What is still open is **narrow**: write ranking under a **fair, multi-seed budget**.

### Proposed exp06 — write bake-off (then stop ranking)

| Knob | Setting |
|------|---------|
| Models | `fen_bag`, `fen_hard_bag`, `fen_roll` **only** |
| Datasets | FordA + MIT-BIH (existing loaders) |
| Seeds | `[1, 2, 3]` |
| Epochs | FordA **40**, MIT-BIH **15–20** (fixed) |
| Skip | residual, lstm (already dead / secondary) |

**Decision rule**

```text
if roll mean − bag mean ≥ ~2–3 pts on BOTH datasets:
    default 1D write = roll; bag stays dual-role default
elif gaps small or swap by dataset:
    freeze as a menu, not a single universal write
```

Then:

1. **Freeze a small `fen_core` API** — named write modes `bag | hard | roll`, shared deplete + concat deliver.  
2. **Do not** open a new dataset until that freeze exists — more benchmarks expand the surface without sealing the ranking.  
3. Optional later: one extra domain **after** freeze, as a transfer check — not as another open-ended lab.

No new dataset is required to settle the remaining write question.

---

## 8. File map

```text
fen_lab/                              ← active experiments (paste into Colab)
  exp01_baseline_dual_task.py
  exp02_ode_fen_order_ablation.py
  exp03_write_vs_readout.py
  exp04_mid_deliver.py
  exp05_real_data.py                  ← MIT-BIH (FordA optional)
  exp05_forda.py                      ← FordA-only longer re-run
  README.md                           ← short pointer to this document

history/                              ← archived early work only
  README.md
  experiments/  experiments2/  experiments_h100/  plots/
  legacy_research_report.md
```

---

## 9. One-line summary

**FEN, in this lab, is a dual-stream residual that absorbs resolved features into an external escrow and subtracts them from the active pipe — like the intestine clearing nutrients into the bloodstream — so long active computation does not have to carry every committed fact inside the same tensor.**

Synthetic toys proved **when** bag vs hard vs slot matter and that **delivery is read-not-reinject**.  
Real ECG and FordA proved **escrow is necessary**; **which write wins** is topology- and budget-sensitive, with **roll** currently the best practical 1D default and **bag/hard** still in the game when trained long enough.

---

*Document status: reflects fen_lab through exp05b FordA seed=2 / 40 ep. Update when exp06 (or a freeze) lands.*
