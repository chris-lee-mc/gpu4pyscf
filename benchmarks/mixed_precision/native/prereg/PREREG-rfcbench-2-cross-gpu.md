# PREREG rfcbench-2: commissioning, and cross-GPU do-no-harm

**Status 2026-10-02: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`.
Freezes at the commissioning sentinel (C0). Changes go only in dated amendments above RESULT,
written before the dispatch they govern.

## C0, the commissioning pod (`mode=commission`, `pro6000`)

It is the first paid run of every code path, and it provides the PRO 6000 row used by this scope
and by scope 5. Its cells:
- the trio (paracetamol, propranolol, celecoxib) × {r2SCAN, B3LYP, wB97M-V} × arms
  {`mixed`, `cache`, `stock`}, R = 3, at mTZVPP;
- one downstream cell, paracetamol r2SCAN (`PREREG-rfcbench-4` procedure);
- one basis cell, paracetamol r2SCAN at def2-SVP / def2-universal-jkfit (`PREREG-rfcbench-3`
  procedure).

Est. work ~2600 s. Two attempts are budgeted.

## X1–X4: other cards (`mode=cross`)

C0's trio cells and arms, R = 3, on one card each. Every cascade is a single model, with no
fallback:

| class | model |
|---|---|
| `5090` | NVIDIA GeForce RTX 5090 |
| `l40s` | NVIDIA L40S |
| `h100` | NVIDIA H100 80GB HBM3 |
| `a100` | NVIDIA A100-SXM4-80GB |

Try order: 5090, L40S, H100, A100. The capacity rule of PREREG-0 §6 applies. The molecules beyond
the trio are never run on these cards.

## Readings, per (card, xc), on the trio geomean (disclosed, never gated)

- **Treatment**, mixed/stock: PAYS / NEUTRAL / HARMS.
- **Precision effect**, mixed/cache: PAYS / NEUTRAL / HARMS. This is the do-no-harm reading for the
  precision switch itself, with the cache's own gain removed.
- **Cache effect**, cache/stock: disclosed only.

## Predictions (MODEL-MISS band ±0.15 absolute)

| card | mixed/stock r2SCAN / B3LYP / wB97M-V | mixed/cache r2SCAN / B3LYP / wB97M-V |
|---|---|---|
| PRO 6000 (C0) | 1.80 / 2.05 / 2.65 | 1.55 / 1.80 / 2.45 |
| 5090 | 1.75 / 2.00 / 2.60 | 1.50 / 1.75 / 2.40 |
| L40S | 1.60 / 1.80 / 2.40 | 1.40 / 1.60 / 2.20 |
| H100 | 1.25 / 1.20 / 1.35 | 1.05 / 1.05 / 1.20 |
| A100 | 1.10 / 1.10 / 1.25 | 0.95 / 1.00 / 1.15 |

**Falsifiers of the RFC's framing:**
- the precision effect PAYS on the A100;
- the precision effect HARMS on the 5090 or L40S.

## Budget

Five pods: C0 and X1–X4. Capacity misses cost nothing; they are not pods.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## Amendments

None yet.

## Amendment 1 (2026-10-02, before any dispatch)

- **C0's extra cells.** C0's downstream cell (paracetamol r2SCAN) and basis cell (paracetamol
  r2SCAN, def2-SVP / def2-universal-jkfit) run arms `mixed` and `stock` only. Only the trio cells
  carry the `cache` arm.
- **X2 split.** The L40S has no banked data, so X2 is estimated with a guessed factor of 2.0. It
  may be split after the C0 calibration (PREREG-0 Amendment 1).

## RESULT

### RESULT, part 1 (2026-10-02): C0 **PASS**

**The run.** Run `36951733967` attempt 1, sentinel
`ci-results/runs/rfcbench-0ee605b64285-36951733967-1-0ee605b6.txt`, line 2
`RUN_ID=36951733967-1-0ee605b6` (checked). Repo `0ee605b6`, fork `63af0568`, on
`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`.

**Status and gates.**
- `STATUS=PASS`, `WORK_RC=0`, no problems.
- All 14 lock pins and `cutensor-cu12==2.3.1` verified; cuTENSOR observed in every group.
- 11/11 cells OK and TREATED. Every arm converged in identical cycle counts.
- No NOISY cell, so the pod is not DEGRADED.
- The canary (paracetamol r2SCAN stock) read 3.58 s. That is below the 3.8–5.5 s band, i.e. a fast
  pod, not a degraded one.

**Readings (trio geomeans, def2-mTZVPP, R = 3 warm pairs, cuTENSOR):**

| | mixed/stock | prediction | | mixed/cache (precision effect) | cache/stock (cache effect) |
|---|---|---|---|---|---|
| r2SCAN | **1.607** PAYS | 1.80 (band 1.53–2.07) | in band | 1.543 PAYS | 1.041 NEUTRAL |
| B3LYP | **1.688** PAYS | 2.05 (band 1.74–2.36) | **MODEL-MISS** | 1.633 PAYS | 1.034 NEUTRAL |
| wB97M-V | **2.059** PAYS | 2.65 (band 2.25–3.05) | **MODEL-MISS** | 2.051 PAYS | 1.004 NEUTRAL |

- Per-cell mixed/stock:
  - r2SCAN 1.605 / 1.578 / 1.638;
  - B3LYP 1.787 / 1.654 / 1.626;
  - wB97M-V 2.097 / 2.131 / 1.954;
  - paracetamol r2SCAN at def2-SVP 1.729 (single cell).
- Per-arm spread ≤ 0.9 % everywhere.

**What the misses mean.** The predictions were built from the g4pport single-warm-pair trio
(1.81 / 1.97 / 2.69) on another pod. Within one pod, the cache's own gain is only 0.4–4 %
(cache/stock). The precision effect alone is 1.54 / 1.63 / 2.05.

**Downstream cell** (paracetamol r2SCAN; PREREG-4 procedure). It passes every ceiling:
- max |g_M − g_R| = 1.37e-6 Ha/Bohr, against the 1e-5 ceiling;
- max |μ_M − μ_R| = 4.99e-5 D, against 1e-4;
- |E_M − E_R| = 1.3e-10 Ha.

Its readings:
- **ρ_g = 4.09 and ρ_μ = 8.55, both above 3: "mixed-specific residual"**. That is a PASS under the
  ceilings, and it is disclosed.
- The S' noise comparator gives max |g_S' − g_R| = 3.35e-7 and max |μ_S' − μ_R| = 5.83e-6 D.
- The ≤ 1e-6 gradient prediction is a MODEL-MISS, at 1.37e-6.

### RESULT, part 2 (2026-10-02): C0w **PASS**, the wB97M-V trio re-run on a healthy pod

**Why it ran.** Under PREREG-0 Amendment 3, C0 is CONTENTION-UNKNOWN: it predates the telemetry and
shows the contended signature. C0w re-ran C0's three wB97M-V trio cells with all three arms.

**The run.** Run `37044790110` attempt 1, sentinel
`ci-results/runs/rfcbench-3890e545d9cc-37044790110-1-3890e545.txt`, line 2
`RUN_ID=37044790110-1-3890e545` (checked). On `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`,
idle 25.5 W, so healthy. `STATUS=PASS`, `WORK_RC=0`, 3/3 cells OK and TREATED, no NOISY cell.

| wB97M-V, R = 3 | mixed/stock | mixed/cache (precision effect) | cache/stock (cache effect) |
|---|---|---|---|
| paracetamol | 3.019 | 2.981 | 1.013 |
| propranolol | 3.126 | 3.105 | 1.007 |
| celecoxib | 2.756 | 2.748 | 1.003 |
| **trio geomean** | **2.963** PAYS | **2.941** PAYS | **1.008** NEUTRAL |
| C0, for comparison (unknown contention) | 2.059 | 2.051 | 1.004 |
| prediction (±0.15) | 2.65: **MODEL-MISS, above** | 2.45: **MODEL-MISS, above** | — |

**What changes.**
- The PRO 6000 wB97M-V row for this scope is now C0w's. C0's wB97M-V numbers are reported
  separately, labelled CONTENTION-UNKNOWN.
- C0's r2SCAN and B3LYP rows (1.607 / 1.688) are also CONTENTION-UNKNOWN. Against L12's healthy
  trio cells, from another pod and without a cache arm:
  - r2SCAN: 1.638 there, so C0 is within 2 %;
  - B3LYP: 1.859 there, so C0 is **10 % lower**.

  C0's B3LYP row therefore looks affected too. The healthy PRO 6000 r2SCAN and B3LYP rows are taken
  from L12's trio cells, and C0's rows are kept apart. The PRO 6000 precision-effect (mixed/cache)
  reading exists for wB97M-V only (C0w). For r2SCAN and B3LYP it exists only on C0. No further
  re-run is pre-registered; that is the owner's call.
- On a healthy card, the precision effect for wB97M-V is about 2.94×.
- The cache effect is under 1.5 % on both pods.


### RESULT, part 3 (2026-10-06): the other cards. L40S **PAYS**, H100 **HARMS**, A100 DEGRADED twice (clean cells only), 5090 **n/a-by-capacity**

All pods ran the fork at `63af0568`, on repo `3890e545`, with C0's trio cells and three arms at
R = 3. Every sentinel's line 2 matches its run. No pod is CONTENDED: idle power was 73–114 W.

**L40S (X2a + X2b): PASS.**
- Runs `37065229751` (paracetamol, propranolol) and `37067826972` (celecoxib) on `NVIDIA L40S`;
  9/9 cells are OK and TREATED.
- One cell of X2a is NOISY: propranolol B3LYP, cache arm, 23 % spread. That is 1 of 6 cells, so the
  pod is not DEGRADED.

| L40S trio geomean | mixed/stock | prediction ±0.15 | mixed/cache (precision) | prediction ±0.15 | cache/stock |
|---|---|---|---|---|---|
| r2SCAN | **1.729** PAYS | 1.60, in band | **1.655** PAYS | 1.40, **MODEL-MISS** (above) | 1.045 |
| B3LYP | **1.524** PAYS | 1.80, **MODEL-MISS** (below) | **1.463** PAYS | 1.60, in band | 1.041 |
| wB97M-V | **2.994** PAYS | 2.40, **MODEL-MISS** (above) | **2.981** PAYS | 2.20, **MODEL-MISS** (above) | 1.004 |

Per cell (paracetamol / propranolol / celecoxib), mixed/stock:
- r2SCAN 1.766 / 1.688 / 1.735;
- B3LYP 1.489 / 1.606 / 1.479;
- wB97M-V 3.023 / 3.205 / 2.772.

**H100 (X3): PASS.**
- Run `37070570095` on `NVIDIA H100 80GB HBM3`; 9/9 cells are OK and TREATED, and none is NOISY.

| H100 trio geomean | mixed/stock | prediction ±0.15 | mixed/cache (precision) | prediction ±0.15 | cache/stock |
|---|---|---|---|---|---|
| r2SCAN | **1.320** PAYS | 1.25, in band | **1.084** NEUTRAL | 1.05, in band | **1.217** PAYS |
| B3LYP | **0.905** HARMS | 1.20, **MODEL-MISS** (below) | **0.745** HARMS | 1.05, **MODEL-MISS** (below) | **1.215** PAYS |
| wB97M-V | **0.577** HARMS | 1.35, **MODEL-MISS** (below) | **0.539** HARMS | 1.20, **MODEL-MISS** (below) | 1.070 |

Per cell, the precision effect is:
- r2SCAN 1.109 / 1.081 / 1.063;
- B3LYP 0.879 / 0.729 / 0.646;
- wB97M-V 0.611 / 0.520 / 0.494.

It falls with molecule size for every functional.

**A100 (X4, X4r): DEGRADED twice, so its trio geomeans are not quotable.**
- X4 is run `37071560009`: 2 of 9 cells NOISY (22 %). Under PREREG-0 §3 that makes the pod DEGRADED,
  which I missed when I first read it.
- X4r is run `37391778846`, the owner-approved re-dispatch of the same SHA `3890e545`, launched from
  a holding branch: 4 of 9 cells NOISY (44 %).
- Both ran on `NVIDIA A100-SXM4-80GB`, and both PASS their gates with 9/9 cells TREATED.
- Two consecutive degraded repeats of the same pod trigger PREREG-0 §6 ("accept the clean data that
  exists"). So **only the non-NOISY cells are reported** below, and no trio geomean is stated as a
  reading. The NOISY cells' numbers stay in `x4.csv` and `x4r.csv`.

| A100 clean cells | X4 mixed/cache | X4r mixed/cache | X4 mixed/stock | X4r mixed/stock |
|---|---|---|---|---|
| paracetamol r2SCAN | (NOISY) | 1.184 PAYS | (NOISY) | 1.390 |
| propranolol r2SCAN | 1.025 NEUTRAL | 0.987 NEUTRAL | 1.276 | 1.179 |
| celecoxib r2SCAN | 0.991 NEUTRAL | (NOISY) | 1.143 | (NOISY) |
| paracetamol B3LYP | 0.901 HARMS | (NOISY) | 1.091 | (NOISY) |
| propranolol B3LYP | 0.689 HARMS | (NOISY) | 0.817 | (NOISY) |
| celecoxib B3LYP | 0.611 HARMS | (NOISY) | 0.711 | (NOISY) |
| paracetamol wB97M-V | 0.531 HARMS | 0.566 HARMS | 0.562 | 0.596 |
| propranolol wB97M-V | 0.472 HARMS | 0.492 HARMS | 0.490 | 0.509 |
| celecoxib wB97M-V | (NOISY) | 0.470 HARMS | (NOISY) | 0.488 |

- Where both pods have a clean reading, they agree in direction.
- Every clean B3LYP and wB97M-V cell reads HARMS on the precision effect.
- The cache alone is worth 1.15–1.25 for r2SCAN and B3LYP on this card, the same as on the H100.

**5090 (X1): n/a-by-capacity.**
- Two attempts were refused for capacity, with no pod created: run `37064884792` at 21:07 UTC and run
  `37073071979` at 22:33 UTC, both on 2026-10-02.
- PREREG-0 §6 allows up to three attempts over two days. **The third attempt was not made inside that
  window.** That is a procedural lapse, disclosed here.
- The 5090 row is therefore n/a-by-capacity on two attempts, not three. An attempt now would be
  outside the pre-registered window, and is the owner's call.

**Falsifiers of the RFC's framing.**
- **"The precision effect PAYS on the A100": not met.**
  - No trio geomean is quotable.
  - Eleven of the twelve clean cells read NEUTRAL or HARMS.
  - The one exception is X4r's paracetamol r2SCAN at 1.184 PAYS. The same cell on X4 is NOISY, so it
    has no clean second reading.
- **"The precision effect HARMS on the 5090 or L40S": not met on the L40S**, which PAYS for all three
  functionals. The 5090 is n/a.

**What this means for the RFC.**
- On cards whose FP64 rate is a small fraction of FP32 (RTX PRO 6000, L40S), the precision switch
  pays for every functional: 1.46–2.98 on the L40S.
- On FP64-strong cards (H100, A100), it is NEUTRAL for r2SCAN and HARMS for B3LYP and wB97M-V:
  0.745 and 0.539 on the H100 trio geomeans, and down to 0.47 in single cells. The treated arm's apparent r2SCAN gain there (1.32 on the H100) is the cache's, not the
  precision switch's.
- So "do no harm" requires the precision switch to be **off by default on FP64-strong cards**, or
  gated on the device's FP64:FP32 throughput. The FP64 AO cache can stay on independently: it pays
  1.15–1.25 on the H100 and on the clean A100 cells for r2SCAN and B3LYP.
- This is a reading of these pods, not a tested gating rule.


### Erratum (2026-10-06): found by the evidence-package review (`benchmarks/mixed_precision/native/` in the fork)

The text above is left as written. These corrections supersede it where they differ; `verify_native.py` checks each corrected number against the data.

- **Part 2:** L12's healthy trio B3LYP is **1.860** (1.8595), not 1.859.
- **Part 3, undisclosed flags.** The paracetamol B3LYP cell has warm walls under 1 s (`under_1s`) on the H100 (X3: medians 0.60–0.73 s) and on both A100 pods. On the H100 that cell feeds the quoted B3LYP precision effect, 0.745.
- **Part 3, A100 wording.** "Eleven of the twelve clean cells" means 12 clean *readings* over 9 distinct cells: three cells are clean on both pods. Applying PREREG-0 §6's "two infra-starved repeats" rule to two DEGRADED pods extends that rule; it was not written for that case.
- **Part 3, the cache alone.** "1.15–1.25 on the H100 and on the clean A100 cells" holds for r2SCAN and B3LYP only. For wB97M-V the cache alone is NEUTRAL: 1.070 on the H100, 1.03–1.06 on the A100.
- **Part 3, "gated on the device's FP64:FP32 throughput".** The data has only two FP64:FP32 levels, and those two groups also differ in architecture and bandwidth. It is a hypothesis, not a supported criterion; the mechanism and the threshold are unknown.
