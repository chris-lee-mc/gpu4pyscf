# What the RFC may claim from the native-port campaign

Every number below is recomputed from `data/` by `python verify_native.py` (stdlib only; exit 0
means every check reproduced). The `[N…]` tags are its check groups. The protocols are in
`prereg/`, and the pods are listed in `SOURCES.md`. Before quoting anything, read **What can and
cannot be verified** and the **ledger** at the end.

## The configuration that was measured, which is not the library default

| setting | measured | library default |
|---|---|---|
| FP64 AO cache | `ao_cache_fp64=True` in every `mixed` arm | `False` |
| B3LYP XC switch threshold | `xc_switch_tol=3e-4` | `1e-3` |
| grid | level 3, **pruning off** (`grids.prune = None`) | level 3, `nwchem_prune` |
| tensor contractions | cuTENSOR 2.3.1, required and gated on every pod | optional |
| convergence | `conv_tol 1e-9`, the library-default `conv_tol_grad` | same |
| basis | def2-mTZVPP / def2-tzvpp-jkfit unless stated | — |

Unpruned grids make XC a larger share of the run, and XC is the part the FP32 switch speeds up. They
also make the AO cache larger, which moves the MIG fit limits. **No number here measures the
defaults, nor the README's example `MixedPrecision(xc=True, k=True)`.** Expect smaller gains on
pruned grids. How much smaller is not measured.

**Method.**
- R = 3 warm pairs per cell, plus one cold pair that is reported but excluded.
- Within each pair the arms run in the order mixed, cache, stock, so stock runs last with a warm
  memory pool.
- A ratio is stock median / mixed median: above 1 means mixed is faster.
- Readings: PAYS ≥ 1.15; NEUTRAL in between; HARMS ≤ 0.95.
- Within a pod, the spread of the R = 3 walls is mostly ≤ 3 %.
- **Between pods of the same card class, the same reading varies by 10–20 % or more.** For example,
  the PRO 6000 B3LYP trio is 1.688 on C0, 1.860 on L12 and 2.019 on the Server Edition (M0). Treat
  the third decimal as reproducibility of the data, not of the hardware.

## C1. RTX PRO 6000 (Blackwell, FP64-limited): the mode pays for every functional tested `[N1]`

| 24 drug-like molecules, 20–44 atoms | mixed/stock geomean | S / M / L tiers | per-cell range | pods |
|---|---|---|---|---|
| r2SCAN | **1.657** | 1.576 / 1.665 / 1.804 | 1.523–1.871 | 1 (L12) |
| B3LYP | **1.783** | 1.736 / 1.779 / 1.897 | 1.584–1.994 | 1 (L12) |
| wB97M-V (VV10 component) | **2.914** | 2.848 / 2.995 / 2.806 | 2.593–3.250 | 5 healthy (W1–W3, W4r, W5) |

- All three are inside their pre-registered ±15 % bands. Bands that wide make "in band" a weak
  test.
- The gain rises from S to L. The "flat across tiers" prediction failed.
- **Two pods, C0 and W4, read about 30 % lower for wB97M-V** and are excluded from the 2.914:
  - C0 reads 2.059 on the trio;
  - W4 reads 1.92–2.13, its GPU drawing 576 W at idle;
  - C0w (the C0 cells re-run) reads 2.963, and W4r (the W4 cells re-run) reads 2.87–3.05.

  The exclusion rule (PREREG-0 Amendment 3: idle draw above 200 W) **was written after W4 and C0
  had been seen**, and it has replaced only low readings so far. On shared or cloud GPUs, expect
  readings like C0's and W4's.
- At `conv_tol_grad = 1e-5` (R = 1), the "bridge" leg reads 1.586. That is a MODEL-MISS against the
  prototype's 1.78–1.84.

## C2. Larger molecules and bases (PRO 6000 Workstation, single pods) `[N3]`

**Six molecules of 63–98 atoms:**
- **r2SCAN 1.931.** One molecule carries much of it: ritonavir at 2.729, whose *stock* wall
  (72.2 s) is anomalously above lopinavir's (50.9 s). Without ritonavir the geomean is 1.802. The
  stock behaviour on ritonavir was not investigated.
- **B3LYP 1.338.** That is a MODEL-MISS below the L tier's 1.897.
- **wB97M-V 2.813** (sildenafil) and **2.709** (atorvastatin).

Every cell was TREATED. r2SCAN and B3LYP kept the `fp64+fp32` cache, and wB97M-V kept its `fp64`
cache, on the 96 GB card.

**Bases** (four molecules, def2-universal-jkfit throughout):

| | def2-SVP | def2-mTZVPP | def2-TZVP |
|---|---|---|---|
| r2SCAN | 1.693 | 1.762 | 2.004 |
| B3LYP | 1.957 | 1.744 | 1.590 |

B3LYP's gain falls with both system size and basis size. The suspected cause is the FP32 DF-K
path, but **no stage was timed**, so it is not established.

## C3. On the two FP64-strong data-centre cards tested, the FP32 switch is harmful for B3LYP and wB97M-V `[N2]`

Trio geomeans (paracetamol, propranolol, celecoxib). The precision effect is mixed/cache: the FP32
arithmetic's gain, with the cache's own gain removed.

| card | pods | mixed/stock r2SCAN / B3LYP / wB97M-V | precision effect | cache alone |
|---|---|---|---|---|
| L40S | 1 per cell (X2a, X2b) | 1.729 / 1.524 / 2.994 | 1.655 / 1.463 / 2.981, PAYS | 1.045 / 1.041 / 1.004 |
| H100 | 1 (X3) | 1.320 / 0.905 / 0.577 | 1.084 / 0.745 / 0.539, NEUTRAL / HARMS / HARMS | 1.217 / 1.215 / 1.070 |
| A100 | 2, both DEGRADED | not quotable as geomeans | clean cells only (9 cells, 12 readings): 11 of 12 NEUTRAL or HARMS | 1.153–1.245 on the clean r2SCAN/B3LYP cells |

- **How big the harm is.** On these cards the mode makes wB97M-V about 2× *slower*: mixed/stock
  0.470–0.596 on the A100 and 0.521 for celecoxib on the H100. It makes B3LYP up to 1.4× slower
  (0.711).
- **What the data does not show:** a threshold or a mechanism.
  - The data has only two levels of FP64:FP32 throughput, about 1:64 (PRO 6000, L40S) and about 1:2
    (H100, A100).
  - Those two groups also differ in architecture, bandwidth, and the NVRTC/df64 kernel tuning, which
    was done on Blackwell.
  - "Gate on the FP64:FP32 ratio" is therefore a hypothesis, not a tested rule. The RESULT itself
    says "a reading of these pods, not a tested gating rule".
- **The implementation has no device check today.** On these cards the mode runs and is slower.
  The RFC must either propose a guard (for example a warning or refusal on sm_80/sm_90-class
  devices, with a test) or say plainly that users must not enable it there.
- **The FP64 AO cache alone** (`MixedPrecision(ao_cache_fp64=True)`, no FP32 arithmetic) pays about
  1.15–1.25 on these cards **for r2SCAN and B3LYP only**. For wB97M-V it is NEUTRAL: 1.070 on the
  H100.
- **No data:** the RTX 5090 (n/a-by-capacity, two attempts of a pre-registered three) and the PRO
  6000's r2SCAN/B3LYP precision effect, which exists only on the CONTENTION-UNKNOWN C0 (1.543 /
  1.633) and is not claimed.
- The H100 and L40S rows each rest on a single pod per cell.

## C4. Accuracy: 17 of 18 downstream cells are within every ceiling `[N4, N7]`

- **Ceilings:** max |Δg| 1e-5 Ha/Bohr, max |Δμ| 1e-4 D, |ΔE| 1e-8 Ha.
- **Reference:** a tight stock SCF (`conv_tol 1e-11`, `conv_tol_grad 3e-6`).
- **Gradients:** every gradient is within the ceiling with ≥ 3× margin; the largest is 3.10e-6
  Ha/Bohr.
- **Energies:** every |ΔE| is ≤ 1.6e-10 Ha.
- **The one FAIL:** fluconazole r2SCAN's dipole, at 1.004e-4 D. Its pod, D1, is `STATUS=FAIL`, so
  all 12 of D1's numbers are **UNQUOTABLE** under PREREG-0 Amendment 1. They are shown here with that
  label.
- **What stock itself reaches.** Stock started from `init_guess='atom'` (S′), rather than the
  default guess, reaches 1.058e-4 D on celecoxib B3LYP. So the dipole ceiling is **of the same order
  as stock's own dependence on the initial guess** at these settings. A same-guess stock comparator
  was not recorded. ρ (mixed's deviation over S′'s) uses a single S′ sample.
- **Mixed-specific residual (ρ > 3):** r2SCAN paracetamol (ρ_g 4.09, ρ_μ 8.56), wB97M-V propranolol
  (ρ_μ 4.89) and wB97M-V fluconazole (ρ_μ 3.97).
  - r2SCAN paracetamol's gradient residual reproduces on two pods: 1.368e-6 on C0 and 1.369e-6 on
    D1. It is systematic, not noise.
- **Predicted gradient ≤ 1e-6 Ha/Bohr:** missed in **14 of 18** cells (8 of 12 in D1). PREREG-4's
  RESULT printed 15 and 9; see the ledger.
- **Not covered:** the mode does not touch gradients or Hessians, which run stock. No geometry
  optimisation was run end to end.

## C5. MIG (deployment advice, not a property of the library) `[N5, N6, N7]`

**MIG gain** = 4 × whole-card wall / one-instance wall. The whole card is an RTX PRO 6000 Server
Edition running **one SCF at a time**. Four concurrent SCFs (or MPS) on the whole card were not
measured, and that is the comparison an operator would face.

| | trio stock (M1 / M2) | trio mixed (M1 / M2) | 475–559 Da stock | 475–559 Da mixed |
|---|---|---|---|---|
| r2SCAN | 1.43 / 1.46 | 1.75 / 1.78 | 1.19–1.28 | 1.29–1.44 |
| B3LYP | 1.39 / 1.42 | 1.25 / 1.26 (celecoxib 0.92) | 1.09–1.16 | 0.98–1.11 |
| wB97M-V | 1.03 / 1.03 | 1.19 / 1.18 | not measured | not measured |

- **The gain is a projection.** It assumes four instances run concurrently without slowing each
  other. Only one instance was measured at a time, and the neighbouring slices belonged to other
  tenants.
- **Replication.** The instance side is replicated: two cards, three slices, walls within 3 %. The
  whole-card side is **one pod per scope**: M0 for the trio, F0 for the larger molecules. Whole-card
  stock walls vary about 15 % between pods, so labels near the thresholds are fragile. That covers
  1.16–1.20 read as PAYS and 0.92/0.93 read as HARMS.
- **M0's paracetamol wB97M-V cell is NOISY** (stock spread 15.7 %). Both wB97M-V paracetamol gains
  rest on it.
- **Fit.** One instance holds the full cache tier (`fp64+fp32`, or `fp64` for wB97M-V) up to at
  least celecoxib (381 Da).
  - From sildenafil (475 Da) up, it falls back to the FP32-only cache, as pre-registered.
  - The B3LYP DF tensor stayed on the device up to atorvastatin (559 Da). r2SCAN builds none.
  - These limits are for unpruned grids, and pruning would move them.
- CPU and host RAM per pod were not recorded.

## Cold start

The first `mixed` SCF in each process is slow on Blackwell:
- C0 83.8 s, L12 96.1 s, M0 82.3 s, M1a 96.0 s and XL1 87.9 s (r2SCAN);
- C0w 64.0 s (wB97M-V);
- warm runs of the same cells take about 2–14 s.

On the L40S and H100, the first `mixed` SCF in a process takes about 8–9 s (8.6 s and 8.4 s). Because `mixed` ran first in each
pair, that first run also absorbs process-wide start-up costs. How much is the mode's own JIT/kernel
compilation was not separated. **For workflows that run one SCF per process, the first SCF can be
a net loss.** The gains above are steady-state, per-process figures.

## What can and cannot be verified from this package

- **Can:** every number in this file, against the per-run CSVs (`verify_native.py`); that the CSVs
  carry consistent run identifiers; the 30 geometries against their checksums.
- **Cannot, from public material:**
  - The sentinels the CSVs were extracted from, the measuring harness, the extractor, and the git
    history that orders protocols, amendments and runs all live in
    `chris-lee-mc/gpu-conformer-engine`, which is **private**.
  - `verify_native.py` checks claims against CSVs, not CSVs against sentinels.
  - Two governing amendments (PREREG-0 Amendment 3, PREREG-6 Amendment 1) were committed 17 s and
    87 s before the dispatches they govern, on a results branch; the dispatched code does not
    contain them. Their order rests on commit times.
- **Dates.** Some RESULT and status dates in `prereg/` are a day late (written 2026-10-06 for
  commits of 2026-10-05). See the ledger.
- **`reproduce_native.py`** transcribes the measurement path and has not been run on a GPU in this
  form.

## Not claimed

- **Untouched by the mode:** gradients and Hessians run stock.
- **Refused:** range-separated XC and K. The VV10 component alone handles wB97M-V.
- **Not measured:**
  - library-default settings and pruned grids;
  - running without cuTENSOR;
  - UKS and multi-GPU;
  - the RTX 5090;
  - MIG profiles other than 1g.24gb;
  - concurrent instances, and a concurrent whole card;
  - wB97M-V on a MIG instance above 259 Da.
- **Not transferable:** timings on one card model say nothing about another. The PRO 6000
  Workstation and Server Editions are treated as different models (five models in all: PRO 6000 WS,
  PRO 6000 SE, L40S, H100, A100).

## Ledger: every pre-registered miss, correction and procedural lapse

**MODEL-MISSes:**
- PREREG-1: the bridge leg (1.586 vs 1.78–1.84); "flat across tiers".
- PREREG-2:
  - C0 B3LYP and wB97M-V; C0w (above);
  - L40S B3LYP mixed/stock (below), wB97M-V mixed/stock and precision, r2SCAN precision (above);
  - H100 B3LYP and wB97M-V, mixed/stock and precision (far below: predicted 1.20 / 1.35, measured
    0.905 / 0.577).
- PREREG-3: XL B3LYP (below); basis predictions: r2SCAN and B3LYP at SVP (above), B3LYP at TZVP
  (below).
- PREREG-4: gradient ≤ 1e-6 (14 of 18).
- PREREG-5: mixed/stock inside an instance, all 8 cells; mixed MIG gains for r2SCAN and paracetamol
  B3LYP (above); stock paracetamol B3LYP and wB97M-V (vs the per-molecule table, which PREREG-5's
  RESULT decided to apply to every functional); M0 B3LYP against C0.
- PREREG-6: B3LYP sildenafil mixed MIG gain (above).

**Wrong or withdrawn predictions:**
- PREREG-6: the three r2SCAN DF-placement predictions (withdrawn by Amendment 1: no tensor is
  built); B3LYP atorvastatin placement (host predicted, device measured).
- After PREREG-5, my estimate that the DF tensor moves to host past ~500 Da, and that the whole card
  wins there, was wrong.

**Corrections to the RESULT texts**, each found while building or reviewing this package; the
listed numbers are checked by `verify_native.py`:

| where | printed | correct |
|---|---|---|
| PREREG-2 part 2 and PREREG-5, L12 trio B3LYP | 1.859 | 1.860 (1.8595) |
| PREREG-6, atorvastatin B3LYP stock MIG gain (twice) | 1.09 | 1.10 (1.0950) |
| PREREG-4 part 1, gradient > 1e-6 | 9 of 12 | 8 of 12 (C0's cell was counted in) |
| PREREG-4 part 2, gradient > 1e-6 | 15 of 18 | 14 of 18 |
| PREREG-4 part 1, ρ_g ≤ 3 | 10 of 12 | 11 of 12 |
| PREREG-3, XL cache tiers | "all at fp64+fp32" | r2SCAN/B3LYP fp64+fp32, wB97M-V fp64 |
| PREREG-5, r2SCAN stock MIG gain range | 1.4–1.6 | 1.35–1.55 |
| PREREG-5 and PREREG-6, RESULT/status dates | 2026-10-06 | committed 2026-10-05 |
| PREREG-5, "A sustained ~18 % throughput gain at scale is material" | — | added after the data; "at scale" contradicts the untested concurrency assumption, so withdrawn |

**Procedural lapses:**
- The RTX 5090's third capacity attempt was not made inside its window.
- X4's DEGRADED flag was missed at first, so its re-run came three days later.
- PREREG-0 §6's "two infra-starved repeats" rule was extended to two DEGRADED A100 pods.
- The contention rule was written after its first case had been seen.
