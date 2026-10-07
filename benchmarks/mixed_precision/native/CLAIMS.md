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
- **Between pods.**
  - Healthy pods of the same edition agree closely, to about 1 % on stock walls and about 6 % on
    ratios. For example, paracetamol wB97M-V stock is 19.07 s on W1 against 18.94 s on C0w.
  - Including the CONTENTION-UNKNOWN C0 or the Server Edition, readings span 10–20 % or more, and
    stock walls 15–37 %. For example, the PRO 6000 B3LYP trio is 1.688 on C0, 1.860 on L12 and
    2.019 on M0.
  - L12 (r2SCAN, B3LYP) and the H100 were each replicated once (PREREG-7):
    - L12's geomeans moved by at most 0.090. Both replicates were DEGRADED, and they drew the
      same physical card.
    - On the H100, B3LYP's mixed/stock moved by 0.115, past the ±0.10 band.
  - Treat the third decimal as reproducibility of the data, not of the hardware.

## C1. RTX PRO 6000 (Blackwell, FP64-limited): the mode pays for every functional tested `[N1]`

| 24 drug-like molecules, 20–44 atoms | mixed/stock geomean | S / M / L tiers | per-cell range | pods |
|---|---|---|---|---|
| r2SCAN | **1.657** | 1.576 / 1.665 / 1.804 | 1.523–1.871 | 1 (L12), replicated below |
| B3LYP | **1.783** | 1.736 / 1.779 / 1.897 | 1.584–1.994 | 1 (L12), replicated below |
| wB97M-V (VV10 component) | **2.914** | 2.848 / 2.995 / 2.806 | 2.593–3.250 | 5 healthy (W1–W3, W4r, W5) |

- All three are inside their pre-registered ±15 % bands. Bands that wide make "in band" a weak
  test.
- **Replication** (PREREG-7, `[N8]`). L12 was re-run twice, as L12r and L12r2. Both were DEGRADED
  (19 and 21 of 48 cells NOISY), so only clean cells are quoted:

  | | L12 | L12r / L12r2, all 24 | own clean cells | clean on both |
  |---|---|---|---|---|
  | r2SCAN | 1.657 | 1.721 / 1.718 | 1.717 (16) / 1.734 (16) | 1.709 / 1.720 (11) |
  | B3LYP | 1.783 | 1.844 / 1.868 | 1.872 (13) / 1.873 (11) | 1.830 / 1.818 (5) |

  - Every figure is within 0.090 of L12, inside the ±0.10 replication band. Every figure is also
    above L12.
  - Per-cell ratios on the replicates span 1.55–2.02 (r2SCAN) and 1.26–2.13 (B3LYP).
  - Both replicates and ST1 ran on one physical card, so the replication is two pods, not two
    cards. L12 had no NOISY cell, and the replicates' noise (mostly mixed-arm walls of 1–5 s) is
    unexplained.
  - **Quote 1.66–1.73 (r2SCAN) and 1.78–1.87 (B3LYP)**, not the third decimal.
- The gain rises from S to L. The "flat across tiers" prediction failed.
- **Two pods, C0 and W4, read about 30 % lower for wB97M-V** and are excluded from the 2.914:
  - C0 reads 2.059 on the trio;
  - W4 reads 1.92–2.13, its GPU drawing 576 W at idle;
  - C0w (the C0 cells re-run) reads 2.963, and W4r (the W4 cells re-run) reads 2.87–3.05.

  The exclusion rule (PREREG-0 Amendment 3: idle draw above 200 W) **was written after W4 and C0
  had been seen**.
  - On whole cards it has replaced only low readings: C0 and W4.
  - On MIG it replaced M2a and M2b, whose instance walls were 1.3–5.1 % *faster*, i.e. higher MIG
    gains, than their re-runs. On a MIG slice `nvidia-smi` power is card-wide, so the rule is
    uninformative there.

  On shared or cloud GPUs, expect readings like C0's and W4's.
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

B3LYP's gain falls with both system size and basis size.

**Why, from stage timing** (PREREG-7 ST1, one 96 GB Workstation card, `[N8]`). Exclusive stage
times, warm pair, stock/mixed:

| B3LYP | J/K share of stock | XC stock/mixed | J/K stock/mixed |
|---|---|---|---|
| paracetamol | 0.09 | 2.56 | 0.95 |
| celecoxib | 0.34 | 3.12 | 1.50 |
| sildenafil | 0.43 | 1.99 | 1.05 |
| atorvastatin | 0.50 | 3.08 | 1.07 |

- As molecules grow, J/K takes half of the stock SCF, while the mode's FP32 DF-K speeds it up by only
  5–7 % at 475–559 Da. The XC stage keeps a 2–3× gain throughout. So B3LYP's whole-SCF gain tends
  to the XC gain diluted by an unaccelerated half. **The K path is what limits B3LYP at size.**
- r2SCAN keeps K in FP64. Its XC is 85–86 % of the stock SCF, which is why r2SCAN's gain holds up.
- On a MIG 1g.24gb slice (ST2r), FP32 K gains a little (1.09 and 1.17). The prediction that it
  harms there was wrong.
- The FP64 AO cache alone barely moves XC (0.91–1.00 of stock) and leaves J/K unchanged.

## C3. On the two FP64-strong data-centre cards tested, the FP32 switch is harmful for B3LYP and wB97M-V `[N2]`

Trio geomeans (paracetamol, propranolol, celecoxib). The precision effect is mixed/cache: the FP32
arithmetic's gain, with the cache's own gain removed.

| card | pods | mixed/stock r2SCAN / B3LYP / wB97M-V | precision effect | cache alone |
|---|---|---|---|---|
| L40S | 1 per cell (X2a, X2b) | 1.729 / 1.524 / 2.994 | 1.655 / 1.463 / 2.981, PAYS | 1.045 / 1.041 / 1.004 |
| H100 | 2 (X3, X3r) | X3: 1.320 / 0.905 / 0.577; X3r: 1.398 / 1.021 / 0.615 | X3: 1.084 / 0.745 / 0.539; X3r: 1.084 / 0.764 / 0.563. NEUTRAL / HARMS / HARMS on both (B3LYP includes paracetamol, median warm walls 0.60–0.99 s) | X3: 1.217 / 1.215 / 1.070; X3r: 1.290 / 1.336 / 1.092 |
| A100 | 2, both DEGRADED | not quotable as geomeans | clean cells only (9 cells, 12 readings): 11 of 12 NEUTRAL or HARMS | 1.153–1.245 on the clean r2SCAN/B3LYP cells |

- **How big the harm is.**
  - wB97M-V: the mode makes it 1.6–2× *slower*. The H100 trio is 0.577 and 0.615, i.e. 1.73× and
    1.63× slower; the clean A100 cells are 0.488–0.596.
  - B3LYP: up to 1.4× slower (0.711).
  - r2SCAN: mixed/stock still reads 1.32–1.40 on the H100, but all of that is the FP64 AO cache.
    The FP32 switch itself is neutral (1.084 on both pods).
- **H100 B3LYP mixed/stock is a range, 0.905–1.021 (HARMS to NEUTRAL).** The replication moved it by
  0.115, past the ±0.10 band, so no single-pod headline is quoted. The *precision effect* replicated
  to within 0.024 on all three functionals. What moved is the cache's own gain (cache/stock).
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
- The L40S row rests on a single pod per cell.

## C4. Accuracy: 17 of 18 downstream cells are within every ceiling `[N4, N7]`

- **Ceilings:** max |Δg| 1e-5 Ha/Bohr, max |Δμ| 1e-4 D, |ΔE| 1e-8 Ha.
- **Reference:** a tight stock SCF (`conv_tol 1e-11`, `conv_tol_grad 3e-6`).
- **Gradients:** every gradient is within the ceiling with ≥ 3× margin; the largest is 3.10e-6
  Ha/Bohr.
- **Energies:** every |ΔE| is ≤ 1.6e-10 Ha.
- These two bounds pool D1's UNQUOTABLE cells with D2's. On the quotable D2 cells alone, the largest
  gradient is still 3.10e-6, on propranolol wB97M-V.
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

## Deployment notes, not claims: MIG and a concurrent whole card `[N5, N6, N7, N8]`

**This was C5. It is no longer a claim of the RFC.**
- PREREG-7's decision rule, written before CC1 ran, said: if a whole card running four SCFs
  concurrently reaches a throughput gain G ≥ 1.25 on both arms, MIG leaves the claims. It did.
- The material below is advice for people deploying the mode. It is not a property of the library.

**A concurrent whole card under MPS beats four MIG slices** (CC1, one RTX PRO 6000 Server Edition,
the trio × {r2SCAN, B3LYP}). G4 = 4 × T_S / T_C4. T_S is one process doing the six cells one at a
time; T_C4 is four processes doing them together, timed by makespan.

| arm | serial T_S (s) | 4 processes, time-sliced | 4 processes under MPS | MIG projection (PREREG-5) |
|---|---|---|---|---|
| stock | 51.20 | 0.96 | **1.52** | 1.333 / 1.363 |
| mixed | 29.19 | 1.02 | **1.75** | 1.340 / 1.363 |

- **Without MPS, four concurrent processes gain nothing.** That missed the prediction of 1.3–1.4.
  **MPS adds +0.56 and +0.73**, beyond the predicted 0 to +0.3. All four are MODEL-MISSes.
- MPS really engaged: the server process was seen in both MPS phases. No phase ran out of memory.
  Every mixed SCF under four processes kept the `fp64+fp32` tier, and every B3LYP tensor stayed on
  the device.
- **The memory pool was freed after every SCF in CC1**, in all its phases, so that four processes
  could share the card. That departs from the warm-pool rule used everywhere else.
- **Timing definition.** T uses each SCF's span, which includes building the molecule and the SCF
  object. The MIG figures use kernel walls.
- **Not measured: true four-instance MIG concurrency.** RunPod could not place four 1g.24gb pods on
  one card, so the MIG figure is still a projection.
- **Advice.** For throughput of many small SCFs on one card, run them concurrently under MPS. Use
  MIG only where isolation is the point.

### MIG, as measured (the former C5)

**MIG gain** = 4 × whole-card wall / one-instance wall. The whole card is an RTX PRO 6000 Server
Edition running **one SCF at a time**. That is not the comparison an operator faces; see the CC1 table above.

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

## Cold start: the CUDA driver's JIT cache, paid once per machine `[N7, N8]`

**On a fresh pod, the first SCF is slow.**
- On Blackwell (whole cards and MIG instances), the first mixed SCF took **64–168 s**, against warm
  walls of 1–108 s for the same cells.
- On the L40S, H100 and A100, the same first run was about 6–10 s slower than warm.

**What causes it** (PREREG-7 CS1, paracetamol; all five pre-registered thresholds met). Each phase
is a fresh process with the caches as shown. Cold extra = cold wall − warm wall:

| phase | caches at start | arm | r2SCAN cold extra (s) |
|---|---|---|---|
| P1 | both empty | **stock** | **134.0** |
| P2 | as P1 left them | mixed | 1.86 |
| P3 | as P2 left them | mixed | 0.81 |
| P4 | CuPy kernel cache wiped | mixed | 9.93 |
| P5 | CUDA driver cache wiped | mixed | **125.7** |

- **It is the driver JIT-compiling gpu4pyscf's PTX.** The gpu4pyscf-cuda12x 1.8.1 wheel is built
  for `70-real;80;90-real`, so on sm_120 the driver compiles compute_80 PTX on first load. P1
  leaves 60 files, 128 MB, in `CUDA_CACHE_PATH`. Wiping that cache brings back about 126 s; wiping
  CuPy's costs about 10 s.
- **It is not the mode's cost.** The stock arm paid it first (P1), and the mixed arm then paid under
  2 s (P2). In the campaign pods the mixed arm ran first in each pair, which is why the cost showed
  up there.
- **It is charged to whichever functional runs first in a process.** r2SCAN ran first in every
  phase, so the other functionals' small extras (≤ 1.62 s) say nothing about their own kernels.
- **Remedy.** Keep `CUDA_CACHE_PATH` on persistent storage, or ship a wheel that includes sm_120
  code. The campaign's speed figures exclude the cold pair, so they are steady-state figures either
  way.

## What can and cannot be verified from this package

- **Can, with nothing but this directory** (`verify_native.py`, 640 checks):
  - every number in this file, recomputed from the per-run CSVs;
  - every per-run CSV, re-extracted from the pod's own result file (`provenance/sentinels/`) by the
    shipped extractor, column for column;
  - each pod's status, contention, idle power, NOISY and DEGRADED flags;
  - the downstream readings;
  - the accuracy gates (|ΔE| ≤ 1e-8 Ha, cycles ±1), the policy and the cache tier, on every timed
    row;
  - the 30 geometries, against their checksums.

  `data/SHA256SUMS` and the sentinel hashes in `pods.csv` catch accidental edits. They are not
  tamper evidence: they sit beside what they hash.
- **Cannot be verified independently** (see `provenance/README.md`):
  - **Where the sentinels came from.** The pods wrote them to a branch of the private
    `chris-lee-mc/gpu-conformer-engine`. Nothing here proves a sentinel came from a GPU run.
  - **The order of protocols and runs.** `provenance/TIMELINE.csv` lists both, but the protocol
    times are git committer times and the run times come from a private repository's Actions API.
    Two governing amendments (PREREG-0 Amendment 3, PREREG-6 Amendment 1) were committed only 17 s
    and 87 s before the dispatches they govern, on a results branch; the dispatched code does not
    contain them. PREREG-7's final text, by contrast, is in the commit every PREREG-7 pod ran
    (`d05888b8`), 7.5 min before the first dispatch.
  - **The exact harness bytes.** `provenance/harness/` is the harness at `d05888b8`, the commit
    the PREREG-7 pods ran. Every earlier pod ran at the earlier commit listed in `pods.csv`.
- **Dates.** A few status dates and one RESULT date in `prereg/` are a day late. See the ledger.
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
  - four MIG instances running concurrently on one card;
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
- PREREG-4: gradient ≤ 1e-6 (14 of 18); gradient ≤ 3 × S′ (1 of 18: r2SCAN paracetamol).
- PREREG-2, A100: the predictions (precision effect 0.95 / 1.00 / 1.15) could not be evaluated,
  because both pods were DEGRADED. The clean wB97M-V cells read 0.47–0.57, far below 1.15.
- PREREG-5: mixed/stock inside an instance, all 8 cells; mixed MIG gains for r2SCAN and paracetamol
  B3LYP (above); stock paracetamol B3LYP and wB97M-V (vs the per-molecule table, which PREREG-5's
  RESULT decided to apply to every functional); M0 B3LYP against C0.
- PREREG-6: B3LYP sildenafil mixed MIG gain (above).
- PREREG-7:
  - ST1 (c): the cache arm's XC is not faster than stock's at sildenafil B3LYP (10.324 s vs
    10.319 s);
  - ST2: FP32 K was predicted to harm on a MIG slice, but J/K stock/mixed is 1.09 (ST2r);
  - CC1: G4 stock 0.96 and mixed 1.02, against predictions of 1.3 and 1.4; the MPS uplift of +0.56
    and +0.73, against a prediction of 0 to +0.3.

**Replication falsifier fired:** PREREG-7 X3r. H100 B3LYP mixed/stock moved by 0.115 (> 0.10), so it
is quoted as the range 0.905–1.021. The L12 replication's falsifier did not fire.

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
| PREREG-5 RESULT date; PREREG-6 "Status" and "Owner decision" dates | 2026-10-06 | committed 2026-10-05 (PREREG-6's RESULT date, 2026-10-06, is correct) |
| PREREG-5, "A sustained ~18 % throughput gain at scale is material" | — | added after the data; "at scale" contradicts the untested concurrency assumption, so withdrawn |
| PREREG-7 RESULT, CS1 cold extras: P1 wB97M-V, P2 wB97M-V, P4 r2SCAN, P4 B3LYP | 0.64, 1.13, 9.92, 1.61 | 0.65, 1.12, 9.93, 1.60 (computed from rounded readings; erratum in the protocol) |
| PREREG-7 RESULT, X3r − X3: B3LYP mixed/stock; r2SCAN precision effect | 0.116; 0.000 | 0.115; −0.001 |
| PREREG-7 RESULT, "DF build and eigensolver the same in every arm" | — | within 5 %, except r2SCAN's ~0.05 s DF-build call (up to 9 %, ≤ 5 ms) |

**Procedural lapses:**
- The RTX 5090's third capacity attempt was not made inside its window.
- X4's DEGRADED flag was missed at first, so its re-run came three days later.
- PREREG-0 §6's "two infra-starved repeats" rule was extended to two DEGRADED A100 pods.
- The contention rule was written after its first case had been seen.
- **`provenance/TIMELINE.csv` was missing from the first version of this package.** The README
  described it, but the repository's `*.csv` ignore rule kept it out of git. It is now shipped,
  regenerated from the same sources. The regenerated file reproduces every row of the unshipped one.
- **L12r was re-dispatched as L12r2 without asking the owner first.** PREREG-0 §3 requires the
  re-dispatch. For X4, the owner had been asked.
- **L12r, L12r2 and ST1 drew the same physical card.** That was not controlled.
- **PREREG-0 §6's two-repeats rule was again applied to two DEGRADED pods**, here L12r and L12r2.
- **The ST2 re-run was approved on 2026-10-07, after CC1 had been dispatched.** That changed the
  dispatch order, CS1 → L12r → L12r2 → X3r → ST1 → ST2 → CC1 → ST2r, but no protocol text.
