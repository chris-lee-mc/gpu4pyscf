# PREREG rfcbench-7: cold start, replication, B3LYP stage timing, and concurrency

**Status 2026-10-06: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`.
It freezes at the first PREREG-7 sentinel. Changes go only in dated amendments above RESULT,
written before the dispatch they govern.

**Owner decision of 2026-10-06.** The owner authorised the extra pods that the evidence-package
review ranked, in this order:
1. attributing the cold-start cost;
2. replicating L12 and the H100;
3. B3LYP stage timing;
4. concurrent-MIG runs, or dropping MIG from the RFC as deployment advice.

## Why

The independent review of the fork's evidence package (`benchmarks/mixed_precision/native/`) found
four questions that the existing data cannot answer:

1. **Cold start.** The first mixed SCF on a fresh Blackwell pod takes 64–168 s. Later processes on
   the same pod do not pay it. Does it belong to the mode, or to the first GPU work of any kind?
   Which on-disk cache removes it?
2. **Single pods.** The headline r2SCAN and B3LYP rows (L12) and the H100 row (X3) each rest on one
   pod.
3. **B3LYP.** B3LYP's gain falls with system size and basis, and reaches HARMS on a MIG instance.
   The suspected cause is the FP32 DF-K path, but no stage was ever timed.
4. **MIG.** The MIG gain was measured against a whole card running one SCF at a time. The
   operator's actual alternative is several SCFs at once on the whole card.

## Pods

| pod | mode | class | what |
|---|---|---|---|
| CS1 | `coldstart` | `pro6000` | the cold-start phases below |
| L12r | `ladder` | `pro6000` | L12's exact inputs: 24 molecules × {r2SCAN, B3LYP}, with the bridge leg |
| X3r | `cross` | `h100` | X3's exact inputs: the trio × 3 functionals × 3 arms |
| ST1 | `stages` | `pro6000` | stage timing: B3LYP {paracetamol, celecoxib, sildenafil, atorvastatin}, r2SCAN {celecoxib, sildenafil} |
| ST2 | `stages` | `pro6000mig` | stage timing on one 1g.24gb instance: B3LYP {celecoxib, sildenafil}. No r2SCAN: it would exceed the work cap. |
| CC1 | `concur` | `pro6000se` | the concurrency phases below, on a whole Server Edition card |

- Six pods, dispatched one at a time in this order.
- **Reserve:** one pod, for an infra failure only.
- **CONTENDED pods:** PREREG-0 Amendment 3 applies. A CONTENDED pod is re-run once, and only after
  the owner says so.
- Every pod uses fork `63af0568`, and the cell settings are those of PREREG-0: mTZVPP /
  tzvpp-jkfit, grid level 3 unpruned, cuTENSOR, `ao_cache_fp64=True`, and B3LYP
  `xc_switch_tol=3e-4`.
- The three new modes run `rfcbench_extra.py` in place of the per-group cell runner. They refuse
  to dispatch unless this file is committed with an empty RESULT.

## Gates

These apply to CS1, ST1, ST2 and CC1, for every planned SCF:
- present exactly once;
- no error;
- converged;
- energy within 1e-8 Ha of the stock energy of the same molecule and functional on the same pod;
- for the mixed and cache arms, the AO-cache tier and the fork's own note form a fit outcome that
  `rfcbench_cells.classify_fit` accepts. An override note, or any note that is not fit-based, is a
  FAIL.
- On CS1 and ST1 (one process on a 96 GB Workstation card), every mixed and cache SCF must be
  `TREATED`.

Phases may be **skipped** without failing the pod only for the reasons listed under CC1: MPS
unavailable, unusable or not engaged, and out of memory under four processes. Warm-up SCFs are not
counted for coverage, but an error in one fails the pod. L12r and X3r carry their original modes'
gates unchanged.

**What every SCF records**, beyond the wall:
- the AO-cache tier and note;
- the XC and K phase lists;
- the DF tensor's placement and size;
- the grid size;
- the CuPy pool's bytes at the end.

Every pod also records the default CuPy and CUDA caches it started with, i.e. the campaign's
seeded kernel cache, which these modes do not use.

## CS1: cold start

The CuPy kernel cache (`CUPY_CACHE_DIR`) and the CUDA driver JIT cache (`CUDA_CACHE_PATH`) are
pointed at directories the harness owns. Both are emptied before P1. Each phase is a fresh process
and runs paracetamol × {r2SCAN, B3LYP, wB97M-V}, twice per functional: a cold run, then a warm one.

| phase | caches at start | arm |
|---|---|---|
| P1 | both empty | stock |
| P2 | as P1 left them | mixed |
| P3 | as P2 left them | mixed |
| P4 | CuPy cache emptied | mixed |
| P5 | CUDA cache emptied (CuPy cache as P4 left it) | mixed |

**Reading.** The *cold extra* of a phase and functional is its cold wall minus its warm wall. Cache
file counts and bytes are recorded after the cuTENSOR self-test of each process and after every
phase. Each process's self-test compiles a few CuPy kernels before the first timed SCF, as in every
campaign group.

**P1 is "empty caches", not "a production pod".** Campaign pods seeded `~/.cupy/kernel_cache` from
a prewarm asset. These phases point CuPy elsewhere, so P1 and P2 start colder than any campaign pod
whose kernel cache was seeded.

**Predictions (thresholds, not bands).** These were revised before any dispatch, after the
pre-flight review. The review pointed out that the gpu4pyscf-cuda12x 1.8.1 wheel is built for
`70-real;80;90-real`. On sm_120 (Blackwell) the CUDA driver therefore JIT-compiles the compute_80
PTX of every gpu4pyscf module on first load, and caches the result in `CUDA_CACHE_PATH`. The
originally written predictions are kept below for the record.
- **P1:** r2SCAN's cold extra ≥ 30 s. The first load of gpu4pyscf's modules pays it, whatever the
  arm, so it is not the mode's.
- **P2:** every cold extra ≤ 15 s. The mode's own FP32/df64 kernels add little once the modules are
  JIT-compiled.
- **P3:** every cold extra ≤ 5 s.
- **P4:** every cold extra ≤ 15 s. Wiping the CuPy cache costs only the CuPy kernels.
- **P5:** r2SCAN's cold extra ≥ 30 s. Wiping the driver JIT cache brings the cost back.

**Falsifiers:**
- If P1's r2SCAN cold extra is below 10 s while P2's is at least 30 s, the cost **is** the mode's.
- If P5's is below 10 s while P4's is at least 30 s, the CuPy cache, not the driver's, carries it.

*Originally written, superseded before dispatch:*
- P2: at least one cold extra ≥ 10 s;
- P4: r2SCAN ≥ 20 s (the CuPy cache carries it);
- P5: every cold extra ≤ 10 s.

## L12r and X3r: replication

- **Readings:** the same as L12's and X3's, computed the same way.
- **Pod-to-pod difference:** L12r − L12 and X3r − X3, per geomean.

**Predictions (band ±0.10 absolute on each geomean):**
- L12r: r2SCAN 1.657 and B3LYP 1.783 (ladder geomeans).
- X3r: mixed/stock 1.320 / 0.905 / 0.577; precision effect 1.084 / 0.745 / 0.539.

**Falsifier:** a difference above 0.10 on any of them. The RESULT would then quote the pair's range
and drop the single-pod headline.

## ST1 and ST2: stage timing

- **Timed calls.** Every SCF (arms mixed, cache, stock; pair 0 cold and pair 1 warm) runs with
  sync-to-sync timers around:
  - XC: `numint.nr_rks`, plus the mixed-precision XC entry points;
  - J/K: `with_df.get_jk`;
  - DF build: `with_df.build`;
  - the eigensolver: `mf.eig`.
- **Exclusive times.** A timed call nested in another is subtracted from it.
- **The timers add synchronisations.** The walls of these pods are never used as speed readings.
  Only pair 1 is read.
- **Readings, per cell:**
  - each stage's exclusive time per arm;
  - the stage ratio stock/mixed;
  - each stage's share of the stock SCF;
  - "other" = wall minus the timed stages.

**Predictions:**
- **ST1, B3LYP:**
  - (a) The J/K share of the **stock** SCF rises by ≥ 15 percentage points from paracetamol to
    atorvastatin.
  - (b) At atorvastatin, the J/K stage's stock/mixed ratio is ≤ 1.3, so FP32 K buys little, while
    the XC stage's is ≥ 2.0.
  - (c) The cache arm's XC stage is faster than stock's, and its J/K equals stock's to within 5 %.
- **ST1, r2SCAN:** XC is ≥ 60 % of the stock SCF for both molecules.
- **ST2, the instance, B3LYP celecoxib:** the J/K stage's stock/mixed ratio is ≤ 1.0, so FP32 K
  harms on the slice.

**Falsifier:** if B3LYP atorvastatin's J/K stock/mixed is ≥ 1.8 on ST1, then the K path is **not**
what limits B3LYP at size.

## CC1: concurrency on a whole Server Edition card

**Cells:** the trio × {r2SCAN, B3LYP}, six cells. For each arm (mixed, then stock):
- **S:** one process runs the six cells twice. The serial pass time T_S is the sum of the second
  pass's walls.
- **C4:** four processes are released together behind a barrier. Each first runs one untimed
  warm-up SCF per functional (paracetamol r2SCAN, paracetamol B3LYP), then runs the six cells once.
  T_C4 is the makespan, from the first start to the last end.
- **One time definition.** Both T_S and T_C4 use an SCF's span, t1 − t0: building the molecule and
  the SCF object, the kernel, and recording. The PREREG-5 comparator below uses kernel walls. For
  these molecules the build is under about a second per SCF, which is disclosed.
- **C4-MPS:** as C4, under an MPS control daemon started for the phase.
  - If the daemon cannot start in the container, the phase is skipped and recorded as "MPS
    unavailable".
  - Before the four clients start, one short probe client runs under MPS. If the probe fails, or
    if no SCF of the phase succeeds, the phase is skipped and recorded as "MPS unusable", with the
    errors.
  - If no `nvidia-cuda-mps-server` process is ever seen during the phase, it is skipped as "MPS not
    engaged", and never quoted as MPS.
  - None of these fails the pod. A phase with some SCFs succeeding and some failing does fail it.
- **Out of memory.** An out-of-memory error in a four-process phase is a **reading**: "four
  processes do not fit". The phase is skipped and its G is not computed. Out of memory in the
  serial phase is a FAIL.
- **Phase cap.** Each phase is capped at twice its over-estimate plus 300 s, so that a hang cannot
  consume the phases after it.

**Memory pool.** In every CC1 phase (S included, so the comparison is like for like), each process
frees its CuPy memory pool after every SCF, so that four processes can share the card. That departs
from the campaign's warm-pool rule (PREREG-0 §1), and only for CC1.

**Reading.** Concurrent gain G4 = 4 × T_S / T_C4 (and the same under MPS); the extractor reports
it as `C4_<arm>/G`: the throughput of four
concurrent SCFs relative to one at a time on the same card.

**Comparator, computed now from PREREG-5's data.** The MIG throughput gain on the same six cells is
4 × Σ M0 wall / Σ instance wall:
- stock: **1.333** (M1) and **1.363** (M2);
- mixed: **1.340** (M1) and **1.363** (M2).

**Predictions (band ±0.3):**
- G4: stock 1.3, mixed 1.4.
- MPS adds 0 to +0.3.
- Cache tiers may degrade under four processes, because each one's budget is 0.7 × the free memory
  it sees. The tiers, notes and DF placement are recorded, and the RESULT reports any change from
  S's.

**Decision rule, written now:**
- Let G be the larger of G4 and G4-MPS for each arm.
- **If G ≥ 1.25 for both arms** (that is, within 0.1 of the lower MIG figure), a concurrent whole
  card matches the four-instance projection. **C5 (MIG) is then dropped from the RFC's claims** and
  kept in the evidence package as deployment notes only.
- **Otherwise** C5 stays as deployment advice, quoting G next to the MIG figure.

True four-instance concurrency on one card is **not** measured. RunPod cannot place four 1g.24gb
pods on the same card on request, and MIG reconfiguration needs host access the pods do not have.
This is disclosed, not worked around.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## Amendments

None yet.

## RESULT

### RESULT (2026-10-07): **PASS** on all six pods and both re-runs. Cold start is the driver's JIT cache; both replications hold on their geomeans, but H100 B3LYP moves past the band; FP32 K buys little at size; a concurrent whole card under MPS beats MIG, so **C5 leaves the RFC's claims**

**Pods.** Every sentinel's line 2 matches its run. Each was dispatched alone, from `main` @
`d05888b8`, with fork `63af0568`.

| pod | run | card | idle W | status |
|---|---|---|---|---|
| CS1 | `37535025280` | `GPU-4ff3576c`, Workstation | 75 | PASS, 30 SCFs |
| L12r | `37540378856` | `GPU-254d2571`, Workstation | 54 | PASS, **DEGRADED** (19/48 NOISY) |
| L12r2 | `37544474854` | `GPU-254d2571`, Workstation | 63 | PASS, **DEGRADED** (21/48 NOISY) |
| X3r | `37548043266` | `GPU-6c6e9dd9`, H100 80GB HBM3 | 123 | PASS, 0 NOISY |
| ST1 | `37549313181` | `GPU-254d2571`, Workstation | 57 | PASS, 36 SCFs |
| ST2 | `37550550221` | `GPU-753a81b8` / `MIG-f0f1840f`, Server Edition | **241** | PASS, **CONTENDED** |
| ST2r | `37621705258` | `GPU-f354bf4c` / `MIG-d21bfa9c`, Server Edition | 85 | PASS, 12 SCFs |
| CC1 | `37552079534` | `GPU-24443e9d`, Server Edition | 87 | PASS, 120 SCFs, nothing skipped |

- **Re-runs.** L12r2 is the re-dispatch PREREG-0 §3 requires for a DEGRADED pod. It was DEGRADED
  again, so §6 applies: the clean cells are reported and the pod aggregates are not quoted alone.
  ST2r is the CONTENDED re-run, dispatched after the owner said so (2026-10-07). ST2's numbers are
  reported separately and never pooled. The reserve pod was not used.
- **Same card three times.** L12r, L12r2 and ST1 all drew the same physical card (`GPU-254d2571`).
  So the second DEGRADED run is not an independent sample of hardware. That is disclosed, not
  corrected.
- Every gate held: coverage, no errors, all converged, |ΔE| vs the pod's stock ≤ 1e-8 Ha, and fit
  classification. Every CS1 and ST1 mixed/cache SCF is `TREATED`. The stock-vs-stock determinism
  floor is ≤ 2.8e-12 Ha.
- CSVs: `rfcbench-data/cs1.csv`, `l12r.csv`, `l12r2.csv`, `x3r.csv`, `st1.csv`, `st2.csv`,
  `st2r.csv`, `cc1.csv`.

#### CS1: the cold start belongs to the CUDA driver's JIT cache. All five thresholds met, no falsifier fired

Cold extra = cold wall − warm wall, in seconds:

| phase | caches at start | r2SCAN | B3LYP | wB97M-V | threshold | outcome |
|---|---|---|---|---|---|---|
| P1 stock | both empty | **134.0** | 1.45 | 0.64 | r2SCAN ≥ 30 | met |
| P2 mixed | as P1 left them | 1.86 | 0.19 | 1.13 | all ≤ 15 | met |
| P3 mixed | as P2 left them | 0.81 | 0.01 | 0.06 | all ≤ 5 | met |
| P4 mixed | CuPy cache wiped | 9.92 | 1.61 | 1.62 | all ≤ 15 | met |
| P5 mixed | CUDA cache wiped | **125.7** | 0.01 | 0.18 | r2SCAN ≥ 30 | met |

- **The cost is the driver JIT-compiling gpu4pyscf's compute_80 PTX for sm_120.** P1 filled
  `CUDA_CACHE_PATH` with 60 files, 128 MB. Wiping it (P5) brings back about 126 s. Wiping the CuPy
  cache (P4) costs about 10 s.
- **It is not the mode's.** The stock arm pays it first (P1), and the mixed arm then pays under 2 s
  (P2).
- **It is charged to whichever functional runs first in the process.** r2SCAN ran first in every
  phase, so B3LYP's and wB97M-V's small extras say nothing about their own kernels' share.
- So the "64–168 s first mixed SCF" in the package is a fresh-pod artifact of the 1.8.1 wheel's
  arch list. A persistent `CUDA_CACHE_PATH` removes it, and so would a wheel built with
  `120-real`.

#### L12r: the ladder replicates within ±0.10, on all cells and on clean cells

| geomean, mixed/stock | L12 | L12r all | L12r2 all | L12r clean (n) | L12r2 clean (n) | clean on both (n) |
|---|---|---|---|---|---|---|
| r2SCAN | 1.657 | 1.721 | 1.718 | 1.717 (16) | 1.734 (16) | 1.709 / 1.720 (11) |
| B3LYP | 1.783 | 1.844 | 1.868 | 1.872 (13) | 1.873 (11) | 1.830 / 1.818 (5) |

- Every replicate figure is within 0.10 of L12's. The largest difference is +0.090 (B3LYP, L12r2
  clean cells). **The falsifier did not fire.**
- Every replicate is above L12. Both drew fast pods (canary 3.11 s and 2.87 s, below the 3.8–5.5 s
  band). L12's canary was 3.08 s.
- L12 had no NOISY cell. Both replicates are noisy mostly on the mixed arm of 1–5 s cells. The
  cause is not known, and none is claimed.
- Per-cell ratios across the two replicates span 1.55–2.02 (r2SCAN) and 1.26–2.13 (B3LYP).

#### X3r: H100 B3LYP moves past the band. **The falsifier fired**, so the single-pod headline is dropped

| H100 | X3 | X3r | difference |
|---|---|---|---|
| r2SCAN mixed/stock | 1.320 | 1.398 | +0.078 |
| **B3LYP mixed/stock** | **0.905 HARMS** | **1.021 NEUTRAL** | **+0.116** |
| wB97M-V mixed/stock | 0.577 | 0.615 | +0.038 |
| r2SCAN precision effect (mixed/cache) | 1.084 | 1.084 | 0.000 |
| B3LYP precision effect | 0.745 | 0.764 | +0.019 |
| wB97M-V precision effect | 0.539 | 0.563 | +0.024 |

- B3LYP mixed/stock differs by 0.116 > 0.10. As pre-registered, the RESULT quotes the pair's range,
  **0.905–1.021 (HARMS to NEUTRAL)**, and drops "H100 B3LYP HARMS" as a single-pod headline.
- **The precision effect replicates tightly** (≤ 0.024) on all three functionals. Its claim stands:
  on an FP64-strong card, the FP32/df64 kernels are neutral at best and harmful for B3LYP and
  wB97M-V. The cache tier is what moves between pods.
- X3r idled at 123 W, under the 200 W limit, but higher than any Blackwell pod here.

#### ST1 and ST2r: XC carries the gain. FP32 K buys little at size, but it does not harm on the slice

Pair 1 only. Stage times are exclusive. Ratios are stock/mixed.

| pod | cell | J/K share of stock | XC ratio | J/K ratio | cache XC / stock | cache J/K / stock |
|---|---|---|---|---|---|---|
| ST1 | paracetamol B3LYP | 0.09 | 2.56 | 0.95 | 0.91 | 0.992 |
| ST1 | celecoxib B3LYP | 0.34 | 3.12 | 1.50 | 0.94 | 0.987 |
| ST1 | sildenafil B3LYP | 0.43 | 1.99 | 1.05 | **1.00** | 1.008 |
| ST1 | atorvastatin B3LYP | 0.50 | 3.08 | 1.07 | 0.95 | 1.000 |
| ST1 | celecoxib r2SCAN | 0.11 (XC 0.85) | 2.06 | 0.97 | 0.92 | 1.009 |
| ST1 | sildenafil r2SCAN | 0.11 (XC 0.86) | 2.31 | 1.04 | 0.97 | 0.974 |
| ST2r | celecoxib B3LYP, slice | 0.40 | 3.80 | **1.09** | 0.95 | 0.999 |
| ST2r | sildenafil B3LYP, slice | 0.48 | 2.01 | 1.17 | 0.95 | 1.001 |

The DF build and the eigensolver are the same in every arm. "Other" is ≤ 6 % of every stock SCF.

- **(a) met.** The J/K share of stock rises by 41 points, from 9 % to 50 %.
- **(b) met.** At atorvastatin, J/K is 1.07 (≤ 1.3) and XC is 3.08 (≥ 2.0).
- **(c) MODEL-MISS, on one cell.** The cache arm's J/K is within 5 % everywhere (≤ 2.6 %). But
  its XC is not faster at sildenafil B3LYP: 10.324 s vs 10.319 s.
- **r2SCAN met.** XC is 85–86 % of stock.
- **ST2 MODEL-MISS.** On the slice, celecoxib's J/K stock/mixed is 1.09, not ≤ 1.0. FP32 K
  gains a little on the slice; it does not harm. ST2 (CONTENDED) read 1.08 and 1.16, the same within
  0.01.
- **The falsifier did not fire** (atorvastatin J/K 1.07 < 1.8). So the K path is what limits
  B3LYP at size. By 500+ Da, J/K is half the stock SCF and FP32 K speeds it up by only 5–7 %, so
  B3LYP's whole-SCF gain tends toward the XC gain diluted by an unaccelerated half.
- **Disclosed, not explained:**
  - On the slice, sildenafil B3LYP mixed runs at tier `fp32` (as in PREREG-6).
  - Its timer counts 22 XC entries, against 15 for stock and cache. Times are exclusive, so the
    total is not double-counted.
  - Celecoxib on the slice is `fp64+fp32`.

#### CC1: a concurrent whole card under MPS beats the MIG projection. **Decision: C5 is dropped from the RFC's claims**

| arm | T_S (s) | T_C4 makespan (s) | G4 | T_C4-MPS (s) | G4-MPS | G = max | MIG comparator |
|---|---|---|---|---|---|---|---|
| stock | 51.20 | 212.92 | 0.96 | 134.91 | **1.52** | 1.52 | 1.333 / 1.363 |
| mixed | 29.19 | 114.63 | 1.02 | 66.70 | **1.75** | 1.75 | 1.340 / 1.363 |

- **MPS engaged** in both MPS phases. The control daemon answered, the probe ran, and
  `nvidia-cuda-mps-server` was seen. Nothing was skipped, and no phase ran out of memory.
- **Under four processes, the cache tiers and DF placement are unchanged from S.** Every mixed SCF
  is `fp64+fp32`, and every B3LYP tensor is on the device.
- **Predictions:**
  - G4 stock 1.3 → 0.96, MODEL-MISS.
  - G4 mixed 1.4 → 1.02, MODEL-MISS.
  - "MPS adds 0 to +0.3": it added +0.56 (stock) and +0.73 (mixed), MODEL-MISS on both.
  - Time-slicing four processes buys nothing. MPS buys more than predicted.
- **Decision rule.** G ≥ 1.25 for both arms, so **C5 (MIG) is dropped from the RFC's claims.** The
  MIG data stays in the evidence package as deployment notes only. For throughput of many small SCFs
  on one card, the measured advice is: run them concurrently under MPS, not MIG.
- **Not measured, as disclosed above:** true four-instance MIG concurrency on one card.
- The span definition (t1 − t0) includes the molecule and SCF build. The comparator uses kernel
  walls.

### Erratum (2026-10-07, found by the fork package's `verify_native.py` before merge)

Several figures above were computed from the pod's 2-decimal readings, not from the walls. The
values recomputed from the CSVs are:

| where | printed | correct |
|---|---|---|
| CS1 P1, wB97M-V cold extra | 0.64 | 0.65 |
| CS1 P2, wB97M-V cold extra | 1.13 | 1.12 |
| CS1 P4, r2SCAN cold extra | 9.92 | 9.93 |
| CS1 P4, B3LYP cold extra | 1.61 | 1.60 |
| X3r − X3, B3LYP mixed/stock (twice) | 0.116 | 0.115 |
| X3r − X3, r2SCAN precision effect | 0.000 | −0.001 |

None of these changes an outcome. All five CS1 thresholds still hold, and 0.115 is still above the
0.10 band, so the X3r falsifier still fires.

**Also imprecise:** "The DF build and the eigensolver are the same in every arm."
- They agree with stock to within 5 % in every pair-1 cell, with one exception: r2SCAN's DF-build
  call, which builds no tensor and takes about 0.05 s, varies by up to 9 % (≤ 5 ms).
- On the CONTENDED ST2, celecoxib's DF build is 4.8 % faster in the mixed and cache arms than in
  stock.

### Erratum 2 (2026-10-07, from the independent review of the fork package, before merge)

No outcome of a pre-registered decision changes: the CS1 thresholds, the X3r falsifier and the CC1
decision rule all reproduce. What changes is how some of the RESULT reads.

- **L12r: replication is not testable as pre-registered.**
  - PREREG-0 §3 makes a DEGRADED pod's aggregates unquotable. The RESULT nonetheless evaluated the
    falsifier on L12r's and L12r2's all-24 geomeans, and softened the rule to "not quoted alone".
    That was a reinterpretation.
  - The correct reading:
    - the pre-registered test cannot be run;
    - the all-24 geomeans are UNQUOTABLE;
    - a **post-hoc** comparison restricted to the same clean cells of L12 is consistent with it.
  - Same-cell differences:
    - r2SCAN: +0.057 and +0.062 on each pod's own clean cells; +0.049 and +0.060 on cells clean on
      both;
    - B3LYP: +0.074 and +0.086; and +0.072 and +0.059.
  - "The ladder replicates" is withdrawn.
  - The headline sentence "both replications hold on their geomeans" is also wrong: the X3r
    falsifier fired on a geomean.
- **L12r per-cell spans.** The spans quoted (1.55–2.02 and 1.26–2.13) include NOISY cells. On clean
  cells they are 1.57–2.02 (r2SCAN) and 1.56–2.10 (B3LYP).
- **L12r bridge legs.** These were not reported: 1.643 (L12r) and 1.658 (L12r2), against L12's
  1.586.
- **X3r: the mechanism was misattributed.** "The cache tier is what moves between pods" is wrong.
  - X3r was a slower pod: every median wall is 4–34 % above X3's, and stock slowed the most.
  - By functional, the slowdown is 1.25 / 1.25 / 1.12 for stock, against 1.18 / 1.14 / 1.10 for
    cache and 1.18 / 1.11 / 1.06 for mixed.
  - So every ratio over stock rose, and mixed/cache barely moved.
- **ST1: the causal sentence is withdrawn.** "The K path is what limits B3LYP at size" affirmed the
  consequent of the falsifier.
  - Holding celecoxib's stage ratios fixed and moving only the stage shares (untimed remainder held
    unaccelerated) explains about 19 % (sildenafil) and 37 % (atorvastatin) of the drop in
    whole-SCF stock/mixed.
  - The rest is unexplained. The FP32 J/K speedup is not monotonic (0.95, 1.50, 1.05, 1.07), and at
    sildenafil the XC switch came early (8 FP32 iterations against 10–11).
  - The J/K stage includes J, which is always FP64. K ran in FP32 for only 8–9 of 13–15 calls.
- **Cold start: the mechanism was stated beyond the data.**
  - CS1 establishes the driver JIT cache. It does not establish whose PTX is compiled, and that a
    `120-real` wheel would remove the cost is untested.
  - The cost is charged to the first SCF against an empty cache, not to whichever functional runs
    first in a process: in P2 and P3 the first functional paid ≤ 1.86 s.
  - Persisting the driver cache alone would leave about 10 s (P4).
- **CC1 caveats not stated.**
  - It is one pod.
  - Freeing the memory pool after every SCF makes T_S 2.1 % (stock) and 6.2 % (mixed) slower than
    M0's summed warm walls for the same cells, which inflates G by about that much.
  - "Beats MIG" means it beats the four-slice *projection*.
- **Small corrections:**
  - "X3r idled higher than any Blackwell pod here" is wrong: ST2 idled at 241 W.
  - The stock-vs-stock floor is 3.2e-12 Ha when CC1 is pooled across its phases, not ≤ 2.8e-12.
