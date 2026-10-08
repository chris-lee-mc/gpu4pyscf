# PREREG rfcbench-4: downstream accuracy (dipoles and analytic gradients)

**Status 2026-10-02: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`.
Freezes at the first `mode=downstream` sentinel. Changes go only in dated amendments above RESULT,
written before the dispatch they govern.

## Procedure, per cell (molecule × xc), PRO 6000, mTZVPP / tzvpp-jkfit

Four SCFs in one process, in this order:

| run | what it is |
|---|---|
| **R** | reference: stock at `conv_tol = 1e-11`, `conv_tol_grad = 3e-6`, `max_cycle = 100` |
| **S** | stock at the campaign settings |
| **S'** | stock at the campaign settings with `init_guess = 'atom'`: a different trajectory into the same basin, the honest noise comparator |
| **M** | `mixed`, with the cache, at the campaign settings |

For each run, the pod records:
- the dipole, `mf.dip_moment()` (Debye);
- the analytic gradient, `mf.nuc_grad_method().kernel()` (FP64, stock gradient code).

Before each gradient it asserts `mf._mixed_precision_state is None`. R, S and S' and their deltas
are printed before M runs.

**Fallback.** If R does not converge, the cell is read against S, with `reference = S` disclosed.
This is not a FAIL.

## Gates (GATED; fixed now, never derived from a measured floor)

- max |g_M − g_R| ≤ 1e-5 Ha/Bohr
- max |μ_M − μ_R| ≤ 1e-4 Debye
- |E_M − E_R| ≤ 1e-8 Ha

## Predictions and readings (disclosed)

- **Predictions:**
  - max |g_M − g_R| ≤ 1e-6 Ha/Bohr;
  - it is also no more than 3 × max |g_S' − g_R|.
- **Readings.** ρ_g = max |g_M − g_R| / max |g_S' − g_R|, and ρ_μ likewise:
  - ρ ≤ 3 reads "indistinguishable from SCF-convergence noise";
  - ρ > 3 reads "mixed-specific residual". That is still a PASS under the ceilings, and the RFC
    says so.

## Cells and pods

The molecules are the trio plus fluconazole, warfarin and omeprazole.

| pod | functionals |
|---|---|
| D1 | the 6 molecules × {r2SCAN, B3LYP} |
| D2 | the 6 molecules × wB97M-V |

## Budget

Two pods.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## Amendments

None yet.

## Amendment 1 (2026-10-02, before any dispatch)

- **M against S.** M is also gated against S under PREREG-0 §2: |ΔE| ≤ 1e-8 Ha, cycles ±1.
- **Recording.** The gradient and dipole are recorded for all four runs.
- **Convergence.**
  - If neither R nor S converges, the cell FAILs.
  - If S' does not converge, ρ is reported as `None` and the cell does not fail.
- **D2 split.** D2 may be split after the C0 calibration (PREREG-0 Amendment 1).

## RESULT

### RESULT, part 1 (2026-10-02): D1 **FAIL** on one dipole gate (fluconazole r2SCAN)

**The run.** Run `37038641497` attempt 1, sentinel
`ci-results/runs/rfcbench-3890e545d9cc-37038641497-1-3890e545.txt`, line 2
`RUN_ID=37038641497-1-3890e545` (checked). Repo `3890e545`, fork `63af0568`, on the PRO 6000
Workstation Edition, healthy (idle 77 W, PREREG-0 Amendment 3).

**Status.** `STATUS=FAIL`, `WORK_RC=0`. 11 of 12 cells OK; one is `GATE_FAIL`:
`downstream/fluconazole/r2scan`, max |μ_M − μ_R| = **1.004e-4 D**, against the **1e-4 D** ceiling.
The gate was fixed before the data and is **not changed**. The cell and the run FAIL as
pre-registered, and the numbers below are read `--unquotable`.

**All 12 cells** (reference R in every cell; ρ = M's deviation from R over S′'s deviation from R):

| | max \|Δg\| (Ha/Bohr) | max \|Δμ\| (D) | \|ΔE\| (Ha) | ρ_g | ρ_μ |
|---|---|---|---|---|---|
| ceiling | 1e-5 | 1e-4 | 1e-8 | | |
| r2SCAN paracetamol | 1.37e-6 | 5.00e-5 | 1.3e-10 | 4.09 | 8.56 |
| r2SCAN propranolol | 7.04e-7 | 2.10e-5 | 3.1e-11 | 0.76 | 0.31 |
| r2SCAN celecoxib | 1.16e-6 | 7.00e-5 | 3.7e-11 | 0.81 | 2.70 |
| **r2SCAN fluconazole** | 1.96e-6 | **1.004e-4** | 1.4e-10 | 2.20 | 1.39 |
| r2SCAN warfarin | 6.55e-7 | 6.43e-5 | 3.9e-11 | 0.24 | 1.17 |
| r2SCAN omeprazole | 1.49e-6 | 6.46e-5 | 9.3e-11 | 0.82 | 0.78 |
| B3LYP paracetamol | 1.34e-6 | 4.79e-5 | 2.3e-11 | 1.64 | 2.22 |
| B3LYP propranolol | 7.36e-7 | 1.91e-5 | 2.0e-11 | 0.68 | 0.72 |
| B3LYP celecoxib | 1.84e-6 | 8.49e-5 | 9.0e-11 | 0.70 | 0.80 |
| B3LYP fluconazole | 8.29e-7 | 5.16e-5 | 1.6e-11 | 0.53 | 1.06 |
| B3LYP warfarin | 1.67e-6 | 2.01e-5 | 2.1e-11 | 1.24 | 0.38 |
| B3LYP omeprazole | 1.00e-6 | 3.48e-5 | 4.5e-11 | 0.84 | 0.48 |

**Disclosed context. It does not change the verdict.**
- The stock noise comparator S′ is stock at the campaign settings from `init_guess='atom'`. Its own
  max |μ_S′ − μ_R| reaches **1.058e-4 D** on celecoxib B3LYP, above the same ceiling, and
  7.2e-5 D on fluconazole r2SCAN.
- The 1e-4 D ceiling therefore sits below the stock SCF's own convergence floor at
  `conv_tol = 1e-9` with the library-default `conv_tol_grad`.
- The failing cell's ρ_μ = 1.39 reads "indistinguishable from SCF-convergence noise".
- 10 of 12 cells have ρ_g ≤ 3 and 11 of 12 have ρ_μ ≤ 3. r2SCAN paracetamol, ρ_g 4.09 and
  ρ_μ 8.56, repeats C0's reading.
- Every gradient and energy passes with ≥ 5× margin.

**Predictions.**
- **Gradient ≤ 1e-6 Ha/Bohr: MODEL-MISS.** 9 of 12 cells exceed it; the largest is 1.96e-6.
- **Gradient ≤ 3 × S′'s deviation:** met in 11 of 12 cells.

D2 (wB97M-V) runs under the same, unchanged gates.

### RESULT, part 2 (2026-10-02): D2 (wB97M-V) **PASS**, all 6 cells

**Pods.** Three pods, each `STATUS=PASS` and `WORK_RC=0`, RUN_ID checked, healthy under PREREG-0
Amendment 3:

| pod | run | idle power | molecules |
|---|---|---|---|
| D2a | `37040169998` | 55 W | paracetamol, propranolol, celecoxib |
| D2b | `37042063662` | 26 W | fluconazole, warfarin |
| D2c | `37043596852` | 27 W | omeprazole |

Sentinels are at `ci-results/runs/rfcbench-3890e545d9cc-<run>-1-3890e545.txt`.

| wB97M-V | max \|Δg\| (Ha/Bohr) | max \|Δμ\| (D) | \|ΔE\| (Ha) | ρ_g | ρ_μ |
|---|---|---|---|---|---|
| ceiling | 1e-5 | 1e-4 | 1e-8 | | |
| paracetamol | 1.20e-6 | 4.07e-5 | 2.9e-11 | 0.76 | 1.14 |
| propranolol | 3.10e-6 | 7.66e-5 | 1.2e-10 | 0.96 | 4.89 |
| celecoxib | 1.87e-6 | 4.08e-5 | 6.3e-11 | 0.62 | 0.49 |
| fluconazole | 2.37e-6 | 3.76e-5 | 1.5e-10 | 2.41 | 3.97 |
| warfarin | 2.20e-6 | 1.03e-5 | 6.4e-11 | 0.84 | 0.66 |
| omeprazole | 2.17e-6 | 1.44e-5 | 1.0e-10 | 1.57 | 0.68 |

**Scope 4 summary.** The readings are pre-registered and disclosed.
- **Pods:** D1 FAIL (part 1), D2 PASS.
- **Cells:** 17 of 18 within every ceiling. The one FAIL is a dipole 0.4 % over the ceiling, on a
  cell whose ρ_μ is 1.39.
- **Gradients:** every gradient is within the ceiling with ≥ 3× margin (largest 3.10e-6 Ha/Bohr).
- **Energies:** every |ΔE| is ≤ 1.6e-10 Ha.
- **ρ_g:** 17 of 18 cells read ρ_g ≤ 3. r2SCAN paracetamol (4.09) is the exception.
- **ρ_μ:** 15 of 18 read ρ_μ ≤ 3. The exceptions are r2SCAN paracetamol (8.56), wB97M-V
  propranolol (4.89) and wB97M-V fluconazole (3.97). They read "mixed-specific residual", still
  within the ceilings.
- **Gradient prediction (≤ 1e-6 Ha/Bohr):** MODEL-MISS for 15 of 18 cells.


### Erratum (2026-10-06): found by the evidence-package review (`benchmarks/mixed_precision/native/` in the fork)

The text above is left as written. These corrections supersede it where they differ; `verify_native.py` checks each corrected number against the data.

- **Part 1:** the gradient prediction (≤ 1e-6 Ha/Bohr) is exceeded in **8** of D1's 12 cells, not 9: C0's downstream cell had been counted in. Separately, **11** of 12 D1 cells have ρ_g ≤ 3, not 10. That one is a plain miscount, since counting C0's cell (ρ_g 4.09) in would give 11 of 13.
- **Part 2:** the gradient prediction is a MODEL-MISS in **14** of 18 cells, not 15.
- **Part 1, "convergence floor".** S′ uses `init_guess='atom'`, not the default guess. What it measures is stock's dependence on the initial guess at these settings. A same-guess comparator was not recorded, and ρ uses a single S′ sample.
- **Part 1, r2SCAN paracetamol.** Its gradient residual reproduces on two pods: 1.368e-6 on C0 and 1.369e-6 on D1. It is systematic, not noise.

### Erratum 2 (2026-10-08): the same-guess stock comparator was recorded, and it changes the reading

**Found by an independent analysis of the shipped sentinels.**
- Every downstream cell ran four arms: R (tight stock reference), **S (stock, same guess and
  settings as M)**, S′ (stock from `init_guess='atom'`) and M (mixed).
- Each sentinel prints the full gradient and dipole of each arm (`RFCBENCH_CELL`), plus the S−R
  deltas (`RFCBENCH_CONTROLS`). Neither the extractor nor the RESULT read S.
- Part 1's erratum said "a same-guess comparator was not recorded". **That is wrong:** it is S.

**Recomputed from the raw vectors**, max over atoms and components:

| cell | M−R grad | **S−R grad** | **M−S grad** | M−S dipole (D) | cycles S / M |
|---|---|---|---|---|---|
| paracetamol r2SCAN | 1.37e-6 | 1.36e-6 | 9.9e-9 | 6.2e-7 | 13 / 13 |
| propranolol r2SCAN | 7.04e-7 | 7.03e-7 | 3.8e-9 | 3.0e-7 | 13 / 13 |
| celecoxib r2SCAN | 1.16e-6 | 1.15e-6 | 1.4e-8 | 5.0e-7 | 14 / 14 |
| fluconazole r2SCAN | 1.96e-6 | 1.95e-6 | 1.3e-8 | 1.1e-6 | 13 / 13 |
| warfarin r2SCAN | 6.55e-7 | 6.65e-7 | 5.3e-8 | 2.4e-6 | 15 / 15 |
| omeprazole r2SCAN | 1.49e-6 | 1.47e-6 | 2.4e-8 | 1.2e-6 | 15 / 15 |
| paracetamol B3LYP | 1.34e-6 | 1.33e-6 | 2.1e-8 | 2.0e-7 | 12 / 12 |
| propranolol B3LYP | 7.36e-7 | 7.20e-7 | 4.5e-8 | 1.0e-6 | 12 / 12 |
| celecoxib B3LYP | 1.84e-6 | 1.79e-6 | 4.6e-8 | 6.4e-7 | 13 / 13 |
| fluconazole B3LYP | 8.29e-7 | 6.88e-7 | 1.4e-7 | 9.3e-6 | 13 / 13 |
| warfarin B3LYP | 1.67e-6 | 1.70e-6 | 2.7e-7 | 9.3e-6 | 14 / 14 |
| **omeprazole B3LYP** | 1.00e-6 | **3.39e-6** | 4.4e-6 | 1.9e-4 | **13 / 14** |
| wB97M-V, six cells | 1.20e-6 to 3.10e-6 | equal to M−R to 3 s.f. | 1.2e-8 to 3.2e-8 | 1.2e-12 to 5.9e-12 | equal |

Units: gradients in Ha/Bohr, dipoles in D.

**What this means:**
- **In 17 of 18 cells, M−R and S−R point the same way**, with cosine +0.98 to +1.000. Mixed and
  stock stop at essentially the same iterate.
  - In 16 of those cells max |M−R| is within 3 % of max |S−R|.
  - In fluconazole B3LYP it is within 21 %.
  - The residual this RESULT called "mixed-specific" (ρ > 3) is **stock's own convergence-tolerance
    residual**, at `conv_tol` 1e-9 and the library-default `conv_tol_grad` √1e-9 = 3.16e-5. That
    is why r2SCAN paracetamol "reproduces across two pods": stock reproduces it.
  - ρ compares two independently oriented tolerance residuals, from different guesses, and says
    nothing about precision.
- **The precision imprint is M−S:**
  - gradients 3.8e-9 to 5.3e-8 for r2SCAN, 2.1e-8 to 2.7e-7 for B3LYP, 1.2e-8 to 3.2e-8 for
    wB97M-V;
  - dipoles ≤ 2.4e-6 D (r2SCAN), ≤ 9.3e-6 D (B3LYP), ≤ 5.9e-12 D (wB97M-V).
- **Omeprazole B3LYP is the exception, in the other direction.** Mixed took one more cycle than
  stock and landed 3.4× closer to R than stock did.
- **The one gate FAIL**, fluconazole r2SCAN's dipole (M−R 1.004e-4 D), is also the tolerance
  residual. Stock S−R is 9.93e-5 D, and M−S is 1.05e-6 D.
- **Stock itself would breach the 1e-4 D dipole ceiling** on omeprazole B3LYP (S−R 1.57e-4 D).
- **The gates and the FAIL stand as computed.** No gate is re-evaluated. What changes is the
  attribution:
  - the ceilings measure the campaign's convergence tolerance, not the mode;
  - the "gradient ≤ 1e-6" MODEL-MISS (14 of 18) is a miss about stock's convergence floor.
