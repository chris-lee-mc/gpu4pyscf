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

