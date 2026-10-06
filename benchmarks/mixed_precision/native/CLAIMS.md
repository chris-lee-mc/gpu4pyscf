# What the RFC may claim from the native-port campaign

Every number below is recomputed from `data/` by `python verify_native.py` (stdlib only; exit 0
means every check reproduced). The `[N…]` tags are its check groups. The protocol, with its
pre-registered predictions, gates and amendments, is in `prereg/`. Pods and sentinels are listed in
`SOURCES.md`.

**What was measured.**
- The fork's own port of the mode, at `63af0568` (gpu4pyscf 1.8.1 base), run as
  `mf.mixed_precision = MixedPrecision(...)`.
- Arms: `mixed`, the mode with its FP64 AO cache on; `stock`, `mixed_precision = None`. On the trio
  cells there is also `cache`, the FP64 AO cache alone with no FP32 arithmetic.
- def2-mTZVPP / def2-tzvpp-jkfit unless stated, grid level 3 unpruned, `conv_tol 1e-9`.
- R = 3 warm pairs per cell, stock always last, cuTENSOR 2.3.1 loaded and observed.
- A ratio is the stock median over the mixed median, so values above 1 mean faster.
- Readings: PAYS ≥ 1.15; NEUTRAL in between; HARMS ≤ 0.95.

## C1. On the RTX PRO 6000 (Blackwell, FP64-limited), the mode pays for every functional tested `[N1]`

| 24 drug-like molecules, 20–60 atoms | mixed/stock geomean | S / M / L tiers | per-cell range |
|---|---|---|---|
| r2SCAN | **1.657** | 1.576 / 1.665 / 1.804 | 1.523–1.871 |
| B3LYP | **1.783** | 1.736 / 1.779 / 1.897 | 1.584–1.994 |
| wB97M-V (VV10 component) | **2.914** | 2.848 / 2.995 / 2.806 | 2.593–3.250 |

- All three are in their pre-registered bands.
- The r2SCAN and B3LYP rows come from one pod (L12). The wB97M-V row is the union of five healthy
  pods.
- At a looser `conv_tol_grad = 1e-5` (R = 1), r2SCAN reads 1.586. That is a MODEL-MISS against the
  prototype's 1.78–1.84.

## C2. The gain holds for larger molecules for r2SCAN and wB97M-V, and shrinks for B3LYP `[N3]`

On six 63–98-atom molecules on the same card:
- **r2SCAN 1.931**, in band;
- **B3LYP 1.338**, a MODEL-MISS below the L tier's 1.897;
- **wB97M-V 2.813** (sildenafil) and **2.709** (atorvastatin).

Every cell kept the full `fp64+fp32` cache on the 96 GB card.

Across bases (four molecules, def2-universal-jkfit):

| | def2-SVP | def2-mTZVPP | def2-TZVP |
|---|---|---|---|
| r2SCAN | 1.693 | 1.762 | 2.004 |
| B3LYP | 1.957 | 1.744 | 1.590 |

B3LYP's gain falls with system size and basis size. The mechanism is **not** established: no stage
was timed.

## C3. The precision switch must not be on by default on FP64-strong GPUs `[N2]`

Trio geomeans (paracetamol, propranolol, celecoxib). The precision effect is mixed/cache: the gain
of FP32 arithmetic, with the cache's own gain removed.

| card | mixed/stock r2SCAN / B3LYP / wB97M-V | precision effect | cache alone |
|---|---|---|---|
| L40S | 1.729 / 1.524 / 2.994 | **1.655 / 1.463 / 2.981 (PAYS)** | 1.045 / 1.041 / 1.004 |
| H100 | 1.320 / 0.905 / 0.577 | **1.084 / 0.745 / 0.539 (NEUTRAL / HARMS / HARMS)** | 1.217 / 1.215 / 1.070 |
| A100 | not quotable (DEGRADED twice) | 11 of 12 clean cells NEUTRAL or HARMS | 1.15–1.25 on clean r2SCAN/B3LYP cells |
| RTX PRO 6000 | see C1 | wB97M-V 2.941 (C0w) | wB97M-V 1.008 |

- **Recommendation the data supports:** gate the FP32 switch on the device's FP64:FP32 throughput,
  or leave it off by default on FP64-strong cards. The FP64 AO cache can stay available there,
  because it pays about 1.2× on its own.
- **RTX 5090: no data** (n/a-by-capacity).
- **The PRO 6000's r2SCAN/B3LYP precision effect** exists only on C0, which is
  CONTENTION-UNKNOWN: 1.543 / 1.633. It is not used as a claim.

## C4. Accuracy: 17 of 18 downstream cells are within every ceiling; one dipole is 0.4 % over `[N4]`

- Ceilings: max |Δg| 1e-5 Ha/Bohr, max |Δμ| 1e-4 D, |ΔE| 1e-8 Ha.
- The reference is a tight stock SCF (`conv_tol 1e-11`, `conv_tol_grad 3e-6`).
- **Gradients:** every gradient is within the ceiling with ≥ 3× margin; the largest is 3.10e-6
  Ha/Bohr.
- **Energies:** every |ΔE| is ≤ 1.6e-10 Ha.
- **The one FAIL:** fluconazole r2SCAN's dipole, at 1.004e-4 D. It is reported as the FAIL it was.
  - Stock itself, started from a different guess (S′), reaches 1.058e-4 D on celecoxib B3LYP.
  - So that ceiling sits below stock's own convergence floor at these settings.
- **Mixed-specific residual:** three cells read ρ > 3, i.e. a deviation more than 3× stock's own
  noise. They are r2SCAN paracetamol (ρ_g 4.09, ρ_μ 8.56), wB97M-V propranolol (ρ_μ 4.89) and
  wB97M-V fluconazole (ρ_μ 3.97).
- The ≤ 1e-6 Ha/Bohr gradient prediction was a MODEL-MISS in 15 of 18 cells.

## C5. MIG: four 1g.24gb instances against one whole RTX PRO 6000 Server Edition `[N5, N6]`

MIG gain = 4 × whole-card wall / one-instance wall.

| | trio stock | trio mixed | 475–559 Da stock | 475–559 Da mixed |
|---|---|---|---|---|
| r2SCAN | 1.43 / 1.46 | 1.75 / 1.78 | 1.19–1.28 | 1.29–1.44 |
| B3LYP | 1.39 / 1.42 | 1.25 / 1.26 (celecoxib 0.92, HARMS) | 1.09–1.16 | 0.98–1.11 |
| wB97M-V | 1.03 / 1.03 | 1.19 / 1.18 | not measured | not measured |

The trio columns give the values on two separate pods, M1 / M2.

- **One instance holds the full `fp64+fp32` cache up to at least celecoxib (381 Da).** From
  sildenafil (475 Da) up it falls back to the FP32-only cache, as pre-registered. The DF tensor
  stayed on the device up to atorvastatin (559 Da). Every cell converged.
- **The gain is a projection.** It assumes four instances run at once without slowing each other.
  One instance was measured at a time; neighbouring slices belonged to other tenants. Instance walls
  agreed within 3 % across two cards and three slices.

## Not claimed

- **Gradients and Hessians run stock.** The mode does not touch them.
- **Range-separated XC and K are refused.** The VV10 component alone handles wB97M-V.
- **Not measured:** UKS, multi-GPU, the RTX 5090, MIG profiles other than 1g.24gb, and wB97M-V on a
  MIG instance above 259 Da.
- **Timings on other cards** say nothing about the RTX PRO 6000, and vice versa.

## Known discrepancies

Two RESULT texts printed a number one unit off in the last digit. `verify_native.py` reports both
as DISCREPANCY, and this file quotes the recomputed values.

| where | printed | recomputed |
|---|---|---|
| PREREG-2 part 2, L12 trio B3LYP | 1.859 | 1.8595, so 1.860 |
| PREREG-6 RESULT, atorvastatin B3LYP stock MIG gain | 1.09 | 1.0950, so 1.10 |
