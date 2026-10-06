# PREREG rfcbench-1: the native ladder on the RTX PRO 6000

**Status 2026-10-02: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`
(measurand, gates, rules). Freezes at the first `mode=ladder` sentinel. Changes go only in dated
amendments above RESULT, written before the dispatch they govern.

## Cells and pods

The 24-molecule drug ladder (`src/jqc_conformer/drugset.py` `DRUGS`, tiers S 8 / M 12 / L 4;
committed geometries, unchanged), at def2-mTZVPP / def2-TZVPP-JKFIT. Arms `mixed` and `stock`,
R = 3, card `pro6000`.

| pod | cells | est. work |
|---|---|---|
| L12 | 24 × {r2SCAN, B3LYP}, plus the bridge leg | ~3400 s |
| W1 | S + L tiers (12) × wB97M-V | ~3300 s |
| W2 | M tier (12) × wB97M-V | ~3500 s; a `remainder` dispatch is the recovery |

**Bridge leg** (L12 only). One extra R = 1 pass of the 24 r2SCAN cells at `conv_tol_grad = 1e-5`,
the setting of the prototype ladder. It is disclosed, and never pooled with the default-setting
cells.

## Predictions (disclosed; MODEL-MISS band ±15 % of the predicted value)

Ladder geomean of per-cell mixed/stock ratios:

| | prediction |
|---|---|
| r2SCAN | 1.80 |
| B3LYP | 2.05 |
| wB97M-V | 2.65 |

The basis is the g4pport AO-cache validation trio (run `36942908651`, einsum backend, single warm
pairs): 1.81 / 1.97 / 2.69. B3LYP is raised for cuTENSOR (the banked +6.8 % on its ratio,
`b3lyp_cutensor_aba_pro6000.csv`). The ratios are expected to be flat across tiers.

Prediction for the bridge leg: the r2SCAN geomean at 1e-5 falls within ±0.15 of the prototype's
banked 1.78–1.84.

## Readings

- Per xc: the tier geomeans S/M/L and the ladder geomean, each with PAYS / NEUTRAL / HARMS.
- **Falsifiers of the RFC's purpose.** Either one is stated in the RFC if it happens:
  - a tier geomean below 1.3 means "the port does not deliver the mode's purpose on its target
    card";
  - any cell FAIL on accuracy.

## Budget

Three pods: L12, W1, W2. A `remainder` dispatch of W2 is counted within the campaign's retry
allowance.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## Amendments

None yet.

## Amendment 1 (2026-10-02, before any dispatch)

- **Bridge leg.** One pair, mixed then stock, labelled pair 1, with no cold pair. It runs in its
  own group after the speed groups. It is disclosed, and excluded from every speed aggregate.
- **Splits.** L12, W1 and W2 may be split after the C0 calibration, as in PREREG-0 Amendment 1.
  The cells, arms, R and settings are unchanged, and the readings are computed over the union.

## RESULT

### RESULT, part 1 (2026-10-02): L12 **PASS**, r2SCAN and B3LYP on all 24

**The run.** Run `36968109711` attempt 1, sentinel
`ci-results/runs/rfcbench-78a3412e8629-36968109711-1-78a3412e.txt`, line 2
`RUN_ID=36968109711-1-78a3412e` (checked). Repo `78a3412e`, fork `63af0568`, on
`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`.

**Status.**
- `STATUS=PASS`, `WORK_RC=0`, 2298 s of work.
- 72/72 cells OK and TREATED.
- No NOISY cell, so the pod is not DEGRADED.
- The canary read 3.08 s, below the band, i.e. a fast pod.

**Readings** (geomean of per-cell mixed/stock, R = 3, def2-mTZVPP, cuTENSOR):

| | ladder (24) | prediction (±15 %) | reading | S (8) | M (12) | L (4) | per-cell range |
|---|---|---|---|---|---|---|---|
| r2SCAN | **1.657** PAYS | 1.80 (1.53–2.07) | in band | 1.576 | 1.665 | 1.804 | 1.523–1.871 |
| B3LYP | **1.783** PAYS | 2.05 (1.74–2.36) | in band | 1.736 | 1.779 | 1.897 | 1.584–1.994 |

- **The ratio is not flat across tiers.** The prediction said it would be; it rises with molecule size,
  from S to L, for both functionals.
- **Falsifiers:** none. Every tier geomean is ≥ 1.3, and no cell failed on accuracy.

**Bridge leg** (r2SCAN at `conv_tol_grad = 1e-5`, R = 1, disclosed). Geomean **1.586** over 24 cells,
range 1.19–1.76. Against the prototype's 1.78–1.84 (±0.15), that is a **MODEL-MISS**. With one pair
per cell it is the noisiest number here, and it is never pooled with the default-setting cells.

wB97M-V (W1–W5) follows in part 2.

### RESULT, part 2 (2026-10-02): wB97M-V on all 24, **PASS**

**Pods.** Five healthy pods, each `STATUS=PASS` and `WORK_RC=0`, RUN_ID checked, no NOISY cell, and
`contended=False` under PREREG-0 Amendment 3:

| pod | run | idle power | cells |
|---|---|---|---|
| W1 | `36971083681` | 57 W | S tier + celecoxib |
| W2 | `36973444064` | 91 W | fluconazole, warfarin, omeprazole |
| W3 | `36975697910` | 92 W | ibuprofen, naproxen, lidocaine, salbutamol, phenytoin |
| W4r | `37010453926` | 75 W | ketoprofen, diphenhydramine, propranolol, atenolol |
| W5 | `37008076314` | 75 W | metoprolol, trimethoprim, diclofenac |

Every cell is OK and TREATED. Sentinels are at
`ci-results/runs/rfcbench-78a3412e8629-<run>-1-78a3412e.txt`.

**Readings** (geomean of per-cell mixed/stock, R = 3, def2-mTZVPP, cuTENSOR; a cross-pod union, so
disclosed as such):

| | ladder (24) | prediction (±15 %) | reading | S (8) | M (12) | L (4) | per-cell range |
|---|---|---|---|---|---|---|---|
| wB97M-V | **2.914** PAYS | 2.65 (2.25–3.05) | in band | 2.848 | 2.995 | 2.806 | 2.593–3.250 |

**W4 is CONTENDED and reported separately** (run `36977987637`, idle 576 W). It passed its gates, but
its ratios were 2.102 / 1.916 / 2.134 / 2.101. The same four cells re-run on the healthy W4r gave
3.047 / 2.873 / 2.986 / 3.028. W4 is excluded from the reading above, as Amendment 3 requires, and
its sentinel and CSV are kept. The pair shows that one contended GPU lowers wB97M-V's ratio by about
30 %, with no sign in the per-cell spreads.

**Scope 1 is complete:**

| | ladder geomean | reading |
|---|---|---|
| r2SCAN | 1.657 | in band |
| B3LYP | 1.783 | in band |
| wB97M-V | 2.914 | in band |

- No tier falls below 1.3, so no falsifier is met.
- The bridge leg is a MODEL-MISS (part 1).


### Erratum (2026-10-06): found by the evidence-package review (`benchmarks/mixed_precision/native/` in the fork)

The text above is left as written. These corrections supersede it where they differ; `verify_native.py` checks each corrected number against the data.

- **Undisclosed flag.** L12's metformin B3LYP cell has a warm wall under 1 s (`under_1s`). It is retained; it was not mentioned above.
