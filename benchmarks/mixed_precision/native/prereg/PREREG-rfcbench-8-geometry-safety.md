# PREREG rfcbench-8: what sets the gradient residual, and does the mode survive a geometry optimisation

**Status 2026-10-08: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`.
It freezes at the first PREREG-8 sentinel. Changes go only in dated amendments above RESULT,
written before the dispatch they govern.

**Owner decision of 2026-10-08.**
- The owner asked for an analysis of the results, with additional mixed-precision measures for
  stable geometry optimisations.
- The principle: gradients are summed only in FP64. Similar disciplines apply: FP64 accumulators,
  FP64 tails and certificates.
- The owner approved pods for any promising path with a hypothesis.
- The analysis was an independent Fable 5.1 review. It produced the hypotheses below. Its central
  finding was checked against the raw sentinels before this protocol was written (PREREG-4
  erratum 2).

## Why

1. **The downstream residual is stock's, not the mode's.**
   - Mixed (M) and same-guess stock (S) stop at essentially the same iterate in 17 of 18 cells. The
     residual against the tight reference R (0.66–3.4e-6 Ha/Bohr) is the stock convergence-tolerance
     residual at `conv_tol` 1e-9 and the default `conv_tol_grad` √1e-9 = 3.16e-5.
   - The precision imprint M−S is 4e-9 to 2.7e-7 Ha/Bohr.
   - Untested so far:
     - whether tightening `conv_tol_grad` removes the residual for both arms;
     - whether the ~1e-8 floor seen on wB97M-V is simply object-to-object summation order (the
       same stock SCF built twice);
     - whether B3LYP's larger imprint is carried by FP32-built vectors in the DIIS subspace.
2. **No geometry optimisation has run end to end with the mode.**
   - A scanner warm-starts every step from the previous density, and the mode starts every SCF in
     FP32. Its controller needs two E_xc observations before it can switch, and the convergence
     guard then requires two clean FP64 iterations.
   - So on late optimisation steps the mode is predicted to cost cycles. It may be slower than
     stock exactly where an optimisation spends most of its steps.

## The fork commit measured

`chris-lee-mc/gpu4pyscf` @ `09271907d51580104c7a4e8334a8f65efc1b5d2a` (branch
`mixed-precision-geo`). It is the campaign commit `63af0568` plus two opt-in policy fields, both
default off; with both off, the default `repr` and every code path are unchanged:
- **`diis_reset_at_switch=True`:** restart the DIIS subspace on the first iteration built entirely
  in FP64.
- **`warm_start_gorb=τ`:** when the SCF starts from a supplied density with orbitals, and the
  orbital-gradient norm of the initial Fock is below τ, every call after the first runs in FP64.
  The record carries the measured norm.
- **Before any PREREG-8 measurement**, this commit must PASS the g4pport validation, which includes
  the new `GeometrySafety` tests. The rfcbench harness must also admit its tree.
  - That is a harness change: a set of validated trees replaces the single `VALIDATED_FORK_SHA`.
  - `63af0568` stays in the set.

Cell settings are PREREG-0's: def2-mTZVPP / def2-tzvpp-jkfit, grid level 3 unpruned, cuTENSOR,
`ao_cache_fp64=True`, and B3LYP `xc_switch_tol` 3e-4. RTX PRO 6000 Workstation (`pro6000`).

## Pods

| pod | mode | cells | what |
|---|---|---|---|
| V8 | g4pport | — | validation of `09271907` (a correctness gate; its own workflow) |
| A1 | `attrib` | 6 molecules × {r2SCAN, B3LYP} | the eight-arm attribution below |
| A2 | `attrib` | 6 molecules × wB97M-V | the same |
| G1 | `geoopt` | paracetamol r2SCAN, paracetamol B3LYP, celecoxib r2SCAN | optimisations, three arms each |

- **Order:** dispatched one at a time, in this order. Each needs nothing queued or in progress.
- **Reserve:** one pod, for an infrastructure failure only.
- **CONTENDED pods:** PREREG-0 Amendment 3 applies; re-run once, only after the owner says so.

## A1, A2: attribution (eight arms per cell)

The molecules are PREREG-4's: paracetamol, propranolol, celecoxib, fluconazole, warfarin and
omeprazole. Each arm is a fresh SCF from the default guess, followed by the dipole and the stock
analytic gradient:

| arm | SCF | conv_tol | conv_tol_grad |
|---|---|---|---|
| R′ | stock | 1e-12 | 1e-6 |
| R | stock (PREREG-4's R) | 1e-11 | 3e-6 |
| S | stock | 1e-9 | default (3.16e-5) |
| S2 | stock, repeated in the same process | 1e-9 | default |
| St | stock | 1e-9 | 3e-6 |
| M | mixed (campaign policy) | 1e-9 | default |
| Mt | mixed | 1e-9 | 3e-6 |
| Md | mixed, `diis_reset_at_switch=True` | 1e-9 | default |

Order within a cell: R′, R, S, S2, St, M, Mt, Md. Each sentinel line carries the full gradient and
dipole vectors.

**Gates** (fail the pod; fixed now):
- all eight arms present, converged, no error, gradient and dipole present;
- |E_arm − E_R′| ≤ 1e-8 Ha;
- on M, Mt and Md: an FP64 tail and the fit outcome `TREATED`;
- on Md: `diis_reset_call` recorded;
- on M and Mt: no reset.

**Predictions.** max|Δ| is over all gradient components, in Ha/Bohr.
- **P-A (the floor):** max|g_S2 − g_S| ≤ 5e-8 in every cell.
- **P-B (tolerance sets the residual):**
  - max|g_St − g_R′| ≤ 3e-7 and max|g_Mt − g_R′| ≤ 3e-7 in every cell;
  - max|g_Mt − g_St| ≤ 1e-7 in every cell.
- **P-C (the cost of tightening):**
  - cycles: Mt = St ± 1;
  - **disclosed**: Mt's wall over M's, and St's over S's. Single-run walls are not speed readings.
- **P-D (DIIS carries the B3LYP imprint):**
  - max|g_Md − g_S| ≤ 3e-8 in every B3LYP cell except omeprazole (where M and S stop on different
    cycles);
  - cycles: Md = M ± 1 everywhere.
- **P-E (reproduction):** max|g_M − g_S| reproduces PREREG-4 erratum 2's value per cell within a
  factor of 3, or within 3e-8 absolute.

**Falsifiers:**
- more than 3 cells with max|g_Mt − g_R′| > 1e-6: the residual is not set by the tolerance, and
  PREREG-4 erratum 2's reading is wrong;
- in more than half the B3LYP cells, max|g_Md − g_S| is within a factor of 2 of max|g_M − g_S|:
  DIIS does not carry the imprint, and the stopping iterate does;
- max|g_S2 − g_S| > 1e-7 in any cell: the r2SCAN and wB97M-V "imprints" were never resolvable.

## G1: geometry optimisations through the scanner

**Setup.**
- geomeTRIC through `pyscf.geomopt.geometric_solver.kernel` on a gradient scanner, as in the
  engine's l3b work.
- Convergence set `convergence_energy=1e-6, grms=3e-4, gmax=4.5e-4, drms=1.2e-3, dmax=1.8e-3`.
- `maxsteps` 90.
- Start geometries: the campaign's frozen ones.
- SCF settings as the campaign.

**Arms per cell, in this order:**
- **M0:** the campaign policy, with `warm_start_gorb=1e-300`. That threshold cannot trigger. It
  only records each step's initial orbital-gradient norm, at the cost of one extra Fock build and
  one orbital-gradient evaluation per step.
- **MW:** the campaign policy with `warm_start_gorb=1e-2`. This τ is fixed now, before any data.
- **S:** stock.

**Per step, recorded:**
- SCF cycles;
- the XC (and K) precision list;
- the warm-start record;
- SCF wall and gradient wall, synchronised;
- the energy.

**Per optimisation:** converged, steps, final coordinates. At each mixed endpoint, a fresh stock
SCF and stock gradient certify it.

**Gates:**
- every optimisation converges within 90 steps;
- at each M0 and MW endpoint, the stock gradient meets grms ≤ 3e-4 and gmax ≤ 4.5e-4, and
  |E_stock(endpoint) − E_S(final)| ≤ 1e-6 Ha;
- every SCF of every step converged, and every mixed SCF ended on an FP64 tail.

**Predictions** (steps ≥ 1, i.e. warm-started; step 0 is cold):
- **P-F (the warm-start penalty):** on M0, cycles ≥ S's cycles + 1 on at least 30 % of warm steps,
  and on at least 80 % of the warm steps where S converges in ≤ 5 cycles.
- **P-G:**
  - M0's summed SCF wall over warm steps, stock/mixed, lies in 0.9–1.3. Cold, the same cells read
    1.66–1.78.
  - MW's lies in 0.95–1.15.
  - On MW, cycles ≤ S's + 0 on at least 90 % of warm steps.
- **P-H (endpoints):** Kabsch RMSD of M0 and MW against S is ≤ 9.19e-3 Å (r2SCAN) and ≤ 5.88e-4 Å
  (B3LYP). These are the banked stock-control envelopes of PREREG-l3b-geo-endpoint and
  PREREG-l3b-xf-repeat.
- **Disclosed:**
  - each warm step's initial orbital-gradient norm on M0;
  - the fraction of MW steps that triggered;
  - steps to convergence per arm.

**Falsifier:** if M0 shows cycle parity with S on warm steps, and its warm-step wall ratio is ≥ 1.5,
then the warm-start concern is wrong and the rule is unnecessary.

## Not measured

- An `xc` gradient treatment. The gradients here run stock and FP64 end to end, which is the
  owner's principle.
- Tight geomeTRIC convergence sets.
- Molecules larger than celecoxib in G1.
- Cards other than the PRO 6000 Workstation.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## Amendments

### Amendment 1 (2026-10-08, before any A1/A2/G1 dispatch)

**V8 PASSED.**
- Run `37725279648` (g4pport) validated fork `09271907d51580104c7a4e8334a8f65efc1b5d2a` on an RTX
  PRO 6000 Workstation, with problems `[]`.
- Upstream regression: 42/42 stock and 42/42 with the branch.
- New tests: **55/55**, the 49 earlier ones plus the 6 `GeometrySafety` tests.
- VV10 identity: bitwise.
- Trio: \|ΔE\| ≤ 8.2e-12 Ha, with identical cycles in all 9 cells and every certificate ok.
- Sentinel `ci-results/g4pport-220b9f0213a3.txt`, `RUN_ID=37725279648-1-220b9f02`.

**A2 is reduced to wB97M-V on paracetamol and propranolol.**
- Six wB97M-V cells × eight arms is estimated at 8.8–13.4 ks, against the 4140 s work cap. PREREG-4's
  wB97M-V SCF walls alone come to about 3.3 ks for eight arms, so this is a real overrun, not
  padding.
- Splitting A2 into five pods would exceed the approved pod count. Instead A2 keeps the two smallest
  PREREG-4 molecules (estimate 3150 s). The other four wB97M-V cells are **not measured**.
- A1 (3292 s) and G1 (3950 s) are unchanged.
- **G1's celecoxib step count has never been measured.** If it overruns, only its own cell is lost;
  the paracetamol cells run first.

**How the harness reads this protocol, fixed now:**
- **R′** uses `max_cycle` 100.
- **Record format.** The warm-start record is exactly
  `{'tol', 'supplied', 'gorb', 'fp64'}`, as written by fork `09271907`. M0's per-step norms are
  `gorb`, and MW's trigger is `fp64`.
- **G1 timing.** The SCF and the gradient of every step are each timed between two device
  synchronisations, by wrapping the scanner's SCF `kernel` and the gradient `kernel` on the
  instance.
- **Extra gates.** The harness adds PREREG-0's observation gates to A1/A2 and G1:
  - settings read back on every arm;
  - the recorded policy repr equals the class's repr for that arm's variant;
  - stock arms carry no policy;
  - grid and basis sizes identical across arms;
  - no mixed state alive before the gradient;
  - the treatment and cache checks on M, Mt and Md.

  No prediction or band changes.

## RESULT

### RESULT (2026-10-08): **PASS** on A1, A2 and G1. Tightening `conv_tol_grad` sets the residual, and mixed tracks stock; DIIS restart is rejected; there is no warm-start penalty, and the warm-start rule never triggers

**Pods.** Each sentinel's line 2 matches its run. Each pod ran alone from `main` @ `5013b101`, with
fork `09271907`, on an RTX PRO 6000 Workstation. All three are `STATUS=PASS` with every gate held.

| pod | run | wall | cells |
|---|---|---|---|
| A1 | `37734904781` | 1271 s | 12, all OK / TREATED |
| A2 | `37736924342` | — | 2, all OK / TREATED |
| G1 | `37738477503` | 1010 s of groups | 3, all OK / TREATED |

- **A1's first dispatch**, run `37728120899`, produced no sentinel: the runner gave up at 4815 s.
  Under #139's caps a pod that started on time writes a FAIL within its 4440 s work cap, so the
  pod started late or never ran its work. It was treated as an infrastructure failure and
  re-dispatched once, unchanged, on the pre-registered reserve.
- **Sentinels** are banked in `rfcbench-data/prereg8/{a1,a2,g1}.txt`.

#### A1, A2: what sets the residual

Gradient ranges are max over atoms and components, in Ha/Bohr, across each functional's cells:

| | r2SCAN (6) | B3LYP (6) | wB97M-V (2) |
|---|---|---|---|
| S2−S (stock repeated, same process) | 5.1e-13 – 1.8e-12 | 5.4e-9 – 2.9e-8 | 1.3e-8 – 2.0e-8 |
| S−R′ (default tolerance) | 6.7e-7 – 1.9e-6 | 5.1e-7 – 3.4e-6 | 1.3e-6 – 3.2e-6 |
| M−R′ | 6.6e-7 – 2.0e-6 | 6.3e-7 – 1.8e-6 | 1.3e-6 – 3.2e-6 |
| **St−R′** (`conv_tol_grad` 3e-6) | **8.3e-8 – 3.2e-7** | **1.3e-7 – 2.6e-7** | **7.4e-7 – 7.5e-7** |
| **Mt−R′** | **8.2e-8 – 3.2e-7** | **1.7e-7 – 2.8e-7** | **7.5e-7** |
| **Mt−St** | **1.5e-9 – 2.3e-8** | **6.8e-9 – 2.2e-7** | **1.1e-8 – 2.1e-8** |
| M−S | 4.0e-9 – 5.2e-8 | 2.3e-8 – 4.5e-6 | 1.6e-8 |
| Md−S (DIIS restart) | 1.1e-6 – 3.2e-6 | 7.4e-7 – 3.8e-6 | 1.7e-6 – 2.2e-6 |
| extra cycles, St over S | 2–3 | 1–3 | 1–2 |
| stock/mixed wall, S/M → St/Mt (geomean, single runs) | 1.71 → 1.65 | 1.94 → 1.85 | 2.80 → 3.01 |

**P-A, the floor: met.**
- Stock r2SCAN repeats to ≤ 1.8e-12.
- Stock B3LYP and wB97M-V repeat only to 3e-8. This is consistent with `atomicAdd` on their DF-K or
  VV10 paths; that cause is not established here.
- So mixed's r2SCAN imprint, M−S 4e-9 to 5.2e-8, sits well above its floor. The wB97M-V imprint is
  at its floor and not resolvable.

**P-B, the tolerance sets the residual: met in substance, missed narrowly.**
- Tightening `conv_tol_grad` from the default 3.16e-5 to 3e-6 cuts the residual against R′ by
  1.8–22× per cell (stock) and 1.8–10× (mixed), to the same level for both.
- Three of 14 cells exceed the predicted 3e-7, all on both arms: paracetamol r2SCAN (3.2e-7) and
  the two wB97M-V cells (7.5e-7).
- Mt−St ≤ 1e-7 is missed only by omeprazole B3LYP (2.2e-7).
- **The falsifier did not fire:** no cell is above 1e-6. PREREG-4 erratum 2's reading stands.

**P-C, cycles: met.** Mt equals St in every cell. Tightening costs stock and mixed the same 1–3
cycles. Mixed keeps its single-run wall advantage at the tighter tolerance.

**P-D, DIIS restart: MODEL-MISS, and rejected.**
- Md−S is 10–100× worse than M−S in every functional, r2SCAN included. Restarting the subspace
  changes the trajectory, and the SCF stops at a different iterate within the loose tolerance.
- B3LYP Md costs +1 cycle in 4 of 6 cells.
- The falsifier, as worded ("within 2×"), did not fire, because the effect went the other way.
- **`diis_reset_at_switch` is not recommended**, and it stays off by default.

**P-E, reproduction: met.** M−S reproduces PREREG-4 erratum 2's per-cell values, for example
9.3e-9 against 9.9e-9 and 4.5e-6 against 4.4e-6.

**Dipoles, at `conv_tol_grad` 3e-6:**
- Mt−St ≤ 1.4e-6 D (r2SCAN), ≤ 8.7e-6 D (B3LYP), ≤ 4.5e-12 D (wB97M-V).
- St−R′ is up to 1.9e-5 D.

#### G1: geometry optimisations

| cell | steps M0 / MW / S | endpoint RMSD vs S (Å) M0 / MW | endpoint stock grms / gmax (M0) | warm-step SCF wall, stock/mixed | whole optimisation, stock/MW |
|---|---|---|---|---|---|
| paracetamol r2SCAN | 23 / 23 / 23 | 9.9e-6 / 9.4e-6 | 7.2e-5 / 2.1e-4 | 1.41 | 1.28 |
| paracetamol B3LYP | 26 / 26 / 26 | 1.5e-6 / 3.5e-6 | 5.8e-6 / 1.8e-5 | 1.42 | 1.19 |
| celecoxib r2SCAN | 16 / 16 / 16 | 4.0e-7 / 4.1e-7 | 6.8e-5 / 2.3e-4 | 1.44 | 1.30 |

**Gates held.**
- Every optimisation converged.
- Every mixed endpoint passes the stock certificate, with |E − E_S| ≤ 3.2e-9 Ha.
- Every SCF converged, and every mixed SCF ended on an FP64 tail.

**P-F, the warm-start penalty: FALSIFIED.**
- M0's SCF cycles equal S's on **every** warm step of all three optimisations, apart from one
  B3LYP step at −1. Warm steps take 4–11 cycles in both arms.
- No warm step converges in ≤ 5 cycles in the r2SCAN cells.
- The analysis's premise was that a scanner's warm start is near-converged. It does not hold here,
  as the next point shows.

**MW never triggered (P-G's MW band not testable).**
- The orbital-gradient norm of the initial Fock on warm steps is 0.014–43 (M0's record). The
  previous geometry's orbitals are a poor start in the new geometry's metric.
- τ = 1e-2 was met on no step, and MW ran identically to M0.
- The rule is moot for this scanner. It stays off by default and is not recommended.

**P-G, wall ratios: MODEL-MISS, in the mode's favour.**
- M0's warm-step SCF stock/mixed is 1.41–1.44, above the predicted 0.9–1.3.
- Over the whole optimisation, SCF plus stock gradient, stock/MW is 1.19–1.30. The gradient is
  26 % (r2SCAN) and 50 % (B3LYP) of stock's optimisation time.
- In paracetamol r2SCAN, M0 ran first in its process and paid the cold driver JIT (CS1) in its
  first SCF. That optimisation's M0 total is therefore not a speed reading; MW's is.

**P-H, endpoints: met.** M0 and MW land 4e-7 to 1e-5 Å from stock, which is at least 168× (B3LYP)
and 928× (r2SCAN) inside the banked stock-control envelopes. The step counts are identical.

**What follows for the RFC:**
- **For geometry optimisation, set `conv_tol_grad` ≈ 3e-6.** It cuts the gradient residual by up to
  10–22× (at least 1.8×), to the same level for stock and mixed, costs both the same 1–3 cycles, and leaves mixed's advantage intact.
- **The mode runs through a real optimisation with stock's trajectory:** the same steps, the same
  cycles, and endpoints within 1e-5 Å. Its SCF gain on warm steps is 1.41–1.44 on this card.
- **Gradients run stock, in FP64 end to end.** That is unchanged.
- **The two new options do not help**, and neither enters the RFC.
