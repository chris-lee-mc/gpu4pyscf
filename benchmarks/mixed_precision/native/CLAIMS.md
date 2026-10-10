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
also make the AO cache larger, which moves the MIG fit limits. Call the table above the **campaign
configuration**. **Only C7 measures the library defaults** (the default grid with `nwchem_prune`,
`ao_cache_fp64=False`, the default tolerances, and the README's own `MixedPrecision(xc=True)`,
`(xc=True, k=True)` and `(vv10=True)`), on three molecules and one card: r2SCAN 1.577, B3LYP 1.636,
wB97M-V 3.094. Every other number in this file is in the campaign configuration. On the trio the
defaults read lower than the campaign configuration for r2SCAN and B3LYP (by 0.10 and 0.33) and
higher for wB97M-V (by 0.15); beyond the trio, how the two compare is not measured.

**Method.**
- R = 3 warm pairs per cell, plus one cold pair that is reported but excluded.
- Within each pair the arms run in the order mixed, cache, stock, so stock runs last with a warm
  memory pool.
- A ratio is stock median / mixed median: above 1 means mixed is faster.
- Readings: PAYS ≥ 1.15; NEUTRAL in between; HARMS ≤ 0.95.
- Within a healthy, non-DEGRADED pod, the spread of the R = 3 walls is mostly ≤ 3 %. On the
  DEGRADED replicates (L12r, L12r2) and on X3r it is not.
- **Between pods.**
  - Healthy PRO 6000 Workstation pods agree closely, to about 1 % on stock walls and about 6 % on
    ratios. For example, paracetamol wB97M-V stock is 19.07 s on W1 against 18.94 s on C0w.
    **That does not hold on the H100.** Two healthy H100 pods, X3 and X3r, differ by 4–34 % on
    every wall, and by about 25 % on stock (see C3).
  - Including the CONTENTION-UNKNOWN C0 or the Server Edition, readings span 10–20 % or more, and
    stock walls 15–37 %. For example, the PRO 6000 B3LYP trio is 1.688 on C0, 1.860 on L12 and
    2.019 on M0.
  - PREREG-7 re-ran L12 (r2SCAN, B3LYP) twice and the H100 once:
    - L12's two replicates were both DEGRADED and drew the same physical card. The pre-registered
      test therefore cannot be run on them; a post-hoc comparison on clean cells is consistent
      with L12 (C1).
    - On the H100, B3LYP's mixed/stock moved by 0.115, past the ±0.10 band.
  - Treat the third decimal as reproducibility of the data, not of the hardware.

## C1. RTX PRO 6000 (Blackwell, FP64-limited): the mode pays for every functional tested `[N1]`

**Scope: the campaign configuration** (FP64 AO cache on, unpruned grid, B3LYP `xc_switch_tol=3e-4`;
the table at the top), one card family (RTX PRO 6000 Workstation), one pod for r2SCAN and B3LYP.
The same mode at the **library defaults** is C7, on the trio only.

| 24 drug-like molecules, 20–44 atoms, campaign configuration | mixed/stock geomean | S / M / L tiers | per-cell range | pods |
|---|---|---|---|---|
| r2SCAN | **1.657** | 1.576 / 1.665 / 1.804 | 1.523–1.871 | 1 (L12), replicated below |
| B3LYP | **1.783** | 1.736 / 1.779 / 1.897 | 1.584–1.994 | 1 (L12), replicated below |
| wB97M-V (VV10 component) | **2.914** | 2.848 / 2.995 / 2.806 | 2.593–3.250 | 5 healthy (W1–W3, W4r, W5), on 3 cards |

- All three are inside their pre-registered ±15 % bands. Bands that wide make "in band" a weak
  test.
- **Replication: not testable as pre-registered; a post-hoc comparison is consistent** (PREREG-7,
  `[N8]`).
  - L12 was re-run twice, as L12r and L12r2. Both were DEGRADED (19 and 21 of 48 cells NOISY).
  - Under PREREG-0 §3 a DEGRADED pod's aggregates are not quotable. So the pre-registered test,
    pod geomean against L12's within ±0.10, cannot validly be run. The all-24 geomeans are shown
    only as UNQUOTABLE.
  - The comparison below is **post hoc**. It restricts L12 to the same clean cells, so it compares
    like with like:

  | | replicate, clean cells (n) | L12, same cells | difference |
  |---|---|---|---|
  | r2SCAN, L12r | 1.717 (16) | 1.660 | +0.057 |
  | r2SCAN, L12r2 | 1.734 (16) | 1.671 | +0.062 |
  | r2SCAN, clean on both, L12r / L12r2 | 1.709 / 1.720 (11) | 1.660 | +0.049 / +0.060 |
  | B3LYP, L12r | 1.872 (13) | 1.799 | +0.074 |
  | B3LYP, L12r2 | 1.873 (11) | 1.787 | +0.086 |
  | B3LYP, clean on both, L12r / L12r2 | 1.830 / 1.818 (5) | 1.759 | +0.072 / +0.059 |

  UNQUOTABLE all-24 geomeans: r2SCAN 1.721 / 1.718, B3LYP 1.844 / 1.868.

  - Every replicate reads higher than L12 on the same cells, by 0.05–0.09.
  - Clean-cell ratios on the replicates span 1.57–2.02 (r2SCAN) and 1.56–2.10 (B3LYP).
  - The replicates' bridge legs (R = 1, `conv_tol_grad = 1e-5`) read 1.643 and 1.658, against
    L12's 1.586.
  - Both replicates and ST1 ran on one physical card, so this is two pods, not two cards.
  - L12 had no NOISY cell. On the replicates, the NOISY arms are 28 mixed against 17 stock (L12r 16
    vs 7, L12r2 12 vs 10). The cause is unexplained.
  - **The headline stays L12's 1.657 and 1.783.** The replicates do not contradict it, but they
    cannot replace it.
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

An Amdahl decomposition of one pod, not an established cause:
- **The share shift.** J/K grows from 9 % to 50 % of the stock SCF. Add DF build, eigensolver and
  "other" (14–16 % throughout) and at atorvastatin about two-thirds of the stock SCF is accelerated
  by at most 7 %.
- **The share shift explains only part of the drop.** Whole-SCF stock/mixed falls from 1.854
  (celecoxib) to 1.288 (sildenafil) and 1.360 (atorvastatin). Holding celecoxib's stage ratios
  fixed and moving only the shares (the untimed remainder held unaccelerated), those would be 1.748
  and 1.671. So the share shift explains about 19 % and 37 % of the drop.
- **The rest is unexplained:**
  - the FP32 J/K speedup is not monotonic in size: 0.95, 1.50, 1.05, 1.07;
  - at sildenafil the XC switch came early (8 FP32 iterations, against 10–11 elsewhere), which is
    why its XC ratio is 1.99.
- **Recorded, and relevant.** The J/K stage contains J, which is always FP64. Its K part ran in FP32
  in only 8–9 of 13–15 iterations, so most of that stage's time is FP64 by construction.
- What the data support: at 475–559 Da the FP32 J/K path buys 5–7 %, and XC still buys 2–3×.
- r2SCAN has no exact exchange, so its J/K stage is J only, always in FP64. XC is 85–86 % of its
  stock SCF, which fits r2SCAN's gain holding up.
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
  - r2SCAN: mixed/stock still reads 1.32–1.40 on the H100, almost all of it from the FP64 AO
    cache. The FP32 switch itself is NEUTRAL (1.084 on both pods).
- **H100 B3LYP mixed/stock is a range, 0.905–1.021 (HARMS to NEUTRAL).** The replication moved it by
  0.115, past the ±0.10 band, so no single-pod headline is quoted.
  - The *precision effect* (mixed/cache) replicated to within 0.024 on all three functionals.
  - **X3r was a slower pod.** Every wall was 4–34 % slower than X3's, and stock slowed the most:
    by 1.25 / 1.25 / 1.12 by functional, against 1.18 / 1.14 / 1.10 for cache and 1.18 / 1.11 /
    1.06 for mixed. So every ratio over stock rose, and mixed/cache barely moved.
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
  1.15–1.34 on these cards **for r2SCAN and B3LYP only**. The high end is X3r, the slow-stock pod.
  For wB97M-V it is NEUTRAL: 1.070–1.092 on the H100.
- **No data:** the RTX 5090 (n/a-by-capacity, two attempts of a pre-registered three) and the PRO
  6000's r2SCAN/B3LYP precision effect, which exists only on the CONTENTION-UNKNOWN C0 (1.543 /
  1.633) and is not claimed.
- The L40S row rests on a single pod per cell.

## C4. Accuracy: mixed converges to stock's own iterate; the precision imprint is 4e-9 to 2.7e-7 Ha/Bohr `[N4, N7, N9, N10]`

**Setup.** Each downstream cell ran four arms:
- R, a tight stock reference (`conv_tol 1e-11`, `conv_tol_grad 3e-6`);
- **S, stock with M's own guess and settings**;
- S′, stock from `init_guess='atom'`;
- M, mixed.

The campaign SCFs use `conv_tol 1e-9` and the library-default `conv_tol_grad` √1e-9 = 3.16e-5.

**The comparison that isolates precision is M against S.** Before 2026-10-08 this section compared
M with R. The S arm was recorded in every sentinel but never read (see the ledger).

| | gradient, max \|M−S\| (Ha/Bohr) | dipole, max \|M−S\| (D) |
|---|---|---|
| r2SCAN (6 cells) | 3.8e-9 – 5.3e-8 | 3.0e-7 – 2.4e-6 |
| B3LYP (5 cells) | 2.1e-8 – 2.7e-7 | 2.0e-7 – 9.3e-6 |
| wB97M-V (6 cells) | 1.2e-8 – 3.2e-8 | 1.2e-12 – 5.9e-12 |
| omeprazole B3LYP | 4.4e-6 | 1.9e-4 |

- **In 17 of 18 cells, M and S stop at essentially the same iterate.**
  - M−R and S−R point the same way (cosine +0.98 to +1.000).
  - Their magnitudes agree within 3 % in 16 cells, and within 21 % in fluconazole B3LYP.
  - What remains between them, M−S, is the precision imprint.
- **The wB97M-V cells pin a floor.** There the densities agree to 1e-12 D, yet the gradients still
  differ by 1–3e-8. So about 1e-8 Ha/Bohr is the agreement floor between two independently built
  SCF objects. The r2SCAN imprint sits at that floor; B3LYP's (FP32 K as well as XC) is above it,
  up to 2.7e-7.
- **Omeprazole B3LYP is the exception, in the other direction.** M ran one more cycle than S (14 vs
  13) and landed **3.4× closer to R** than stock did: 1.00e-6 against 3.39e-6.
- **Against R, both arms carry the same convergence-tolerance residual:**
  - gradients 6.6e-7 – 3.4e-6 Ha/Bohr;
  - dipoles up to 1.57e-4 D for stock S (omeprazole B3LYP) and 1.004e-4 D for M (fluconazole
    r2SCAN).
  - That residual is set by `conv_tol_grad`, not by precision. Tightening `conv_tol_grad` is what
    reduces it, for stock and mixed alike.
- **Gates, as computed and unchanged:**
  - ceilings are max |Δg| 1e-5 Ha/Bohr, max |Δμ| 1e-4 D and |ΔE| 1e-8 Ha, all against R;
  - every |ΔE| is ≤ 1.6e-10 Ha;
  - **one FAIL:** fluconazole r2SCAN's dipole, at 1.004e-4 D. It is the tolerance residual: stock
    S−R is 9.93e-5 D, and M−S is 1.05e-6 D.
  - The FAIL makes D1 `STATUS=FAIL`, so all 12 of D1's numbers are **UNQUOTABLE** under PREREG-0
    Amendment 1. They are shown here with that label.
  - Stock S would itself breach the dipole ceiling on omeprazole B3LYP.
- **Withdrawn:**
  - the "mixed-specific residual (ρ > 3)" readings;
  - the statement that r2SCAN paracetamol's 1.37e-6 residual is a systematic property of the mode.
    It reproduces across pods because stock reproduces it (S−R 1.36e-6; M−S 9.9e-9 on D1, 9.0e-9 on
    C0).
  - ρ compared two independently oriented tolerance residuals (cosine of S′−R against S−R ranges
    from −0.79 to +0.88). It says nothing about precision.
- **At a tighter `conv_tol_grad` (3e-6), mixed tracks stock to near the floor** (PREREG-8 A1/A2,
  14 cells). Each cell ran eight arms; R′ is stock at `conv_tol` 1e-12 and `conv_tol_grad` 1e-6.

  | Ha/Bohr | r2SCAN (6) | B3LYP (6) | wB97M-V (2) |
  |---|---|---|---|
  | stock vs stock, repeated in one process | ≤ 1.8e-12 | 5.4e-9 – 2.9e-8 | 1.3e-8 – 2.0e-8 |
  | stock vs R′, default `conv_tol_grad` | 6.7e-7 – 1.9e-6 | 5.1e-7 – 3.4e-6 | 1.3e-6 – 3.2e-6 |
  | stock vs R′, `conv_tol_grad` 3e-6 | 8.3e-8 – 3.2e-7 | 1.3e-7 – 2.6e-7 | 7.4e-7 – 7.5e-7 |
  | mixed vs R′, `conv_tol_grad` 3e-6 | 8.2e-8 – 3.2e-7 | 1.7e-7 – 2.8e-7 | 7.5e-7 |
  | **mixed vs stock, `conv_tol_grad` 3e-6** | **1.5e-9 – 2.3e-8** | **6.8e-9 – 2.2e-7** | **1.1e-8 – 2.1e-8** |

  - **Tightening cuts the residual** by 1.8–22× per cell for stock and 1.8–10× for mixed, to the
    same level for both.
  - **It costs both arms the same 1–3 cycles**: the mixed and stock cycle counts are identical in
    every cell.
  - **The single-run stock/mixed wall ratio holds**: 1.71 → 1.65 (r2SCAN), 1.94 → 1.85 (B3LYP).
    These are not paired repeats.
  - **Stock's own floor.** Stock B3LYP and wB97M-V do not repeat bit-for-bit (≤ 3e-8); stock r2SCAN
    does (≤ 1.8e-12). So the wB97M-V imprint is not resolvable from stock's own scatter.
  - **Dipoles**, mixed vs stock at the tight tolerance: ≤ 1.4e-6 D (r2SCAN), ≤ 8.7e-6 D (B3LYP),
    ≤ 4.5e-12 D (wB97M-V).
  - **Recommendation for gradients and properties:** set `conv_tol_grad` ≈ 3e-6. The default
    √`conv_tol` leaves a 1e-6-scale residual for stock and mixed alike.
- **Restarting DIIS at the FP64 switch does not help.** It moves the stopping iterate, which lands
  7.4e-7 – 3.8e-6 from stock. The fork option `diis_reset_at_switch` stays off and is not proposed.
- **Not covered.** The mode does not touch gradients or Hessians, which run stock, in FP64.

## C6. Geometry optimisation: the mode follows stock's trajectory `[N10]`

**Setup** (PREREG-8 G1, one pod):
- paracetamol r2SCAN, paracetamol B3LYP and celecoxib r2SCAN;
- geomeTRIC through the gradient scanner, at its default convergence set;
- campaign SCF settings, stock gradients;
- arms: mixed, mixed with a warm-start rule, and stock.

| cell | steps (mixed / stock) | endpoint RMSD vs stock (Å) | warm-step SCF wall, stock/mixed | whole optimisation, stock/mixed |
|---|---|---|---|---|
| paracetamol r2SCAN | 23 / 23 | 9.9e-6 | 1.41 | 1.28 |
| paracetamol B3LYP | 26 / 26 | 1.5e-6 | 1.42 | 1.19 |
| celecoxib r2SCAN | 16 / 16 | 4.0e-7 | 1.44 | 1.30 |

- **The same trajectory.** SCF cycles are equal on every step, except the last B3LYP step (mixed 4,
  stock 5).
- **Every mixed endpoint passes a stock certificate:** a fresh stock SCF and gradient give grms
  ≤ 7.2e-5 and gmax ≤ 2.3e-4 Ha/Bohr, and |ΔE| vs stock's final energy ≤ 3.2e-9 Ha.
- **The whole-optimisation gain is lower than the SCF gain** because the gradient runs stock. The
  gradient is 26 % of stock's optimisation time for r2SCAN and 50 % for B3LYP.
- **No warm-start penalty.**
  - The predicted cost of starting each SCF in FP32 from the previous geometry's density did not
    appear (P-F falsified). Warm steps take 4–11 cycles in both arms.
  - The initial orbital-gradient norm on warm steps is 0.014–43. The previous geometry's orbitals
    are not a near-converged start, so the fork's opt-in warm-start rule (τ = 1e-2) never
    triggered. It stays off and is not proposed.
- **Scope.**
  - Three optimisations on one card, at the default convergence set, with `conv_tol_grad` at the
    library default.
  - Wall ratios come from one run each.
  - In paracetamol r2SCAN, the plain-mixed arm ran first and paid the cold driver JIT on its first
    SCF, so its whole-optimisation total is not a speed reading. The table quotes the
    warm-start-rule arm, whose steps were identical.

## C7. Library defaults (the trio, one card): the mode still pays; less than the campaign configuration for r2SCAN and B3LYP, more for wB97M-V `[N11]`

**Setup** (PREREG-9, pods LD1 and LD2, two RTX PRO 6000 Workstation pods, fork tree `63af0568`, the
1.8.1-overlay build, as for every speed number here):
- paracetamol, propranolol and celecoxib × r2SCAN, B3LYP and wB97M-V; def2-mTZVPP /
  def2-tzvpp-jkfit; `conv_tol 1e-9`; R = 3 warm pairs plus a cold pair, as in C1;
- **five arms per pair**, every mixed arm before both stock arms (stock last, with the warm pool):

  | arm | grid | policy |
  |---|---|---|
  | `mixed-default` | **the library default**: level 3, `nwchem_prune`, `mf.grids` untouched | **the README's**: `MixedPrecision(xc=True)` / `(xc=True, k=True)` / `(vv10=True)`, every other field at its default (`ao_cache_fp64=False`, `xc_switch_tol=1e-3`) |
  | `mixed-nocache-campaign` | campaign: level 3, `prune = None` | the campaign policy without the cache (`ao_cache_fp64=False`; B3LYP `xc_switch_tol=3e-4`) |
  | `mixed-campaign` | campaign | the campaign policy, as in C1 (`ao_cache_fp64=True`) |
  | `stock-default` | the library default | none |
  | `stock-campaign` | campaign | none |

- A reading pairs each mixed arm with the stock arm **on the same grid**, stock median / mixed
  median over the warm pairs. The three readings per cell are the library-default number, the FP32
  switch without the cache, and a same-pod **anchor** to C1.
- Policies, grid level and prune function, `ngrids`, cache tiers and switch calls were read back off
  `mf` on every arm (observation, not labels); every cell is `OK` and `TREATED`.

| trio geomean | **`mixed-default/stock-default`** (library defaults) | `mixed-nocache-campaign/stock-campaign` (no cache, campaign grid) | `mixed-campaign/stock-campaign` (the anchor) | C1's 24-molecule campaign figure, for reference |
|---|---|---|---|---|
| r2SCAN | **1.577** | 1.627 | 1.678 | 1.657 |
| B3LYP | **1.636** | 1.954 | 1.968 (REPLICATION-FLAG) | 1.783 |
| wB97M-V | **3.094** | 2.884 | 2.944 (REPLICATION-FLAG) | 2.914 |

Per cell:

| cell | `mixed-default/stock-default` | `mixed-nocache-campaign/stock-campaign` | `mixed-campaign/stock-campaign` |
|---|---|---|---|
| paracetamol r2SCAN | 1.505 | 1.597 | 1.645 |
| propranolol r2SCAN | 1.570 | 1.608 | 1.650 |
| celecoxib r2SCAN | 1.661 | 1.677 | 1.742 |
| paracetamol B3LYP | 1.561 | 1.992 (REPLICATION-FLAG) | 1.962 (NOISY, warm walls under 1 s; REPLICATION-FLAG) |
| propranolol B3LYP | 1.728 | 1.932 (REPLICATION-FLAG) | 1.958 (REPLICATION-FLAG) |
| celecoxib B3LYP | 1.622 | 1.939 (REPLICATION-FLAG) | 1.985 (REPLICATION-FLAG) |
| paracetamol wB97M-V | 3.162 | 2.905 (REPLICATION-FLAG) | 3.016 (REPLICATION-FLAG) |
| propranolol wB97M-V | 3.258 | 3.055 (REPLICATION-FLAG) | 3.095 (REPLICATION-FLAG) |
| celecoxib wB97M-V | 2.875 | 2.703 (REPLICATION-FLAG) | 2.734 (REPLICATION-FLAG) |

- Every one of the 27 readings is PAYS (≥ 1.15). The per-cell ranges at the defaults are
  1.505–1.661 (r2SCAN), 1.561–1.728 (B3LYP) and 2.875–3.258 (wB97M-V).
- **The pre-registered falsifier did not fire.** It was "r2SCAN `mixed-default/stock-default` below
  1.15 on a quotable trio"; the trio reads 1.577, on two pods neither DEGRADED nor CONTENDED. The
  RFC may keep its framing, but must state the library-default numbers beside the campaign ones.
- **Predictions**, fixed before dispatch:

  | | prediction (band) | measured | |
  |---|---|---|---|
  | P-1 r2SCAN, defaults | 1.50 (1.35–1.65) | 1.577 | MET |
  | P-2 B3LYP, defaults | 1.45 (1.25–1.65) | 1.636 | MET |
  | P-3 wB97M-V, defaults | 3.05 (2.85–3.35) | 3.094 | MET |
  | P-4 r2SCAN, no cache | 1.57 (1.45–1.70) | 1.627 | MET |
  | P-4 B3LYP, no cache | 1.80 (1.65–1.90) | 1.954 | **MODEL-MISS, above the band** |
  | P-4 wB97M-V, no cache | 2.78 (2.60–3.10) | 2.884 | MET |

  - P-3's same-pod corollary held in all three wB97M-V cells: `mixed-default/stock-default` ≥
    `mixed-campaign/stock-campaign`. Pruning removes stock time the mode does not accelerate, while
    VV10, the accelerated part, runs on `mf.nlcgrids`, which no arm changes (level 3, `nwchem_prune`
    on all 45 arms).
  - **The B3LYP miss runs the other way from the model: the cache matters less than modelled.**
    `mixed-campaign/mixed-nocache-campaign` is 1.007 for B3LYP (0.985–1.024 per cell), 1.032 for
    r2SCAN and 1.021 for wB97M-V. The prediction had been scaled down from the banked anchor
    (1.860), which these pods exceeded (next item).
- **Anchor check** (`mixed-campaign/stock-campaign` against the banked C1 trio, ±0.10; the banked
  values are the protocol's, 1.638 / 1.860 / 2.803, recomputed here from L12 and from W1/W4r):
  - r2SCAN **REPLICATED**: 1.678, Δ +0.040.
  - B3LYP **REPLICATION-FLAG**: 1.968, Δ +0.108.
  - wB97M-V **REPLICATION-FLAG**: 2.944, Δ +0.141.
  - Both flags are in the **faster** direction, in every cell: B3LYP +0.07 to +0.14, wB97M-V +0.11
    to +0.17 per cell against the banked cell values; every r2SCAN cell is above its banked value
    too. The LD pods' `stock-campaign` walls are 0.93–0.99 of the banked pods' stock medians, so this
    is a faster pod on the same tree, as X3r was a slower one on the H100 (C3).
  - Against the unrounded banked geomeans the deltas are +0.041 / +0.109 / +0.142; the verdicts do
    not change.
  - The flags are shown beside every B3LYP and wB97M-V number from these pods that uses the campaign
    grid. The library-default numbers have no banked comparator and carry no flag. **The C1
    headlines are not revised**: they are banked from their own pods.
- **Disclosed, not gated.**
  - **Grid size.** Default/campaign `ngrids` is 0.632 (paracetamol), 0.634 (propranolol) and 0.626
    (celecoxib), about 0.63, inside the disclosed band 0.45–0.75. Pruning was observed, not assumed:
    the default grid has fewer points in every molecule.
  - **`stock-default/stock-campaign`**, how much faster stock itself is on the default grid: r2SCAN
    1.356, B3LYP 1.278, wB97M-V 1.035, all inside their disclosed bands (1.2–1.6; 1.0–1.15).
  - **The grid on the mixed side**, `mixed-default/mixed-nocache-campaign`: r2SCAN 1.315, B3LYP
    1.070, wB97M-V 1.110.
  - **The grid's energy difference**, |E_stock-default − E_stock-campaign|: 1.2e-8 to 1.7e-6 Ha over
    the nine cells. Eight are below the predicted 1e-6; propranolol r2SCAN is above. There is no ΔE
    gate between the two grids, which solve different quadratures.
  - **B3LYP's switch tolerance.** At the README's `xc_switch_tol=1e-3` (`mixed-default`), XC ran in
    FP32 for 8 calls in every cell, against 9–11 at the campaign's 3e-4 (both campaign arms); K ran
    in FP32 for 8 calls in every B3LYP mixed arm. The tolerance and the grid are confounded in
    `mixed-default`, by protocol; their separate effects are not measured.
  - Gates, as computed: every mixed arm within 5.9e-12 Ha of the same-grid stock arm, with identical
    cycle counts (the gate is ±1) and an FP64 tail; FP32 really used on every mixed arm; cache tiers
    as pre-registered (`fp32` mirror with zero FP64 bytes on the no-cache r2SCAN/B3LYP arms, no cache
    on the no-cache wB97M-V arms, `fp64+fp32` / `fp64` on `mixed-campaign`).
- **Scope.**
  - One card model (RTX PRO 6000 Workstation), the trio only, def2-mTZVPP only, the 1.8.1-overlay
    build at `63af0568`. Molecules beyond the trio, other bases, other cards and a master build are
    not measured at the defaults; the FP64-strong cards (C3) were not re-measured at the defaults.
  - Not measured: the default policy with the cache on, and the campaign policy on the default grid.
  - One NOISY arm: LD1 paracetamol B3LYP `mixed-campaign`, whose warm walls are under 1 s. Neither
    pod is DEGRADED (idle 43.4 W and 36.1 W). Nothing was re-run; the reserve pod was not used.
  - The protocol's final text is in the commit both pods ran (`e1cc2bb5`), 46.6 min before the LD1
    dispatch; LD2 was dispatched after LD1 finished.

## Deployment notes, not claims: MIG and a concurrent whole card `[N5, N6, N7, N8]`

**This was C5. It is no longer a claim of the RFC.**
- PREREG-7's decision rule, written before CC1 ran, said: if a whole card running four SCFs
  concurrently reaches a throughput gain G ≥ 1.25 on both arms, MIG leaves the claims. It did.
- The material below is advice for people deploying the mode. It is not a property of the library.

**A concurrent whole card under MPS beats the four-slice MIG projection** (CC1: **one pod**, one RTX
PRO 6000 Server Edition, the trio × {r2SCAN, B3LYP}). G4 = 4 × T_S / T_C4. T_S is one process doing the six cells one at a
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
  - It makes the serial baseline slower. T_S is 2.1 % (stock) and 6.2 % (mixed) above M0's summed
    warm walls for the same six cells.
  - That inflates G relative to the MIG comparator by about that much. The decision rule still
    holds with a wide margin.
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

- **It is the CUDA driver's JIT cache.** P1 leaves 60 files, 128 MB, in `CUDA_CACHE_PATH`. Wiping
  that cache brings back about 126 s; wiping CuPy's costs about 10 s.
- **Whose code is compiled is not established.** The likely source is gpu4pyscf's own modules: the
  gpu4pyscf-cuda12x 1.8.1 wheel's build lists `70-real;80;90-real`, so on sm_120 the driver must
  compile compute_80 PTX. But that is an external fact, not shipped here. The 60 cache files are not
  attributed to a binary, and gpu4pyscf-libxc-cuda12x is loaded too.
- **It is not the mode's cost.** The stock arm paid it first (P1), and the mixed arm then paid under
  2 s (P2). In the campaign pods the mixed arm ran first in each pair, which is why the cost showed
  up there.
- **It is charged to the first SCF run against an empty cache**, not to the first SCF of every
  process. In P2 and P3 the first functional of a new process paid ≤ 1.86 s. r2SCAN ran first in
  every phase, so the other functionals' small extras (≤ 1.62 s) say nothing about their own
  kernels.
- **Remedy, untested.** Keep both `CUDA_CACHE_PATH` and CuPy's kernel cache on persistent storage.
  The driver cache alone would leave about 10 s (P4). A wheel that includes sm_120 code should
  remove the driver's share, but that was not measured. The campaign's speed figures exclude the
  cold pair, so they are steady-state figures either way.

## What can and cannot be verified from this package

- **Can, with nothing but this directory** (`verify_native.py`, all of its checks):
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
  - **The exact harness bytes.** `provenance/harness/` is the harness at `5013b101`, the commit
    the PREREG-8 pods ran. Every earlier pod ran at the earlier commit listed in `pods.csv`. The
    PREREG-9 pods (LD1, LD2) ran at `e1cc2bb5`, which adds the `defaults` mode; that harness is
    **not** snapshotted here, only its extractor (`provenance/rfcbench_extract.py`), which
    re-extracts every earlier CSV unchanged (group P2).
- **Dates.** A few status dates and one RESULT date in `prereg/` are a day late. See the ledger.
- **`reproduce_native.py`** transcribes the measurement path and has not been run on a GPU in this
  form.

## Not claimed

- **Untouched by the mode:** gradients and Hessians run stock.
- **Refused:** range-separated XC and K. The VV10 component alone handles wB97M-V.
- **Not measured:**
  - library-default settings beyond C7's scope: the trio on one PRO 6000 Workstation at def2-mTZVPP
    is measured; the 24-molecule ladder, the larger molecules, other bases and other cards are not;
  - running without cuTENSOR;
  - UKS and multi-GPU;
  - the RTX 5090;
  - MIG profiles other than 1g.24gb;
  - four MIG instances running concurrently on one card;
  - geometry optimisations beyond the three in C6, tight geomeTRIC criteria, and optimisations at
    `conv_tol_grad` 3e-6;
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
- PREREG-4: gradient ≤ 1e-6 (14 of 18); gradient ≤ 3 × S′ (1 of 18: r2SCAN paracetamol). Both are
  misses about stock's convergence floor at the campaign's `conv_tol_grad`, not about precision
  (PREREG-4 erratum 2).
- PREREG-2, A100: the predictions (precision effect 0.95 / 1.00 / 1.15) could not be evaluated,
  because both pods were DEGRADED. The clean wB97M-V cells read 0.47–0.57, far below 1.15.
- PREREG-5: mixed/stock inside an instance, all 8 cells; mixed MIG gains for r2SCAN and paracetamol
  B3LYP (above); stock paracetamol B3LYP and wB97M-V (vs the per-molecule table, which PREREG-5's
  RESULT decided to apply to every functional); M0 B3LYP against C0.
- PREREG-6: B3LYP sildenafil mixed MIG gain (above).
- PREREG-8:
  - P-B, residual against R′ ≤ 3e-7 at `conv_tol_grad` 3e-6: missed in 3 of 14 cells on both arms
    (paracetamol r2SCAN 3.2e-7; wB97M-V 7.5e-7);
  - mixed vs stock ≤ 1e-7: missed in omeprazole B3LYP (2.2e-7);
  - P-D, the DIIS restart removes the B3LYP imprint: wrong in the other direction (10–100× worse);
  - P-F, the warm-start penalty: falsified;
  - P-G, warm-step wall ratios (predicted 0.9–1.3): measured 1.41–1.44, above the band. MW's band
    was not testable, because the rule never triggered.
- PREREG-7:
  - ST1 (c): the cache arm's XC is not faster than stock's at sildenafil B3LYP (10.324 s vs
    10.319 s);
  - ST2: FP32 K was predicted to harm on a MIG slice, but J/K stock/mixed is 1.09 (ST2r);
  - CC1: G4 stock 0.96 and mixed 1.02, against predictions of 1.3 and 1.4; the MPS uplift of +0.56
    and +0.73, against a prediction of 0 to +0.3.
- PREREG-9: P-4 B3LYP, the FP32 switch without the cache on the campaign grid (predicted 1.80, band
  1.65–1.90): measured 1.954, above the band. The model had scaled the banked anchor down by a cache
  effect of about 4 %; the measured cache effect for B3LYP is 1.007.

**Replication falsifier fired:** PREREG-7 X3r. H100 B3LYP mixed/stock moved by 0.115 (> 0.10), so it
is quoted as the range 0.905–1.021. The L12 replication's falsifier did not fire.

**Replication flags (PREREG-9 anchor, not a falsifier):** LD1/LD2 `mixed-campaign/stock-campaign`
read +0.108 (B3LYP) and +0.141 (wB97M-V) above the banked trio, past the ±0.10 band, in the faster
direction; r2SCAN replicated (+0.040). The C1 headlines are not revised; the flags are shown beside
the campaign-grid B3LYP and wB97M-V numbers from those pods (C7). PREREG-9's own falsifier (r2SCAN
at the defaults below 1.15) did not fire.

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
| PREREG-7 RESULT, L12r replication | evaluated on DEGRADED pods' aggregates; "the ladder replicates" | not testable as pre-registered (PREREG-0 §3); a post-hoc like-for-like clean-cell comparison is consistent (+0.05 to +0.09) |
| PREREG-7 RESULT, replicate per-cell spans | 1.55–2.02, 1.26–2.13 (NOISY cells included) | clean cells 1.57–2.02, 1.56–2.10 |
| PREREG-7 RESULT, X3r | "the cache tier is what moves between pods" | X3r was a slower pod (walls +4–34 %, stock the most), so every ratio over stock rose |
| PREREG-7 RESULT, ST1 | "the K path is what limits B3LYP at size" | withdrawn: the share shift explains about 19–37 % of the drop; the rest is unexplained |
| PREREG-7 RESULT, cold start | "a wheel built with `120-real` would remove it"; charged to "whichever functional runs first in the process" | untested; charged to the first SCF against an empty cache |
| PREREG-7 RESULT, headline | "both replications hold on their geomeans" | the X3r falsifier fired on a geomean; L12r's could not be run as registered |
| PREREG-7 RESULT, X3r idle | "higher than any Blackwell pod here" | ST2 idled at 241 W |
| PREREG-7 RESULT, stock-vs-stock floor | ≤ 2.8e-12 Ha | 3.2e-12 Ha when CC1 is pooled across phases (2.3e-12 within a phase) |
| PREREG-4 RESULT and its erratum, C4 | "a same-guess comparator was not recorded"; "mixed-specific residual" for ρ > 3; r2SCAN paracetamol's residual "systematic" | the same-guess comparator S was recorded in every sentinel and never read. M−R ≈ S−R in 17 of 18 cells: the residual is stock's convergence-tolerance residual. The precision imprint is M−S, 4e-9 to 2.7e-7 Ha/Bohr (PREREG-4 erratum 2) |

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
- **Pods are not cards.** L12r, L12r2 and ST1 drew one physical card. The five healthy wB97M-V
  ladder pods ran on three cards (W2 and W3 on one, W4r and W5 on another). B1, C0w, D2a–c and
  XL1–XL3b all ran on one card. None of this was controlled. Only the first case was disclosed when
  the package was first updated for PREREG-7; the independent review found the rest.
- **PREREG-0 §6's two-repeats rule was again applied to two DEGRADED pods**, here L12r and L12r2.
- **PREREG-8 A1's first dispatch** (run `37728120899`) produced no sentinel. It was re-dispatched
  once, unchanged, on the pre-registered reserve.
- **PREREG-8 A2 was reduced, by Amendment 1 before dispatch, to two wB97M-V cells.** Six would not
  fit one pod. The other four are not measured.
- **The ST2 re-run was approved on 2026-10-07, after CC1 had been dispatched.** That changed the
  dispatch order, CS1 → L12r → L12r2 → X3r → ST1 → ST2 → CC1 → ST2r, but no protocol text.
