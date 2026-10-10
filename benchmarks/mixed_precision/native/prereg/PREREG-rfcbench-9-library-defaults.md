# PREREG rfcbench-9: what the mode gives at library defaults

**Status 2026-10-10: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`.
It freezes at the first PREREG-9 sentinel. Changes go only in dated amendments above RESULT,
written before the dispatch they govern.

## Why

Every speed figure in the RFC's evidence was measured in the **campaign configuration**:

- grid level 3 with `grids.prune = None` (unpruned);
- the FP64 AO cache on (`ao_cache_fp64=True`);
- B3LYP `xc_switch_tol=3e-4`.

The fork package says so itself (`CLAIMS.md`, "The configuration that was measured, which is not
the library default"): "No number here measures the defaults, nor the README's example
`MixedPrecision(xc=True, k=True)`. Expect smaller gains on pruned grids. How much smaller is not
measured."

A maintainer will ask two questions this protocol answers with one measurement:

1. What does the mode give **at library defaults**: the default grid, `ao_cache_fp64=False`, the
   default switch tolerances, and the policies as the README shows them?
2. What does the FP32 switch give **without the cache** on the PRO 6000? The campaign never
   measured it there: the only PRO 6000 mixed/cache numbers are from the CONTENTION-UNKNOWN C0 and
   are not claimed (`CLAIMS.md` C3, "No data").

## The fork commit measured

`chris-lee-mc/gpu4pyscf` @ `63af0568d4fd19935bef51b7fd71f161a9cee56f`, the first entry of
`VALIDATED_FORK_SHAS` (the campaign commit, PORT-PLAN-ao-cache RESULT).

- **Why this tree.** Every banked speed number the anchor below compares against (C1: L12, W1, W4r)
  was measured on this tree. The anchor is only a same-pod replication check if the tree is the same.
- **Why not `09271907d51580104c7a4e8334a8f65efc1b5d2a`.** It is this commit plus two opt-in
  geometry-safety fields, both off by default. It would measure the same algorithm, but it was never
  the tree of a speed reading. The harness enforces the choice: `MODE_REQUIRED_TREE["defaults"]` is
  `63af0568`, so the free runner refuses any other fork SHA and the pod refuses any other tree.
- **The master-based branch** (`mixed-precision-aocache-master`) carries the same algorithm.
- **Which build is timed.** Like every banked number, the speed measured is the **1.8.1-overlay
  build**: the gpu4pyscf 1.8.1 wheel, with the files the branch changes overlaid. It is not a
  master build.
- `ao_cache_fp64=False` is the default on this tree (commit `47bb1e8`, "make the FP64 AO cache
  opt-in"). The default switch tolerances on this tree are `xc_switch_tol=1e-3`,
  `k_switch_tol=1e-3` and `vv10_switch_tol=1e-5`. The default grid is level 3 with
  `nwchem_prune` (gpu4pyscf 1.8.1 `dft/gen_grid.py`, class `Grids`).

## Cells

- **Molecules:** the trio: paracetamol, propranolol and celecoxib.
- **Functionals:** r2SCAN, B3LYP and wB97M-V.
- **Basis:** the campaign's: def2-mTZVPP / def2-tzvpp-jkfit, which is what the trio cells of C0, L12
  and W1–W5 used.
- **Nine cells**, each of kind `defaults`.
- **R = 3 warm pairs per cell**, plus one cold pair that is recorded but excluded, as in the ladder
  methodology (PREREG-0 sections 1 and 3). Within each pair every arm is a fresh `pyscf.M` and a
  fresh `rks.RKS(...).density_fit(...)`. Each arm's `kernel()` is timed between two device
  synchronisations, so grid generation is inside the timed wall, as in every campaign cell.
- **Card:** RTX PRO 6000 Blackwell Workstation (`pro6000`), with no fallback.
- `conv_tol 1e-9`, `max_cycle 60`, the library-default `conv_tol_grad`, cuTENSOR required and
  observed, as in PREREG-0.

## Arms (five per cell)

Run order within each pair is the table's order. Every mixed arm runs before both stock arms, so
the warm memory pool favours stock (PREREG-0's stock-last rule, the bias against the claim).

| arm | SCF | grid | policy |
|---|---|---|---|
| `mixed-default` | mixed | **default**: `mf.grids` never touched (level 3, `nwchem_prune`) | the README's: `MixedPrecision(xc=True)` r2SCAN, `MixedPrecision(xc=True, k=True)` B3LYP, `MixedPrecision(vv10=True)` wB97M-V; every other field at its default, so `ao_cache_fp64=False` and the default tolerances |
| `mixed-nocache-campaign` | mixed | **campaign**: level 3, `prune = None` | the campaign policy without `ao_cache_fp64=True`, so the campaign tolerances (B3LYP `xc_switch_tol=3e-4`, `k_switch_tol=1e-3`) and `ao_cache_fp64=False` |
| `mixed-campaign` | mixed | campaign | the campaign policy, as measured (`ao_cache_fp64=True`) |
| `stock-default` | stock | default | none |
| `stock-campaign` | stock | campaign | none |

- `mixed-nocache-campaign` isolates **the FP32 switch without the cache** on the PRO 6000.
- `stock-campaign` and `mixed-campaign` are a **same-pod anchor** to the banked numbers.
- For r2SCAN and wB97M-V, `mixed-default` and `mixed-nocache-campaign` carry the same policy and
  differ only in the grid. For B3LYP they also differ in `xc_switch_tol` (1e-3 against 3e-4).
- The VV10 grid (`mf.nlcgrids`) is set by no arm, in this protocol or in the campaign. It is read
  back and disclosed.

## Pods: two, split by molecule

The work over-estimate (`rfcbench_sets.work_estimate_s`, with the committed C0 calibration) is
**6664 s** for all nine cells at R = 3, against the 4140 s budget (`WORK_CAP` 4440 s − 300 s). The
pre-registered rule is to split **by molecule**, so that each pod carries every functional and all
five arms, and no arm is dropped:

| pod | mode | cells | estimate |
|---|---|---|---|
| LD1 | `defaults` | paracetamol, propranolol × {r2SCAN, B3LYP, wB97M-V} (6 cells, 3 groups) | 3618 s |
| LD2 | `defaults` | celecoxib × {r2SCAN, B3LYP, wB97M-V} (3 cells, 3 groups) | 3916 s |

- **Total: 2 pods.** Dispatched one at a time, LD1 then LD2, each with nothing else queued or in
  progress. Inputs: `mode=defaults`, `fork_sha=63af0568d4fd19935bef51b7fd71f161a9cee56f`, `mols`
  as in the table, `xcs` and `repeats` blank. (`repeats` must be blank: R = 3 is fixed here.)
- **Budget rules** (as estimated above, all over-estimates):
  - `stock-default` and `stock-campaign` at the calibrated stock wall. The pruned grid has fewer
    points, so `stock-default` is over-budgeted.
  - `mixed-campaign` at the calibrated mixed wall.
  - The two no-cache arms: r2SCAN and B3LYP at the **full stock wall**, assuming no speed-up.
    wB97M-V at the mixed wall × 1.15, because the cache alone moves a wB97M-V SCF by only
    1.003–1.013 (C0w cache/stock, run `37044790110`). Removing the cache therefore adds back at most
    about 1.3 % of a stock wall.
- **Reserve:** one pod, for an infrastructure failure only (no sentinel, a card gate, a
  `ModuleNotFoundError`). It re-runs the failed pod unchanged.
- **CONTENDED pods:** PREREG-0 Amendment 3 applies. Such a pod is re-run once, only after the owner
  says so.
- **Pooling across the two pods.** A trio geomean pools per-cell ratios from LD1 and LD2. Each
  ratio is same-cell and same-pod; only the geomean spans pods, as C1's wB97M-V geomean spans W1 and
  W4r.

## Gates (fail the pod; fixed now)

Per cell and pair:

- **Structure:** pairs 0..3, each running the five arms in the table's order, with no error.
- **Settings read back off `mf`** on every arm:
  - campaign arms: level 3 and `prune is None`, exactly the campaign gate, which stays in force for
    every arm of every other mode;
  - default arms: level 3 and a prune function named `nwchem_prune`, the library default;
  - `conv_tol`, `max_cycle`, `conv_tol_grad`, basis, auxbasis, `naux`/`nao` as PREREG-0.
- **Observation, not labels:** each mixed arm's recorded `policy` equals the repr of its own
  `MixedPrecision`, built from the real class on the pod. Each stock arm carries no policy and no
  record.
- **Correctness**, each mixed arm against the stock arm **on the same grid in the same pair**
  (`mixed-default` against `stock-default`; both campaign-grid mixed arms against
  `stock-campaign`):
  - both converged;
  - |ΔE| ≤ 1e-8 Ha;
  - cycles within ±1;
  - an FP64 tail.
- **The treatment really ran** (PREREG-0's gates):
  - r2SCAN and B3LYP: at least one FP32 XC call, ending on two FP64 calls;
  - B3LYP also: at least one FP32 K call, ending on two FP64 calls, with the full K rebuild
    recorded;
  - wB97M-V: FP32 and df64 VV10 both exercised, ending on two df64 calls, the certificate ok.

  Every mixed arm's fit outcome must be `TREATED`.
- **The cache tier, observed:**
  - `mixed-campaign` must pass PREREG-0's cache gates: tier `fp64+fp32` (r2SCAN, B3LYP) or
    `fp64` (wB97M-V), with every FP64 XC call served from the cache.
  - The two no-cache arms must show the cache **off**: zero FP64 cache bytes, zero cached FP64 XC
    calls, and every FP64 XC call run stock. r2SCAN and B3LYP additionally need an FP32-only
    mirror (tier `fp32`, note `built: <x> GiB, tier fp32`) released on the XC switch call.
    wB97M-V needs no cache of either kind (tier None, empty note).
- **Grid identity:** `ngrids` is identical across every default-grid run and across every
  campaign-grid run. The default grid must have **fewer** points than the campaign grid, so pruning
  is observed, not assumed. `naux`, `nao` and `init_guess` are identical across all runs.
- **No ΔE gate between the two grids.** `stock-default` and `stock-campaign` solve different
  quadratures. |E_stock-default − E_stock-campaign| is **disclosed only**.

## Readings

All readings are computed by `rfcbench_extract.py` from banked sentinels. A speed reading is the
median warm wall ratio stock/mixed, with the existing rules:

- **PAYS** ≥ 1.15, **HARMS** ≤ 0.95, NEUTRAL in between;
- **NOISY:** an arm's warm spread exceeds 10 %;
- **DEGRADED:** more than 20 % of a pod's cells are NOISY;
- **CONTENDED:** idle power above 200 W before the first group. Such a pod's numbers are reported
  separately and never pooled.

The three speed readings per cell are:

- `mixed-default/stock-default`: **the library-default number**;
- `mixed-nocache-campaign/stock-campaign`: the FP32 switch without the cache;
- `mixed-campaign/stock-campaign`: the anchor.

These are disclosed only:

- `stock-default/stock-campaign`: how much the pruned grid speeds up stock;
- `mixed-campaign/mixed-nocache-campaign`: what the cache adds;
- `mixed-default/mixed-nocache-campaign`: the grid on the mixed side;
- per arm: `ngrids`, the grid level and prune as read back, the VV10 grid, the cache tier, and the
  XC / K / VV10 switch calls;
- the grid's energy difference.

A functional's trio geomean exists only when all three of its cells are `OK`. Otherwise it is
None, never a partial geomean.

## Predictions (fixed now)

The model behind them:
- **The default grid** has fewer points. XC is the stage the FP32 switch accelerates (85 % of a
  stock r2SCAN SCF on the campaign grid, ST1), so on the pruned grid the XC share is smaller and so
  is the gain (Amdahl).
- **Without the cache**, the FP64 tail pays AO evaluation on every call. The FP32 phase is
  unchanged: it still builds its FP32 mirror. On the PRO 6000 the cache alone measured 1.03–1.06
  for r2SCAN/B3LYP and 1.00–1.01 for wB97M-V (C0, C0w).
- **VV10** runs on `mf.nlcgrids`, which no arm changes. The default grid shrinks only wB97M-V's
  non-VV10 part, so VV10's share rises.

**P-1, r2SCAN, `mixed-default/stock-default`: 1.50, band 1.35–1.65.**
- Assume the grid ratio is 0.6. Then the XC share falls from about 0.85 to about 0.77.
- The XC stage speed-up implied by the anchor (1.638) is about 1.85 with the cache. Without it, it
  is about 1.75, taking the no-cache arm at 1.57.
- Amdahl then gives 1/(0.23 + 0.77/1.75) ≈ 1.50.

**P-2, B3LYP, `mixed-default/stock-default`: 1.45, band 1.25–1.65.** Three losses compound:
- B3LYP's XC share is already smaller (its J/K share is 9–34 % of stock in ST1), and pruning shrinks
  it further;
- without the cache, the tail pays AO evaluation;
- the README tolerance `xc_switch_tol=1e-3` is three times the campaign's 3e-4. The switch comes
  earlier, and K returns to FP64 when XC does, so both FP32 phases are shorter. That loss is not
  measured anywhere yet.

**P-3, wB97M-V, `mixed-default/stock-default`: 3.05, band 2.85–3.35.**
- That is **above** the anchor (banked 2.803). The cache contributes about 1 %, and pruning removes
  stock time the mode does not accelerate, while VV10, the accelerated part, is unchanged.
- **Same-pod corollary:** `mixed-default/stock-default` ≥ `mixed-campaign/stock-campaign` in each of
  the three wB97M-V cells.

**P-4, without the cache on the campaign grid, `mixed-nocache-campaign/stock-campaign`:**
- **r2SCAN 1.57**, band 1.45–1.70;
- **B3LYP 1.80**, band 1.65–1.90;
- **wB97M-V 2.78**, band 2.60–3.10.

These are the anchors scaled by C0's mixed/cache over mixed/stock (0.960 and 0.967; about 1.0 for
wB97M-V).

**Disclosed predictions** (not bands that decide anything):
- default/campaign `ngrids`: 0.45–0.75;
- `stock-default/stock-campaign`: 1.2–1.6 for r2SCAN and B3LYP, and 1.0–1.15 for wB97M-V;
- |E_stock-default − E_stock-campaign|: 1e-6 to 1e-4 Ha.

## Pre-registered falsifier

**If the r2SCAN `mixed-default/stock-default` geomean over the trio is below 1.15**, the RFC is
reframed:
- it must **not** lead with campaign-configuration speedups;
- it leads with the library-default number;
- it presents the cache and the unpruned grid as the measured configuration, with their own
  numbers.

The falsifier is evaluated only on a quotable trio: all three r2SCAN cells `OK`, on pods neither
DEGRADED nor CONTENDED. Otherwise it is UNDECIDED, and the reserve or the owner's re-run rule
applies. It is not read as passed.

The B3LYP and wB97M-V predictions above are not falsifiers. Below 1.15 on either, the RFC states
that functional's default number with the same prominence as r2SCAN's, whatever the r2SCAN reading.

## Anchor check (a replication flag, not a gate)

`mixed-campaign/stock-campaign` should land within **±0.10** of the banked C1 trio values. These
were recomputed here from the fork package (`mixed-precision-rfc-package-v2`,
`benchmarks/mixed_precision/native/data/`) as stock median / mixed median over pairs 1–3:

| functional | banked trio geomean | per cell: paracetamol / propranolol / celecoxib | pods |
|---|---|---|---|
| r2SCAN | 1.638 | 1.605 / 1.607 / 1.703 | L12 (`36968109711`) |
| B3LYP | 1.860 | 1.820 / 1.846 / 1.913 | L12 |
| wB97M-V | 2.803 | 2.843 / 2.986 / 2.593 | W1 (`36971083681`; paracetamol, celecoxib), W4r (`37010453926`; propranolol) |

- C1's headline 1.657 / 1.783 / 2.914 are the 24-molecule geomeans. The trio values above are the
  comparable subset. C0w's wB97M-V trio (2.963) is a second banked replicate, not the anchor.
- A miss of more than 0.10 is disclosed as a **REPLICATION-FLAG** beside every number from that pod.
  It does not fail the pod.
- `stock-campaign` walls are also disclosed against L12's and W1/W4r's stock medians, as a pod
  speed reading.

## Not measured

- Molecules beyond the trio, and bases other than def2-mTZVPP.
- Cards other than the PRO 6000 Workstation. On the FP64-strong cards C3 already reports harm.
- The default policy with the cache on, or the campaign policy on the default grid. Each is one
  more arm, and neither is asked.
- A master build. The timed build is the 1.8.1 overlay, as for every banked number.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## RESULT

### RESULT (2026-10-10): **falsifier NOT FIRED**. At library defaults the mode reads r2SCAN 1.58, B3LYP 1.64, wB97M-V 3.09

**The runs.**
- **LD1** (paracetamol, propranolol): run `38025861073`, sentinel
  `ci-results/runs/rfcbench-e1cc2bb5e577-38025861073-1-e1cc2bb5.txt`, `RUN_ID=38025861073-1-e1cc2bb5`.
- **LD2** (celecoxib): run `38027495744`, sentinel `…-38027495744-1-e1cc2bb5.txt`,
  `RUN_ID=38027495744-1-e1cc2bb5`.
- Line 2 of each was checked. Both are `STATUS=PASS`, on an RTX PRO 6000 Blackwell Workstation
  Edition, fork `63af0568`.
- **Pod health.** Neither pod is CONTENDED (idle 43.4 W and 36.1 W) or DEGRADED. One NOISY arm: LD1
  paracetamol B3LYP `mixed-campaign`, whose warm walls are under 1 s.
- **All nine cells are `OK` and `TREATED`.** Every gate held:
  - the five arms in order;
  - level and prune read back off `mf`;
  - policy reprs;
  - |ΔE| ≤ 1e-8 Ha, cycles ±1, and an FP64 tail against the same-grid stock arm;
  - FP32 used;
  - the cache tiers observed: `fp32` on the no-cache r2SCAN/B3LYP arms, none on the no-cache wB97M-V
    arms, and `fp64+fp32` / `fp64` on `mixed-campaign`;
  - pruning observed: the default grid has fewer points.
- Nothing was re-run, and the reserve was not used. The summary is
  `rfcbench_extract.py --run-id <LD1> <LD1 sentinel> --with <LD2 sentinel> <LD2 run id>`.

**Per cell, stock median / mixed median over warm pairs 1–3.**

| cell | `mixed-default/stock-default` | `mixed-nocache-campaign/stock-campaign` | `mixed-campaign/stock-campaign` |
|---|---|---|---|
| paracetamol r2SCAN | 1.505 | 1.597 | 1.645 |
| propranolol r2SCAN | 1.570 | 1.608 | 1.650 |
| celecoxib r2SCAN | 1.661 | 1.677 | 1.742 |
| paracetamol B3LYP | 1.561 | 1.992 | 1.962 (NOISY, under 1 s) |
| propranolol B3LYP | 1.728 | 1.932 | 1.958 |
| celecoxib B3LYP | 1.622 | 1.939 | 1.985 |
| paracetamol wB97M-V | 3.162 | 2.905 | 3.016 |
| propranolol wB97M-V | 3.258 | 3.055 | 3.095 |
| celecoxib wB97M-V | 2.875 | 2.703 | 2.734 |

**Trio geomeans and predictions.** Every reading is PAYS.

| | `mixed-default/stock-default` | prediction | `mixed-nocache-campaign/stock-campaign` | prediction | `mixed-campaign/stock-campaign` |
|---|---|---|---|---|---|
| r2SCAN | **1.577** | P-1 1.50 (1.35–1.65): **MET** | 1.627 | P-4 1.57 (1.45–1.70): MET | 1.678 |
| B3LYP | **1.636** | P-2 1.45 (1.25–1.65): **MET** | 1.954 | P-4 1.80 (1.65–1.90): **MODEL-MISS**, above the band | 1.968 |
| wB97M-V | **3.094** | P-3 3.05 (2.85–3.35): **MET** | 2.884 | P-4 2.78 (2.60–3.10): MET | 2.944 |

- **The falsifier did not fire.** The r2SCAN `mixed-default/stock-default` trio geomean is
  1.577 ≥ 1.15, on a quotable trio. The RFC may keep its framing, but must now state the
  library-default numbers beside the campaign ones.
- **P-3's same-pod corollary holds in all three wB97M-V cells:** `mixed-default/stock-default` ≥
  `mixed-campaign/stock-campaign` (3.162 ≥ 3.016, 3.258 ≥ 3.095, 2.875 ≥ 2.734). Pruning removes
  stock time the mode does not accelerate.
- **The B3LYP no-cache miss runs the other way: the cache matters less than modelled.**
  `mixed-campaign/mixed-nocache-campaign` is 1.007 for B3LYP (0.985–1.024 per cell). For r2SCAN it is
  1.032, and 1.021 for wB97M-V. The prediction scaled down from the banked anchor (1.860), which
  this pod exceeded (next item).

**Anchor check** (`mixed-campaign/stock-campaign` against the banked trio, ±0.10):
- **r2SCAN: REPLICATED.** 1.678 against 1.638, Δ +0.040.
- **B3LYP: REPLICATION-FLAG.** 1.968 against 1.860, Δ +0.108.
- **wB97M-V: REPLICATION-FLAG.** 2.944 against 2.803, Δ +0.141.
- Both flags are in the **faster** direction, in every cell (B3LYP +0.07 to +0.14; wB97M-V +0.11 to
  +0.17). They are disclosed beside every B3LYP and wB97M-V number from these pods.
- The C1 headline geomeans are not revised. Those are banked from their own pods. This is a later
  pod on the same tree reading higher, as X3r read against X3 on the H100 (PREREG-7).

**Disclosed, not gated.**
- **Grid size.** Default/campaign `ngrids` is 0.632 (paracetamol), 0.634 (propranolol) and 0.626
  (celecoxib). The predicted band was 0.45–0.75.
- **`stock-default/stock-campaign`** (how much faster stock itself is on the default grid): r2SCAN
  1.356, B3LYP 1.278, wB97M-V 1.035. All are in their predicted bands.
- **|E_stock-default − E_stock-campaign|** is 1.2e-8 to 1.7e-6 Ha over the nine cells. Eight are
  below the predicted 1e-6, so pruning moves the energy less than predicted.
- **`mixed-default/mixed-nocache-campaign`**, the grid effect on the mixed arm: r2SCAN 1.315,
  B3LYP 1.070, wB97M-V 1.110.
- **VV10 grid.** `nlcgrids` was level 3, `nwchem_prune`, in every arm, so no arm changed VV10's
  grid.
- **B3LYP's switch tolerance.** The README's 1e-3 and the campaign's 3e-4 are confounded with the
  grid in `mixed-default` (protocol). The switch calls are recorded per arm in the sentinels.
- **Scope.** One card model (RTX PRO 6000 Workstation), the trio only, and the 1.8.1-overlay build
  at `63af0568`. The FP64-strong cards (C3) were not re-measured at defaults.
