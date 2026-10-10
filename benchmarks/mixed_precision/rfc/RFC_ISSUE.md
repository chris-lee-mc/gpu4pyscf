<!--
STAGING FILE: the text of a GitHub issue for pyscf/gpu4pyscf. Not yet posted. Open this issue
first and wait for maintainer feedback before any PR (see CHECKLIST.md). Every number below is
checked against the evidence package by `python3 benchmarks/mixed_precision/rfc/verify_rfc.py`.
-->

# RFC: opt-in mixed-precision SCF for density-fitted RKS on FP64-limited GPUs

## Problem

On GPUs whose FP64 throughput is a small fraction of FP32 (RTX and workstation Blackwell cards,
L40S), a DF-RKS SCF spends most of its time in two FP64 contractions: the XC quadrature (density
and potential on the grid) and, for global hybrids, the two DF exchange contractions. Early SCF
iterations do not need FP64 there, but the result must still be an FP64 SCF, with stock
convergence and stock gradients.

We have an implementation on a fork and would like to know whether the maintainers want it, and
in what shape, before opening a PR. The code is
[`chris-lee-mc/gpu4pyscf@mixed-precision-upstream`](https://github.com/chris-lee-mc/gpu4pyscf/tree/mixed-precision-upstream)
(four commits on master `c1a6e37`); the evidence package is
[`chris-lee-mc/gpu4pyscf@mixed-precision-rfc-package-v2`, `benchmarks/mixed_precision/`](https://github.com/chris-lee-mc/gpu4pyscf/tree/mixed-precision-rfc-package-v2/benchmarks/mixed_precision),
whose `native/verify_native.py` recomputes every number quoted here from the per-run data.

## Proposed API

A policy object on the SCF object, off by default (`mf.mixed_precision = None`):

```python
from gpu4pyscf.dft import rks
from gpu4pyscf.dft.mixed_precision import MixedPrecision

mf = rks.RKS(mol, xc='b3lyp').density_fit()
mf.mixed_precision = MixedPrecision(xc=True, k=True)
mf.kernel()
mf.mixed_precision_record      # precision of every iteration, and why it switched

mf = rks.RKS(mol, xc='wb97m-v').density_fit()
mf.mixed_precision = MixedPrecision(vv10=True)   # separate component, runs alone
```

```python
MixedPrecision(xc=False, k=False, vv10=False,
               xc_switch_tol=1e-3, k_switch_tol=1e-3, vv10_switch_tol=1e-5,
               stall=2, call_cap=30,
               ao_cache_mem_fraction=0.7, ao_cache_fp64=False)
```

Every component defaults to off. Nothing is enabled by GPU model. This follows the precedent of
explicit reduced precision in RI-MP2 and TDDFT-RIS.

## How precision is controlled

| component | FP32 phase | always FP64 |
|---|---|---|
| `xc` | rho and vxc contractions, against an FP32 copy of the AO values built once per SCF | `eval_xc_eff`, grid weights, electron count, the E_xc reduction, the Vxc accumulator; every block result is promoted before accumulation |
| `k` (global hybrids, DF only) | the two K contractions of `df_jk.get_jk`, on FP32 chunks of `cderi` | J, and the K accumulator |
| `vv10` (exactly one VV10 term, runs alone) | the O(N²) U/W/E pair sum, FP32 pair terms with FP64 accumulation | everything outside the pair sum; after the switch the pair sum runs a df64 (double-float, ~48-bit) kernel, and the last call is certified against stock FP64 |

- **FP32 is used only in early iterations.** Each component switches one way when the change in
  its energy between iterations (E_xc for `xc` and `k`, E_nlc for `vv10`) falls below its
  threshold; a stall detector (`stall`) and an iteration cap (`call_cap`) back that up. The
  thresholds are controller settings, not error bounds. With `xc` and `k` both on, K returns to
  FP64 no later than XC.
- **Convergence is always finished in FP64.** `scf._kernel` accepts convergence only after two
  consecutive iterations built without FP32-phase arithmetic. If the convergence tests pass
  earlier, the switch is forced and the SCF continues. The DF `get_veff` builds J/K from the full
  density every iteration, so no FP32 K contribution survives the switch. Without density fitting
  only XC can be FP32 (`k=True` is refused there), and XC is rebuilt from the full density every
  iteration; `rks.get_veff` additionally forces a full J rebuild on the first FP64 K iteration, as
  a guard.
- **Energies are accumulated in FP64**, and gradients and Hessians are not touched: they run
  stock, in FP64, on the converged FP64 density.
- **VV10 tail and certificate.** After its switch every VV10 call runs the df64 kernel; there is
  no stock FP64 tail call. When the SCF ends, the stock FP64 kernel re-runs on the inputs of the
  last df64 call (bands 1e-10 relative on E/U/W, 1e-11 Ha on E_nlc). Out of band, `mf.converged`
  is set to False and `RuntimeError` is raised.
- **FP64 AO cache** (`ao_cache_fp64=True`, opt-in): the AO values are evaluated once per SCF and
  kept in FP64, so the FP64 tail skips the AO evaluation. The cached path runs `numint.nr_rks`'s
  own kernels on the cached blocks and is bitwise identical to stock (tested). The tier
  (`fp64+fp32`, `fp32`, none) is chosen from the predicted size before allocation, within
  `ao_cache_mem_fraction` of free device memory; what did not fit is recorded, never guessed.
- **Refused with `NotImplementedError` at `kernel()`:** UKS/ROKS/GKS and spin ≠ 0; multi-GPU;
  with `xc`/`k`, range-separated and NLC functionals; `k` without density fitting; with `vv10`, a
  functional without exactly one VV10 term. Nothing falls back silently.
- **With `mf.mixed_precision` unset, no code path changes.** Every hook is behind a `None` check;
  the regression tests below pass identically on both trees.

## Evidence

Measured with the fork's own code against stock GPU4PySCF (a v1.8.1 base; the master-based
branch was re-validated for correctness, below). Ratios are stock median / mixed median over
three warm pairs per cell, so above 1 means mixed is faster. **The measured configuration is not
the library default:** `ao_cache_fp64=True`, grid level 3 with pruning off, B3LYP
`xc_switch_tol=3e-4`, cuTENSOR 2.3.1, def2-mTZVPP / def2-tzvpp-jkfit, `conv_tol=1e-9`. The
defaults and pruned grids were not measured; expect smaller gains on pruned grids.

**Speed, RTX PRO 6000 Blackwell Workstation, 24 drug-like molecules of 20–44 atoms** (CLAIMS C1):

| functional | mixed/stock geomean | per-cell range | pods |
|---|---|---|---|
| r2SCAN (`xc`) | 1.657 | 1.523–1.871 | 1 |
| B3LYP (`xc`, `k`) | 1.783 | 1.584–1.994 | 1 |
| wB97M-V (`vv10`) | 2.914 | 2.593–3.250 | 5, on 3 cards |

Two re-runs of the r2SCAN/B3LYP pod were too noisy to quote as aggregates; on their clean cells,
compared like for like, they read 0.05–0.09 higher. On six molecules of 63–98 atoms: r2SCAN 1.931
(1.802 without one anomalous stock wall), B3LYP 1.338 (CLAIMS C2). B3LYP's gain falls with system
and basis size: at 475–559 Da the J/K stage is about half the stock SCF and gains only 5–7 %,
while XC still gains 2–3×.

**Negative result, FP64-strong cards** (CLAIMS C3). On an L40S the mode pays (r2SCAN 1.729,
B3LYP 1.524, wB97M-V 2.994). On an H100 it does not: wB97M-V 0.577–0.615 (i.e. slower), B3LYP
0.905–1.021, and r2SCAN's 1.320–1.398 comes almost entirely from the FP64 AO cache (the FP32
switch itself reads 1.084). On an A100, 11 of 12 clean cells were neutral or slower. **The code
does not detect the device**; see question 2.

**Accuracy** (CLAIMS C4; 18 cells, r2SCAN/B3LYP/wB97M-V × six molecules, mixed M against stock S
run from the same guess with the same settings):

| max \|M−S\| | gradient (Ha/Bohr) | dipole (D) |
|---|---|---|
| r2SCAN | 3.8e-9 – 5.3e-8 | 3.0e-7 – 2.4e-6 |
| B3LYP | 2.1e-8 – 2.7e-7 | 2.0e-7 – 9.3e-6 |
| wB97M-V | 1.2e-8 – 3.2e-8 | 1.2e-12 – 5.9e-12 |

- Every \|ΔE\| against stock is ≤ 1.6e-10 Ha; cycle counts match in 17 of 18 cells. The exception
  (omeprazole B3LYP, one extra cycle, gradient 4.4e-6 from S) landed 3.4× closer to a tight
  reference than stock did.
- Against that tight reference, stock and mixed carry the same residual, 6.6e-7 – 3.4e-6 Ha/Bohr.
  It is set by the default `conv_tol_grad`, not by precision. At `conv_tol_grad = 3e-6` it falls
  for both arms alike, and mixed then tracks stock to 1.5e-9 – 2.3e-8 (r2SCAN), 6.8e-9 – 2.2e-7
  (B3LYP) and 1.1e-8 – 2.1e-8 (wB97M-V) Ha/Bohr, with identical cycle counts in every cell.

**Geometry optimisation** (CLAIMS C6; geomeTRIC, stock gradients, one card):

| cell | steps mixed / stock | endpoint RMSD vs stock (Å) | warm-step SCF, stock/mixed | whole optimisation |
|---|---|---|---|---|
| paracetamol r2SCAN | 23 / 23 | 9.9e-6 | 1.41 | 1.28 |
| paracetamol B3LYP | 26 / 26 | 1.5e-6 | 1.42 | 1.19 |
| celecoxib r2SCAN | 16 / 16 | 4.0e-7 | 1.44 | 1.30 |

Every mixed endpoint passes a fresh stock SCF and gradient (grms ≤ 7.2e-5, gmax ≤ 2.3e-4 Ha/Bohr,
\|ΔE\| vs stock's final energy ≤ 3.2e-9 Ha). The whole-optimisation gain is lower than the SCF
gain because the gradient runs stock.

**Correctness of the master-based branch** (from source for sm_120, one RTX PRO 6000): upstream
`test_df_rks`, `test_df_jk`, `test_rks` and `test_scf` give 44 passed and 1 skipped on master and
the same on the branch; the 49 new tests pass; stock vs mixed on three molecules gives
\|ΔE\| ≤ 6.4e-12 Ha with identical cycle counts.

## What is not claimed

- **Library-default settings and pruned grids**: not measured. The speed figures need the opt-in
  FP64 AO cache and an unpruned grid.
- **Running without cuTENSOR**: not measured. The K path calls `cupy_helper.contract`, so it uses
  whichever backend GPU4PySCF has; the campaign gated cuTENSOR on every pod.
- **UKS, multi-GPU, gradients, Hessians**: out of scope (refused or untouched).
- **Range-separated functionals** with `xc`/`k`: refused. NLC functionals are treated only by the
  separate `vv10` component.
- **Architectures**: measured on RTX PRO 6000 (Workstation and Server), L40S, H100 and A100. Not
  the RTX 5090, B200, V100 or any consumer card. Timings on one card model say nothing about
  another, and only two FP64:FP32 throughput levels were measured, so no gating threshold is
  established.
- **MIG**: measured only as a projection from one instance at a time, and dropped from the claims.
  For throughput of many small SCFs on one card, four processes under MPS gave 1.52× (stock) and
  1.75× (mixed) the serial throughput on one pod; that is deployment advice, not a library
  property.
- **Cold start**: the first SCF on a fresh Blackwell machine took 64–168 s. That is the CUDA
  driver's JIT cache, which stock pays too (134.0 s when it runs first); it is not the mode's cost.
- Two geometry-safety options (a DIIS reset at the switch, a warm-start rule) were investigated on
  the fork and are not proposed: the DIIS reset did not help (it moved the stopping point to 7.4e-7 – 3.8e-6 Ha/Bohr
  from stock), and the warm-start rule
  never triggered.

## Questions for the maintainers

1. **API shape.** A policy object (`mf.mixed_precision = MixedPrecision(...)`) or a flag on
   `density_fit()` / `RKS`? The object keeps the switch settings and the record in one place.
2. **Device guard.** On the H100 and A100 the FP32 switch is slower than stock. Should the mode
   warn or refuse by compute capability (sm_80/sm_90 class), or only document it? We have no
   measured threshold to gate on.
3. **Scope of a first PR.** XC + K alone, with the VV10 component and the FP64 AO cache as
   follow-ups? The VV10 component adds two NVRTC-compiled kernels (`cupy.RawModule`) and the
   keywords `nr_nlc_vxc(vv10_kernel=)` / `_vv10nlc(uwe_kernel=)`, default `None`.
4. **Convergence contract.** Is "two consecutive FP64 iterations before convergence is accepted",
   enforced in `scf._kernel`, acceptable there, or should it live in the RKS classes?
5. **Defaults and tolerances.** Are `xc_switch_tol = k_switch_tol = 1e-3` and the test tolerance
   \|ΔE\| ≤ 1e-8 Ha against stock acceptable? The campaign used `xc_switch_tol=3e-4` for B3LYP.
6. **The record.** `mf.mixed_precision_record` (a dict) and two new entries in
   `KohnShamDFT._keys`: fine, or would you prefer it on the logger only?
