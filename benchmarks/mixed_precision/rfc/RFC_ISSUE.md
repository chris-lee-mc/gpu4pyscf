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
(four commits on master `c1a6e37`; +2931/−39 over nine files, of which the first PR we would
propose is +928/−17 over five); the evidence package is
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

Every component defaults to off. Nothing is enabled by GPU model. Precedents in tree for explicit
reduced precision: `mp/dfmp2.py`'s `fp_type` (a `__config__`-backed attribute listed in `_keys`),
`tdscf/ris.py`'s `single` (which warns below `conv_tol` 1e-5), the `ENABLE_FP32_MULTIGRID` build
option and the cuEST wrapper's precision parameters. The policy could take a `__config__` default
in the same way (`gpu_dft_mixed_precision`, default `None`); see question 1.

**Fit with the test suite.** The 50 new tests are unmarked, skip on multi-GPU hosts
(`num_devices > 1`, the pattern upstream's own tests use; the refusal itself is tested with a
patched device count, on any host), and need no compiled code: the VV10 kernels are NVRTC
`cupy.RawModule`s (`--std=c++14`, no `__CUDA_ARCH__` branching). On one Tesla V100 (gpu4pyscf
built from source for sm_70, with upstream's single-GPU CI package set, libxc 0.9.0 included)
they pass 50/50 in 133 s, the slowest single test taking 18.3 s, and the VV10 kernels are bitwise
identical to their NumPy emulations on sm_70. Under pyscf 2.8, the multi-GPU job's pin, they pass
50/50 on one RTX PRO 6000. Not tested: a multi-GPU host itself, and scipy 1.17. In passing: with
that V100 package set, upstream's `dft/tests/test_rks.py::test_nr_coach` fails on master and on
the branch alike (the libxc 0.9.0 wheel does not name `COACH`); it is unrelated to this change.

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
three warm pairs per cell, so above 1 means mixed is faster. Two configurations were measured:

- **the library defaults**: the default grid (level 3, `nwchem_prune`), no AO cache, the default
  switch tolerances, and exactly the policies shown above (CLAIMS C7; three molecules, one card);
- **the campaign configuration**, in which most of the evidence was taken: `ao_cache_fp64=True`,
  grid level 3 with pruning off, B3LYP `xc_switch_tol=3e-4`, cuTENSOR 2.3.1, def2-mTZVPP /
  def2-tzvpp-jkfit, `conv_tol=1e-9` (CLAIMS C1–C6).

**Speed, RTX PRO 6000 Blackwell Workstation** (one card family; one pod for the r2SCAN/B3LYP
ladder):

| functional | library defaults, 3 molecules (C7) | campaign configuration, 24 molecules of 20–44 atoms (C1) | campaign per-cell range | campaign pods |
|---|---|---|---|---|
| r2SCAN (`xc`) | **1.577** | 1.657 | 1.523–1.871 | 1 |
| B3LYP (`xc`, `k`) | **1.636** | 1.783 | 1.584–1.994 | 1 |
| wB97M-V (`vv10`) | **3.094** | 2.914 | 2.593–3.250 | 5, on 3 cards |

At the defaults the per-cell readings are 1.505–1.661, 1.561–1.728 and 2.875–3.258. The default
grid has about 0.63 of the unpruned grid's points, and stock itself is 1.356 / 1.278 / 1.035
faster on it. The FP32 switch without the AO cache, on the unpruned grid, reads 1.627 / 1.954 /
2.884. The same pods re-ran the campaign configuration on the same three molecules and read
1.678 / 1.968 / 2.944 against the banked 1.638 / 1.860 / 2.803: B3LYP and wB97M-V are +0.108 and
+0.141 past the ±0.10 replication band, both faster (REPLICATION-FLAG; the 24-molecule figures
are not revised). The pre-registered falsifier for this measurement, r2SCAN at the defaults below
1.15, did not fire; one prediction missed, B3LYP without the cache at 1.954 against a band of
1.65–1.90 (the cache adds only 1.007 for B3LYP).

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

**Correctness of the master-based branch** (fork `4d2f3de`, base `c1a6e37`; gpu4pyscf built from
source; three validation profiles): upstream `test_df_rks`, `test_df_jk`, `test_rks` and
`test_scf` give 44 passed and 1 skipped on master and the same on the branch, on one RTX PRO 6000
at the lock's pins and again under pyscf 2.8; the 50 new tests pass on all three profiles; stock
vs mixed on three molecules gives \|ΔE\| ≤ 5.9e-12 Ha at the lock's pins and ≤ 2.7e-12 Ha under
pyscf 2.8, with identical cycle counts. On the V100 the regression, new-test and VV10-identity
gates pass; the three-molecule comparison was not run there.

## What is not claimed

- **Library-default settings**: measured only on three molecules, one card and def2-mTZVPP (C7).
  The 24-molecule figures, the larger molecules and the other cards are in the campaign
  configuration, which needs the opt-in FP64 AO cache and an unpruned grid.
- **Running without cuTENSOR**: speed not measured. The K path calls `cupy_helper.contract`, so it
  uses whichever backend GPU4PySCF has; the campaign gated cuTENSOR on every pod. Correctness was
  validated on both engines: cuTENSOR 2.3.1 in the campaign, cupy in the source-build validation.
- **UKS, multi-GPU, gradients, Hessians**: out of scope (refused or untouched).
- **Range-separated functionals** with `xc`/`k`: refused. NLC functionals are treated only by the
  separate `vv10` component.
- **Architectures**: speed measured on RTX PRO 6000 (Workstation and Server), L40S, H100 and A100;
  correctness also on a V100. Not the RTX 5090, B200 or any consumer card. Timings on one card
  model say nothing about another, and only two FP64:FP32 throughput levels were measured, so no
  gating threshold is established.
- **Deployment notes** (MIG, measured only as a one-instance projection and dropped from the
  claims; concurrent processes under MPS) and the **cold-start analysis** (the CUDA driver's JIT
  cache, paid once per machine, by stock too) are in CLAIMS.md and are not claims of this RFC.
- Two geometry-safety options (a DIIS reset at the switch, a warm-start rule) were investigated on
  the fork and are not proposed: the DIIS reset did not help (it moved the stopping point to
  7.4e-7 – 3.8e-6 Ha/Bohr from stock), and the warm-start rule never triggered.

## Questions for the maintainers

1. **API shape.** A policy object (`mf.mixed_precision = MixedPrecision(...)`) or a flag on
   `density_fit()` / `RKS`? The object keeps the switch settings and the record in one place; a
   `__config__` default (`gpu_dft_mixed_precision`) could sit beside it, as `fp_type` does.
2. **Device guard.** On the H100 and A100 the FP32 switch is slower than stock. Should the mode
   warn or refuse by compute capability (sm_80/sm_90 class), or only document it? We have no
   measured threshold to gate on.
3. **Scope of a first PR.** XC + K alone, with the VV10 component and the FP64 AO cache as
   follow-ups? The XC + K commit (+928/−17, five files, 22 tests) applies on master alone; the
   VV10 commit applies on it; the AO-cache commit applies cleanly only after VV10 (one file,
   `mixed_precision.py`, conflicts otherwise) and would be re-authored if you want the cache
   second. What XC + K alone gives is the no-cache reading above: 1.627 (r2SCAN) and 1.954 (B3LYP)
   on the unpruned grid, and the library-default readings 1.577 and 1.636, which use no cache
   either. The VV10 component adds two NVRTC-compiled kernels (`cupy.RawModule`) and the keywords
   `nr_nlc_vxc(vv10_kernel=)` / `_vv10nlc(uwe_kernel=)`, default `None`.
4. **Convergence contract.** Is "two consecutive FP64 iterations before convergence is accepted",
   enforced in `scf._kernel`, acceptable there, or should it live in the RKS classes? An
   alternative with in-tree precedent: `SCF.check_convergence` exists and `soscf` honours it, but
   `hf._kernel` does not. We could add that parity call to `_kernel` and have the policy install
   `mf.check_convergence`, which removes the mixed-precision branch from `_kernel` entirely.
5. **Defaults and tolerances.** Are `xc_switch_tol = k_switch_tol = 1e-3` and the test tolerance
   \|ΔE\| ≤ 1e-8 Ha against stock acceptable? The campaign used `xc_switch_tol=3e-4` for B3LYP;
   at the default 1e-3 the B3LYP switch came after 8 FP32 XC calls instead of 9–11.
6. **The record.** `mf.mixed_precision_record` (a dict) and two new entries in
   `KohnShamDFT._keys`: fine, or would you prefer it on the logger only?
