<!--
STAGING FILE. This is the intended body of a pull request to pyscf/gpu4pyscf. It has not been
opened. Delete this file from the branch before opening the PR.
Preferred order: first file the RFC issue asking whether the maintainers want the mode at all,
then open this PR only if they do, referencing that issue.
-->

# Opt-in mixed-precision SCF for closed-shell RKS on FP64-limited GPUs

> **NOT GPU-validated at master.** This branch is the v1.8.1 branch
> (`mixed-precision-scf-v1.8.1`) rebased onto master `307576f`. The GPU validation cited below was measured on the
> v1.8.1 branch, not this one. Adapting to master needed three changes:
> - `get_jk` gained `lr_factor` / `sr_factor`, and FP32 K is now also refused when either is set;
> - `_DFHF.get_veff`'s range-separated branch was restructured, so the hook now wraps only the
>   `omega == 0` exchange build;
> - `_block_loop` lost `strict_grid_order`, because it never skips blocks now.
>
> None of these changes has run on a GPU. Master's compiled libraries differ from the 1.8.1
> wheel, so the overlay method used here does not apply; validating master needs a from-source
> build.


## Summary

This adds an **opt-in**, per-SCF-object policy, `mf.mixed_precision = MixedPrecision(...)`. It is
off by default: with it unset, no code path changes.

When enabled, early SCF iterations run selected large contractions in FP32:
- XC quadrature: the density and XC-potential contractions;
- DF exchange, for global hybrids: the two K contractions.

The SCF then switches one way back to FP64. It cannot report convergence until two consecutive
iterations were built entirely in FP64.

The target is GPUs whose FP64 throughput is a small fraction of FP32, such as RTX and workstation
Blackwell cards. An external prototype of the same scheme measured these whole-SCF speedups
against stock GPU4PySCF 1.8.1 on an RTX PRO 6000 Blackwell:
- r2SCAN, 24 drug-like molecules: **1.78–1.84×** tier geometric means;
- B3LYP, with cuTENSOR on both sides, 4 molecules: **1.92×**.

In every run, the final energy matched stock to ≤ 1e-9 Ha (worst 1.7e-10 Ha), and cycle counts
matched or differed by one. See
`benchmarks/mixed_precision/README.md` for the data and the script that re-derives every
aggregate.

## What changes

| file | change |
|---|---|
| `gpu4pyscf/dft/mixed_precision.py` (new) | `MixedPrecision` policy, one-way `PhaseController`, per-SCF state and record, FP32 AO-value cache, FP32 rho/vxc block kernels, `nr_rks_fp32`, support checks |
| `gpu4pyscf/df/df_jk.py` | `get_jk`: optional FP32 path for the two K contractions (chunked FP32 `cderi`, FP64 accumulator). `_DFHF.get_veff` (RHF branch) reports each iteration to the policy. |
| `gpu4pyscf/dft/rks.py` | `get_veff` (non-DF) reports to the policy. The first FP64 build after an FP32 phase is a full J rebuild. Adds `mixed_precision` / `mixed_precision_record` to `KohnShamDFT._keys`. |
| `gpu4pyscf/scf/hf.py` | `scf()` opens and closes the policy's per-run state. `_kernel` refuses convergence without an FP64 tail: it forces the switch and continues. |
| `gpu4pyscf/dft/tests/test_mixed_precision.py` (new) | 21 tests |
| `benchmarks/mixed_precision/` (new) | Evidence bundle, usage notes, GPU validation record |
| `README.md` | One line under experimental features |

For the XC and K components alone (v1.8.1 to `mixed-precision-scf-v1.8.1`), the code diff is
+914 / −18 across the five `gpu4pyscf/` files above, including the tests. The VV10 component (below) adds
two files and touches `numint.py`; it is not in that count.

## Design points for review

- **Opt-in per object.** This follows the existing precedent of explicit reduced precision in
  RI-MP2 and TDDFT-RIS. Nothing is enabled by GPU model.
- **Refusals are loud.** Each of these raises `NotImplementedError` at `kernel()`:
  - UKS / ROKS / GKS, and spin ≠ 0;
  - with `xc=True` / `k=True`: range-separated functionals and NLC functionals;
  - multi-GPU;
  - `k=True` without density fitting;
  - with `vv10=True` (separate component, below): no NLC term, or not exactly one VV10 term.
- **What stays FP64.** Functional evaluation (`eval_xc_eff`), grid weights, the electron count,
  the XC energy, the Vxc accumulator, J, and the K accumulator. FP32 block results are promoted
  before accumulation.
- **Convergence contract.** This is a structural check in `_kernel`, not a benchmark gate.
  - DF-RKS rebuilds XC/J/K from the full density every iteration, so no FP32 contribution survives
    the switch.
  - Non-DF RKS builds J incrementally, so the first FP64 build is a full rebuild.
- **Switch triggers.** An XC-energy-change threshold, a stall detector and an iteration cap. The
  last two are necessary: a convergence tolerance below the FP32 noise floor may never be met in
  FP32. The thresholds are empirical defaults, not error bounds.
- **Memory.** The FP32 AO cache is built once per SCF, bounded by a fraction of free memory. If it
  does not fit, XC stays FP64, and the record says so. The mode never caches partially.
- **Backend-agnostic.** The K path calls GPU4PySCF's own `contract`, so it inherits cuTENSOR when
  that is active.
- **Transparency.** `mf.mixed_precision_record` lists, for each iteration, the precision used for
  XC and K, and the reason for each switch.

## Testing

- **New tests** (`dft/tests/test_mixed_precision.py`):
  - disabled mode is bit-for-bit stock;
  - enabled vs stock at r2SCAN, PBE and B3LYP, and K-only; XC also without DF: |ΔE| ≤ 1e-8 Ha,
    cycles ±1;
  - comparison with CPU PySCF;
  - the convergence contract, including a forced switch;
  - each refusal;
  - cache invalidation on a scanner geometry change, and state cleared after `kernel()`;
  - the FP32 XC and K kernels against their FP64 counterparts;
  - the memory fallback.
- **GPU validation** on one RTX PRO 6000 Blackwell (`benchmarks/mixed_precision/validation/`).
  The branch's changed files were overlaid onto the released 1.8.1 wheel, after checking that the
  base copies were byte-identical to it.
  - Upstream `test_df_rks`, `test_df_jk`, `test_rks` and `test_scf`: 42/42 before and after the
    overlay.
  - New tests: 21/21.
  - Stock vs mixed on paracetamol, propranolol and celecoxib, at r2SCAN and B3LYP: |ΔE| ≤ 4.5e-12
    Ha, identical cycle counts, and an FP64 tail in every run.

## Limitations and open questions

- **Speed numbers.** The published ones come from the prototype, not from this implementation.
  This PR's own validation is a correctness gate; a like-for-like speed re-measurement with this
  code is still to do.
- **Scope.** The analytic gradient, the Hessian and UKS are out of scope. Range-separated and NLC
  functionals are out of scope for `xc` / `k`; the VV10 component below is the only path that
  treats an NLC functional.
- **Base.** The branch is based on v1.8.1. A copy rebased onto current master exists, but has
  **not** been GPU-validated there. Master's compiled libraries differ from the 1.8.1 wheel, so
  the overlay method does not apply.
- **For the maintainers:** whether the default switch thresholds and the `|ΔE| ≤ 1e-8 Ha` test
  tolerance are acceptable, and whether you prefer the policy object or a flag on `density_fit()`.

## VV10 component (`MixedPrecision(vv10=True)`)

**Scope note.** The upstream RFC (`DISCUSSION-DRAFT-precision-option.md`) lists NLC among the
initial exclusions, so this component is outside what it proposes. It would be proposed separately,
or as a follow-up once the XC/K mode is settled; it is on this branch so that it can be reviewed
and validated against the same 1.8.1 base. It is independent of `xc` / `k`: those still refuse
NLC functionals, and `vv10=True` runs alone.

**What it does.** It treats only the VV10 pair sum of `numint._vv10nlc` (the O(N²) U/W/E kernel on
the masked nlc grid), for functionals with exactly one VV10 term such as wB97M-V.
- Early calls run FP32 pair terms with FP64 accumulation and hi/lo-split coordinates
  (`f32_t1_hilo`).
- A one-way switch on |ΔE_nlc| < 1e-5 Ha (with the same stall detector and cap) moves every later
  call to a df64 kernel (`df64_f32_t1`: double-float arithmetic on FP32 pairs, ~48-bit). There is
  no stock FP64 tail call.
- When the SCF ends, a stock FP64 **certificate** re-runs the stock kernel on the last df64 call's
  inputs, inside the SCF timer. Bands: 1e-10 relative on E/U/W, 1e-11 Ha on E_nlc. Out of band,
  `mf.converged` is set to False and `RuntimeError` is raised. An SCF that raised skips it.
- The convergence guard counts a df64 VV10 call as clean and an FP32 one as not: convergence is
  accepted only after two consecutive iterations built without FP32-phase arithmetic.
- Range-separated functionals (wB97M-V is one) and the non-DF path are allowed for this component.
- A kernel that fails to build or fails its forced probe raises `RuntimeError`; there is no
  fallback.

**New keywords in `numint`.** `_vv10nlc(..., uwe_kernel=None)` and
`nr_nlc_vxc(..., vv10_kernel=None)`, which passes it on as `uwe_kernel`. With the default None the
stock statements run unchanged; positional callers are untouched.

**Measured** with the external prototype, wB97M-V / def2-mTZVPP, DF, RTX PRO 6000 Blackwell (run
`36875554583`; data in `benchmarks/mixed_precision/data/vv10_wb97mv_pro6000.csv`):

| molecule | stock FP64 → treated wall (s) | e2e | \|ΔE\| (Ha) | cycles | certificate rel / wall |
|---|---|---|---|---|---|
| paracetamol | 19.14 → 6.81 | 2.809 | 2.3e-13 | 12 / 12 | 7.8e-14 / 1.17 s |
| propranolol | 75.73 → 25.54 | 2.965 | 9.1e-13 | 13 / 13 | 1.6e-13 / 4.47 s |
| celecoxib | 92.16 → 34.98 | **2.635** | 4.5e-13 | 13 / 13 | 1.5e-13 / 5.07 s |

- 7 FP32 calls, then 6–7 df64 calls, on every molecule. Against stock GPU4PySCF's warm wall the
  geomean is 2.798.
- Kernel level (celecoxib): FP32 30.4× and df64 5.32× faster than the stock UWE kernel; error
  against stock ≤ 1.16e-7 relative (FP32) and ≤ 2.3e-13 (df64).
- Before the df64 tail, the same FP32 phase with a stock FP64 tail gave 1.363× (tol 1e-4) and
  1.592× (tol 1e-5) on celecoxib.
- Not measured: other cards, functionals or nlcgrids, gradients, open shell. The port's own GPU
  validation is a correctness gate, not a speed re-measurement.

**Files.**

| file | change |
|---|---|
| `gpu4pyscf/dft/vv10_mixed.py` (new) | FP32 and df64 VV10 kernel sources (NVRTC, cupy imported lazily), bind with forced probes once per device, `uwe_stock`, `certify` |
| `gpu4pyscf/dft/numint.py` | the `uwe_kernel=` / `vv10_kernel=` keywords |
| `gpu4pyscf/dft/mixed_precision.py` | `vv10` policy field and `vv10_switch_tol`, component-scoped `check_supported`, the VV10 controller, launcher selection, record keys, the certificate in `end()` |
| `gpu4pyscf/df/df_jk.py`, `gpu4pyscf/dft/rks.py` | `get_veff` passes the selected launcher to `nr_nlc_vxc` and reports E_nlc to the policy |
| `gpu4pyscf/scf/hf.py` | `scf()` certifies on success and skips the certificate when the SCF raised |
| `gpu4pyscf/dft/tests/test_mixed_precision_vv10.py` (new) | 18 tests |

**Testing.** `test_mixed_precision_vv10.py` (18 tests): the kernel bind; each kernel against stock
`_vv10nlc` per point on a converged paracetamol density (the FP64 reference bit-identical, FP32
and df64 within their bands, `uwe_kernel=None` bit-identical to `uwe_stock`); wB97M-V SCF mixed vs
stock on water and paracetamol with DF and on water without (|ΔE| ≤ 1e-8 Ha, cycles ±1, df64 tail,
certificate in band on the last call); the convergence guard; a perturbed df64 result that must
fail the certificate; an SCF that raises; each refusal; and the default policy as stock. The 21
existing tests in `test_mixed_precision.py` are unchanged.

**GPU validation: PASS** on one RTX PRO 6000 at `906043fd`
(`benchmarks/mixed_precision/validation/VALIDATION-vv10.md`):
- the base files are byte-identical to the 1.8.1 wheel;
- both kernels are bitwise identical on the GPU to their NumPy emulations;
- the upstream regression tests pass 42/42 before and after the overlay;
- the new tests pass 39/39;
- wB97M-V on paracetamol, propranolol and celecoxib: |ΔE| ≤ 9.1e-13 Ha, identical cycles, a df64 tail
  with no stock tail call, and the certificate ≤ 1.8e-13 relative.
