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

The code diff is +914 / −18 across five files, including the tests.

## Design points for review

- **Opt-in per object.** This follows the existing precedent of explicit reduced precision in
  RI-MP2 and TDDFT-RIS. Nothing is enabled by GPU model.
- **Refusals are loud.** Each of these raises `NotImplementedError` at `kernel()`:
  - UKS / ROKS / GKS, and spin ≠ 0;
  - range-separated functionals;
  - NLC;
  - multi-GPU;
  - `k=True` without density fitting.
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
- **Scope.** The analytic gradient, the Hessian, UKS, range-separated and NLC functionals are all
  out of scope.
- **Base.** The branch is based on v1.8.1. A copy rebased onto current master exists, but has
  **not** been GPU-validated there. Master's compiled libraries differ from the 1.8.1 wheel, so
  the overlay method does not apply.
- **For the maintainers:** whether the default switch thresholds and the `|ΔE| ≤ 1e-8 Ha` test
  tolerance are acceptable, and whether you prefer the policy object or a flag on `density_fit()`.
