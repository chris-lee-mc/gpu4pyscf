# GPU validation of this branch

> **NOT GPU-validated at master.** This branch is the v1.8.1 branch
> (`mixed-precision-scf-v1.8.1`) rebased onto master `307576f`. The result below was measured on the
> v1.8.1 branch, not this one. Adapting to master needed three changes:
> - `get_jk` gained `lr_factor` / `sr_factor`, and FP32 K is now also refused when either is set;
> - `_DFHF.get_veff`'s range-separated branch was restructured, so the hook now wraps only the
>   `omega == 0` exchange build;
> - `_block_loop` lost `strict_grid_order`, because it never skips blocks now.
>
> None of these changes has run on a GPU. Master's compiled libraries differ from the 1.8.1
> wheel, so the overlay method used here does not apply; validating master needs a from-source
> build.

**Result: PASS**, 2026-09-30, on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

- Branch commit tested: `eb5a901f2d8922add275e2009ef47277125a4c87`.
- Base: v1.8.1, `5b284c258a4260baef80e3d150b4e7a81a9dbd57`.
- The full machine-readable record is in `g4pport_run3.json`.

This is a **correctness gate, not a speed measurement.**

## Method

The port is Python/CuPy only. So instead of building the CUDA libraries from source, the pod
installed the released `gpu4pyscf-cuda12x==1.8.1` wheel and overlaid the branch's changed files
onto it. It failed closed at each step:

1. **Base check.** For each of the three modified files, the wheel's copy was byte-identical to
   the v1.8.1 copy. The three files are `df/df_jk.py`, `dft/rks.py` and `scf/hf.py`. Any mismatch
   would have made the run INVALID.
2. **Regression.** These upstream tests ran on the stock wheel and again after the overlay:
   `df/tests/test_df_rks.py`, `df/tests/test_df_jk.py`, `dft/tests/test_rks.py` and
   `scf/tests/test_scf.py`. **42/42 passed both times**, and no test changed outcome.
3. **New tests.** `dft/tests/test_mixed_precision.py`: **21/21 passed.**
4. **Stock vs mixed energies** on three drug-like molecules at r2SCAN and B3LYP.
   - Settings: def2-mTZVPP / def2-TZVPP-JKFIT, grids level 3, `prune=None`, `conv_tol=1e-9`,
     default `conv_tol_grad`. One discarded warm-up per molecule.
   - Policies: r2SCAN used `MixedPrecision(xc=True)`. B3LYP used
     `MixedPrecision(xc=True, k=True, xc_switch_tol=3e-4, k_switch_tol=1e-3)`.
   - Gates: |ΔE| ≤ 1e-8 Ha, cycle difference ≤ 1, an FP64 tail, FP32 actually used, and for
     B3LYP a recorded switch of K back to FP64.

| molecule | xc | \|ΔE\| (Ha) | cycles stock / mixed | FP32 XC calls | FP32 K calls | FP64 tail | AO cache |
|---|---|---|---|---|---|---|---|
| paracetamol | r2SCAN | 6.8e-13 | 13 / 13 | 9 | 0 | yes | 1.04 GiB |
| paracetamol | B3LYP | 4.5e-13 | 12 / 12 | 10 | 8 | yes | 1.04 GiB |
| propranolol | r2SCAN | 4.5e-13 | 13 / 13 | 8 | 0 | yes | 2.69 GiB |
| propranolol | B3LYP | 2.3e-12 | 12 / 12 | 9 | 8 | yes | 2.69 GiB |
| celecoxib | r2SCAN | 1.8e-12 | 14 / 14 | 9 | 0 | yes | 3.43 GiB |
| celecoxib | B3LYP | 4.5e-12 | 13 / 13 | 11 | 8 | yes | 3.43 GiB |

The pod also recorded wall times (e.g. celecoxib B3LYP, 15.4 s stock vs 9.1 s mixed). They are one
warm run per arm with no paired repeats, so they are **disclosed, not claimed**.

## Earlier attempts (both failed, both fixed before this run)

1. **Commit `f843a64`.** The port was hooked into `rks.get_veff` only. Density-fitted RKS calls
   `df_jk._DFHF.get_veff`, so the mode never engaged. That commit fixed the hook and added a guard:
   if the convergence test is reached and no `get_veff` call has reported to the policy, it raises.
2. **Commit `b29d262`.** Everything passed except one test,
   `test_convergence_waits_for_fp64_tail`.
   - The test switches off every trigger except the convergence guard, but used
     `conv_tol=1e-10`. An FP32 trajectory never met that tolerance, so the guard never fired.
   - That property is why the stall and call-cap backstops exist and are on by default.
   - The test now loosens `conv_tol` for its mixed run (`eb5a901`, test only).
