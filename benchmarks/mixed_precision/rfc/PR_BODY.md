<!--
STAGING FILE: the body of the pull request to pyscf/gpu4pyscf, to be opened only after the
maintainers have answered the RFC issue (RFC_ISSUE.md). Replace "#<issue>" with its number.
Every number below is checked by `python3 benchmarks/mixed_precision/rfc/verify_rfc.py`.
-->

# Opt-in mixed-precision SCF for closed-shell DF-RKS (XC, DF-K, VV10)

Refs #<issue>.

## Summary

Adds an opt-in, per-SCF-object policy, `mf.mixed_precision = MixedPrecision(...)`, default
`None`. When enabled, early SCF iterations run selected large contractions in FP32: the XC
density/potential contractions (`xc=True`), the two DF exchange contractions for global hybrids
(`k=True`), or the VV10 pair sum (`vv10=True`, separate component). Each component switches one
way back to FP64 on an energy-change threshold, a stall detector or an iteration cap, and the SCF
cannot report convergence until two consecutive iterations were built without FP32-phase
arithmetic. Functional evaluation, grid weights, J, every accumulator and every energy stay FP64.
Gradients and Hessians are untouched.

Target: GPUs whose FP64 throughput is a small fraction of FP32. Measured with this code against
stock GPU4PySCF on an RTX PRO 6000 Blackwell Workstation, 24 drug-like molecules (20–44 atoms),
def2-mTZVPP, three warm pairs per cell, at the settings listed under Evidence (not the library
defaults): r2SCAN 1.657, B3LYP 1.783, wB97M-V 2.914 (mixed/stock geomeans). At the library
defaults (the default pruned grid, no AO cache, the policies exactly as above), on three of those
molecules: r2SCAN 1.577, B3LYP 1.636, wB97M-V 3.094. On the H100 and A100 tested the FP32 switch
is slower than stock (wB97M-V 0.577–0.615 on the H100), and the code does not detect that device
class; see Limitations.

## Reviewer's guide

Read in this order. Diff: +2931 / −39 over nine files, of which +2070 / −39 is code outside the
tests and README.

| order | file | what it does |
|---|---|---|
| 1 | `gpu4pyscf/dft/mixed_precision.py` (new, +963) | `MixedPrecision` policy; `PhaseController` (one-way switch on an energy trace); `check_supported` (every refusal); `_SCFState` (per-iteration protocol `begin_call` / `end_call`, the clean-streak that defines the FP64 tail, `k_scope`, the record); `begin` / `end` (lifecycle, VV10 certificate); the AO cache (`_build_ao_cache`, tiers decided from predicted size); `nr_rks_fp32` (FP32 rho/vxc, FP64 promotion per block) and `nr_rks_fp64_cached` (the statements of `numint.nr_rks` / `_nr_rks_task` on cached FP64 blocks, bitwise stock) |
| 2 | `gpu4pyscf/scf/hf.py` (+41 / −12) | `scf()` opens the state (`mixed_precision.begin`) before `_kernel` and closes it after (`end`, which certifies VV10; `end(failed=True)` on an exception). `_kernel`: when the convergence tests are met without an FP64 tail, forces the switch and continues; raises if no `get_veff` ever reported to the policy (an unsupported `get_veff` override) |
| 3 | `gpu4pyscf/df/df_jk.py` (+77 / −3) | `_DFHF.get_veff` (RHF branch): `begin_call`, FP32 or cached-FP64 XC, `k_scope` around `get_jk`, `vv10_kernel=` to `nr_nlc_vxc`, `end_call`. `get_jk`: the FP32 K path (`_k_block_fp32`: FP32 half-transform in the FP64 `rhok` buffer, `cderi` cast in aux-index chunks, FP64 accumulator), taken only when `dfobj._k_precision == 'fp32'` for the plain `omega=0`, mode-0 case |
| 4 | `gpu4pyscf/dft/rks.py` (+45 / −4) | `get_veff` (non-DF): the same protocol; on the first FP64 K iteration after FP32 K it drops `vj_last` so J is rebuilt from the full density (a guard: FP32 K is refused without DF). `KohnShamDFT`: `mixed_precision = None` class attribute and two `_keys` entries |
| 5 | `gpu4pyscf/dft/numint.py` (+29 / −20) | `_vv10nlc(..., uwe_kernel=None)` and `nr_nlc_vxc(..., vv10_kernel=None)`: with `None` the stock statements run unchanged |
| 6 | `gpu4pyscf/dft/vv10_mixed.py` (new, +915) | FP32 and df64 VV10 U/W/E kernels (NVRTC via `cupy.RawModule`, no fast-math), bound once per device after a forced probe; `uwe_stock`; `certify` |
| 7 | `gpu4pyscf/dft/tests/test_mixed_precision.py` (32 tests), `test_mixed_precision_vv10.py` (18 tests) | below |
| 8 | `README.md` | one line under experimental features |

**Default-off guarantee.** With `mf.mixed_precision` unset, `scf()` finds no policy and runs the
same `_kernel` calls as before (inside a `try` that only adds cleanup); `get_veff` sees
`_mixed_precision_state is None` and calls `ni.nr_rks` and `get_jk` exactly as before; `get_jk`
computes `k_fp32 = False`; `nr_nlc_vxc` is called without `vv10_kernel`, which is the stock
kernel. Tests: `test_default_is_none` (the attribute is `None`), `test_both_components_off_is_stock`
(a policy with every component off reproduces stock to 1e-11 Ha with the same cycle count and an
all-FP64 record), `test_all_off_is_stock` (VV10 file), and the upstream regression tests, which
pass identically on master and on this branch (Test plan). The cached FP64 XC path is asserted
bitwise equal to `nr_rks` (`test_fp64_cached_is_bitwise_stock_nr_rks`, with a negative control).

## Design points

- **Opt-in per object**, nothing enabled by GPU model; precedent: explicit reduced precision in
  RI-MP2 and TDDFT-RIS.
- **Refusals are loud** (`NotImplementedError` at `kernel()`): UKS/ROKS/GKS and spin ≠ 0;
  multi-GPU; with `xc`/`k`, range-separated and NLC functionals; `k` without density fitting; with
  `vv10`, no NLC term or more than one VV10 term, or a `NumInt` whose `nr_nlc_vxc` lacks
  `vv10_kernel`. A VV10 kernel that does not build or fails its probe raises `RuntimeError`.
- **Convergence contract** is structural, in `_kernel`, not a benchmark gate. DF-RKS rebuilds
  XC/J/K from the full density each iteration, so no FP32 contribution survives the switch.
  Without density fitting only XC can be FP32 (`k=True` is refused), and XC is rebuilt from the
  full density every iteration; the full-J-rebuild hook in `rks.get_veff` is a guard for the first
  FP64 K iteration and is not reached in a supported configuration.
- **With `xc` and `k` both on, K returns to FP64 no later than XC**; a k-only policy runs FP32 K
  under FP64 XC (`test_k_only_paracetamol`).
- **Switch triggers** are empirical controller settings (`xc_switch_tol=1e-3`,
  `k_switch_tol=1e-3`, `vv10_switch_tol=1e-5`, `stall=2`, `call_cap=30`), not error bounds. The
  stall and cap exist because a convergence tolerance below the FP32 noise floor may never be met
  in FP32.
- **Memory.** The AO cache is built once per SCF within `ao_cache_mem_fraction` (0.7) of free
  device memory, with its tier decided from the predicted size before allocation. If the FP32 copy
  does not fit, XC stays FP64 for that SCF and K follows; the record says why. Nothing is cached
  partially.
- **Backend-agnostic.** The K path calls `cupy_helper.contract`, so it inherits cuTENSOR when
  that is active.
- **Transparency.** `mf.mixed_precision_record` lists, per iteration, the precision of XC, K and
  VV10, the reason for each switch, the cache tier and the VV10 certificate.

## Evidence

Full detail, protocols, raw per-run data and the script that recomputes every number:
`benchmarks/mixed_precision/native/` on the fork's `mixed-precision-rfc-package-v2` branch
(`CLAIMS.md`, `verify_native.py`). Most of it is in the campaign configuration, which differs
from the library defaults: `ao_cache_fp64=True`, grid level 3 with `grids.prune = None`, B3LYP
`xc_switch_tol=3e-4`, cuTENSOR 2.3.1, def2-mTZVPP / def2-tzvpp-jkfit, `conv_tol=1e-9`, default
`conv_tol_grad`. The library defaults were measured separately on three molecules (CLAIMS C7).

- **Speed at the library defaults** (CLAIMS C7; paracetamol, propranolol, celecoxib; the default
  grid with `nwchem_prune`, no AO cache, default tolerances, the policies as in the Summary; one
  RTX PRO 6000 Workstation): r2SCAN 1.577 (per-cell 1.505–1.661), B3LYP 1.636 (1.561–1.728),
  wB97M-V 3.094 (2.875–3.258). The FP32 switch without the cache on the unpruned grid: 1.627 /
  1.954 / 2.884. The same pods re-ran the campaign configuration on those molecules at 1.678 /
  1.968 / 2.944 against the banked 1.638 / 1.860 / 2.803; B3LYP and wB97M-V are +0.108 and +0.141
  past the ±0.10 replication band, both faster (REPLICATION-FLAG; the figures below are not
  revised).
- **Speed, campaign configuration** (CLAIMS C1, C2): r2SCAN 1.657 (per-cell 1.523–1.871), B3LYP
  1.783 (1.584–1.994), wB97M-V 2.914 (2.593–3.250, five pods on three cards). Six molecules of
  63–98 atoms: r2SCAN 1.931, B3LYP 1.338. B3LYP's gain falls with size because the J/K stage,
  about half the stock SCF at 475–559 Da, gains only 5–7 % while XC gains 2–3×.
- **FP64-strong cards** (CLAIMS C3): L40S pays (1.729 / 1.524 / 2.994 for r2SCAN / B3LYP /
  wB97M-V). H100: 1.320–1.398 / 0.905–1.021 / 0.577–0.615; the r2SCAN gain there is almost all
  the FP64 AO cache (FP32 switch alone 1.084). A100: 11 of 12 clean cells neutral or slower.
- **Accuracy** (CLAIMS C4; 18 cells, mixed vs stock from the same guess): gradients within
  3.8e-9 – 5.3e-8 (r2SCAN), 2.1e-8 – 2.7e-7 (B3LYP), 1.2e-8 – 3.2e-8 (wB97M-V) Ha/Bohr; dipoles
  within 9.3e-6 D; every \|ΔE\| ≤ 1.6e-10 Ha; cycles equal in 17 of 18 cells. Against a tight
  reference both arms carry the same 6.6e-7 – 3.4e-6 Ha/Bohr residual, set by the default
  `conv_tol_grad`; at `conv_tol_grad = 3e-6` mixed tracks stock to 1.5e-9 – 2.3e-8 / 6.8e-9 –
  2.2e-7 / 1.1e-8 – 2.1e-8 Ha/Bohr with identical cycle counts.
- **Geometry optimisation** (CLAIMS C6; three geomeTRIC runs, stock gradients): identical step
  counts (23, 26, 16), endpoints within 9.9e-6 Å of stock, warm-step SCF 1.41–1.44 and whole
  optimisation 1.19–1.30 faster; every endpoint passes a stock certificate (\|ΔE\| ≤ 3.2e-9 Ha).

## Test plan

- **New tests, 50** (`dft/tests/test_mixed_precision.py`, 32; `test_mixed_precision_vv10.py`, 18):
  the default is `None` and an all-off policy is stock; enabled vs stock at r2SCAN, PBE, B3LYP,
  K-only, and XC without DF (\|ΔE\| ≤ 1e-8 Ha, cycles ±1, an FP64 tail, FP32 really used); against
  CPU PySCF; the convergence guard with every other trigger disabled (a forced switch); each
  refusal, including multi-GPU with a patched device count; the memory fallback; scanner geometry
  change and state cleared after `kernel()`; the FP32 XC and chunked FP32 K kernels against their
  FP64 counterparts; the FP64 AO cache (bitwise stock for LDA/GGA/meta-GGA, tiers, invalidation,
  empty-block refusal, opt-in default); the VV10 kernels per point against stock `_vv10nlc`, the
  df64 tail, the certificate (including a perturbed result that must fail it), an SCF that raises,
  and the all-off policy.
- **GPU validation of this branch (`4d2f3de`) on master `c1a6e37`** (`g4psrc r4`, three profiles;
  gpu4pyscf built from source each time, the released wheel removed, each tree importing itself):
  - **`pro6000-lock`** (engine-repo run `38022477235`; sm_120, one RTX PRO 6000 Blackwell, the
    validation lock's pins): upstream `df/tests/test_df_rks.py`, `df/tests/test_df_jk.py`,
    `dft/tests/test_rks.py`, `scf/tests/test_scf.py` give **44 passed, 1 skipped on master, and
    the same on this branch**; the new tests **50/50**; the VV10 kernels bitwise identical on the
    GPU to their NumPy emulations; stock vs mixed on paracetamol, propranolol and celecoxib
    **\|ΔE\| ≤ 5.9e-12 Ha with identical cycle counts** and an FP64 tail in every run;
  - **`pro6000-pyscf28`** (run `38023769024`; the same card, upstream's `requirements.txt` then
    **pyscf 2.8**, the multi-GPU CI job's pin): the same gates, 44 passed + 1 skipped on both trees,
    50/50, **\|ΔE\| ≤ 2.7e-12 Ha**;
  - **`v100-ci`** (run `38024957471`; **one Tesla V100, sm_70**, built from source for sm_70,
    upstream's `requirements.txt` then **libxc 0.9.0**, the single-GPU CI job's set): regression
    files, new tests (**50/50 in 133 s**) and VV10 bitwise identity pass; the three-molecule
    comparison was not run there (its bands are banked on the PRO 6000).
- Earlier validations (r1–r3 on earlier master commits, and the same code overlaid on the v1.8.1
  wheel; `benchmarks/mixed_precision/validation/`) passed the same gates. A multi-GPU host, scipy
  1.17 and upstream's full suite were not run.

### CI fit

- **Multi-GPU runner.** `check_supported` refuses `num_devices > 1`. Every test class that runs an
  SCF with a policy carries `@unittest.skipIf(num_devices > 1, ...)`, the pattern upstream's own
  tests use, so the 2×T4 job skips them; the refusal itself is asserted by `MultiGPU` with a
  patched device count, so it runs on any host.
- **V100 job.** sm_70 needs nothing special: the VV10 kernels are NVRTC `cupy.RawModule`s with
  `--std=c++14` and no `__CUDA_ARCH__` branching (intrinsics `__fadd_rn`, `__fsub_rn`,
  `__fmul_rn`, `__syncthreads`, `__longlong_as_double`). On one V100 the 50 tests take 133 s
  serially, the slowest 18.3 s (`test_vv10_paracetamol`); the rest are under 18 s each.
- **Pins.** Passes under pyscf 2.8 and under libxc 0.9.0 (above). With libxc 0.9.0, upstream's own
  `test_rks.py::test_nr_coach` fails on master and on this branch alike (`COACH` is not in that
  wheel); it is unrelated to this change. scipy 1.17 was not tested.

## Limitations

- **Most speed numbers are at non-default settings** (above). The library defaults are measured
  only on three molecules, one card and one basis.
- **FP64-strong GPUs.** The FP32 switch is slower on the H100 and A100 tested, and nothing in the
  code warns or refuses there. Only two FP64:FP32 throughput levels were measured, so no threshold
  is established. Open question: a device guard by compute capability, or documentation only?
- **At the library defaults** (CLAIMS C7, three molecules, one RTX PRO 6000): every reading PAYS
  (≥ 1.15 by the campaign's rule): r2SCAN 1.577, B3LYP 1.636, wB97M-V 3.094 trio geomeans. The
  default grid has about 0.63 of the unpruned grid's points and stock is 1.356 / 1.278 / 1.035
  faster on it; without the cache on the unpruned grid the mode reads 1.627 / 1.954 / 2.884. One
  pre-registered prediction missed: B3LYP without the cache, 1.954 against 1.65–1.90, the cache
  adding only 1.007 for B3LYP. The same pods read the campaign configuration +0.108 (B3LYP) and
  +0.141 (wB97M-V) above the banked trio, past the ±0.10 replication band, both faster
  (REPLICATION-FLAG).
- **Cold start.** On a fresh Blackwell machine the first SCF took 64–168 s. It is the CUDA
  driver's JIT cache, paid once per machine; stock pays it too (134.0 s when it runs first). It is
  not the mode's cost.
- **Not measured:** running without cuTENSOR, UKS, multi-GPU, the RTX 5090 and other FP64-strong
  parts (B200, V100), MIG beyond a one-instance projection (dropped from the claims; for
  throughput of many small SCFs, four processes under MPS gave 1.52× stock and 1.75× mixed the
  serial throughput on one pod, which is deployment advice, not a library property).
- **Investigated, not proposed:** a DIIS reset at the switch (made the endpoint worse) and a
  warm-start rule for geometry optimisations (never triggered).

## VV10 component (`MixedPrecision(vv10=True)`)

Independent of `xc`/`k` (which still refuse NLC functionals) and proposed as a separate step if
the maintainers prefer. It replaces only the VV10 pair sum of `numint._vv10nlc` for functionals
with exactly one VV10 term, such as wB97M-V. Early calls use FP32 pair terms with FP64
accumulation and hi/lo-split coordinates; a one-way switch on \|ΔE_nlc\| < 1e-5 Ha (same stall and
cap) moves every later call to a df64 kernel (double-float on FP32 pairs, ~48-bit). There is no
stock FP64 tail call; instead, when the SCF ends, the stock kernel re-runs on the last df64 call's
inputs, inside the SCF timer, with bands 1e-10 relative on E/U/W and 1e-11 Ha on E_nlc; out of
band, `mf.converged = False` and `RuntimeError`. Range-separated functionals and the non-DF path
are allowed for this component. The seam in `numint` is two keyword arguments with default
`None`.
