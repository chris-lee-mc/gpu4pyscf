# Evidence bundle: opt-in mixed-precision DF-RKS on FP64-limited GPUs

This bundle backs the GPU4PySCF RFC "opt-in mixed-precision DF-RKS with FP64 convergence on
FP64-limited GPUs". Every aggregate the RFC quotes can be re-derived from the per-molecule tables
here:

```
python verify_aggregates.py      # stdlib only; exit 0 = every quoted aggregate reproduced
```

The **native-port campaign** is in [`native/`](native/README.md): the fork's own port, measured as a
user runs it, on the RTX PRO 6000 (Workstation and Server), L40S, H100, A100 and RTX PRO 6000 MIG
1g.24gb instances, under pre-registered protocols. Its claims are in
[`native/CLAIMS.md`](native/CLAIMS.md), and `python native/verify_native.py` reproduces them.

## Using the mode

```python
import gpu4pyscf
from gpu4pyscf.dft.mixed_precision import MixedPrecision

mf = gpu4pyscf.dft.RKS(mol, xc='b3lyp').density_fit()
mf.mixed_precision = MixedPrecision(xc=True, k=True)   # default: None, i.e. stock
# The speed numbers in native/ were measured with, in addition, ao_cache_fp64=True,
# xc_switch_tol=3e-4 for B3LYP, mf.grids.prune = None and cuTENSOR. The line above
# (library defaults, pruned grid) is NOT measured; expect a smaller gain.
mf.kernel()
print(mf.mixed_precision_record)   # the precision of each iteration, and why it switched
```

- **Who it is for.** GPUs whose FP64 throughput is a small fraction of FP32. Measured on the RTX
  PRO 6000 Blackwell (Workstation and Server) and the L40S; consumer RTX cards (e.g. the 5090) were
  not measured.
- **Do not enable it on the FP64-strong data-centre GPUs tested.** On the H100 and A100 it made
  wB97M-V 1.6–2× slower and B3LYP up to 1.4× slower. B3LYP on two H100 pods read 0.905–1.021,
  i.e. HARMS to NEUTRAL. r2SCAN still gained 1.32–1.40× on the H100, almost all of it from the FP64
  AO cache; the FP32 switch itself was neutral (1.08) ([`native/CLAIMS.md`](native/CLAIMS.md) C3).
  - Other FP64-strong parts (e.g. B200, V100) were not measured.
  - The code does **not** detect any of this: nothing warns or refuses on such a device.
  - Only two levels of FP64:FP32 throughput were measured, so no threshold is established.
  - On those cards the FP64 AO cache alone (`MixedPrecision(ao_cache_fp64=True)`) pays about
    1.15–1.34× for r2SCAN and B3LYP, and is neutral for wB97M-V.
- **Opt-in only.** It is off unless `mf.mixed_precision` is set. With it unset, nothing changes.
- **What runs in FP32.**
  - XC: the density and XC-potential contractions, against an FP32 copy of the AO values that is
    built once per SCF. Functional evaluation, grid weights, the electron count, the XC energy and
    the potential-matrix accumulator stay FP64.
  - K, for global hybrids with density fitting only: the two exchange contractions. J and the K
    accumulator stay FP64.
- **Accuracy contract.** Each component switches one way, FP32 → FP64 (→ df64 for VV10, below).
  The switch comes from an energy-change threshold, a stall detector or an iteration cap. The SCF
  cannot report convergence until two consecutive iterations were built without FP32-phase
  arithmetic; if the convergence tests pass earlier, the switch is forced and the SCF continues.
- **Refused with `NotImplementedError` at `kernel()`, never silently ignored:**
  - UKS / ROKS / GKS, and RKS with spin ≠ 0 (every component);
  - multi-GPU (every component);
  - with `xc=True` or `k=True`: range-separated functionals, and NLC functionals (use
    `vv10=True` alone for those);
  - `k=True` without density fitting;
  - with `vv10=True`: a functional with no NLC term, more than one VV10 term, or VV10
    coefficients outside b > 0 / finite, or a custom `NumInt` whose `nr_nlc_vxc` does not
    accept the `vv10_kernel` keyword.

  A VV10 kernel that does not build or fails its forced probe raises `RuntimeError` at the start
  of the SCF.
- **Memory.** The AO cache is capped at a fraction of free GPU memory (`ao_cache_mem_fraction`,
  default 0.7). If it does not fit, XC stays FP64 for that SCF, K follows it (K is never FP32 while XC is FP64), and the record says why.
- **FP64 AO cache (opt-in, `ao_cache_fp64=True`, default `False`).**
  - It also keeps an FP64 copy of the AO values, so the FP64 XC calls of the tail skip the AO
    evaluation. The cached FP64 path is bitwise identical to stock `nr_rks`.
  - Its tier is chosen from the predicted size before allocation: `fp64+fp32`, else `fp32` only,
    else none. `mixed_precision_record['ao_cache_tier']` reports it.
  - With a hybrid, the DF tensor is built first, so that DF places it as it would without the
    cache.
  - Every speed number in `native/` was measured **with** this cache on, with the other settings
    listed in [`native/CLAIMS.md`](native/CLAIMS.md), which differ from the library defaults.
- **Not covered.** Gradients and Hessians are unchanged: they run stock, in FP64.

## VV10

`MixedPrecision(vv10=True)` treats the VV10 nonlocal-correlation pair sum of `numint._vv10nlc`
(the O(N²) U/W/E kernel over the masked nlc grid). It is a separate component, for functionals
with exactly one VV10 term such as wB97M-V, and runs **alone**: `xc=True` / `k=True` still refuse
NLC functionals. Range-separated functionals and the non-DF path are allowed for it.

```python
mf = gpu4pyscf.dft.RKS(mol, xc='wb97m-v').density_fit()
mf.mixed_precision = MixedPrecision(vv10=True)
mf.kernel()
rec = mf.mixed_precision_record
rec['vv10']        # per call: 'fp32' ... then 'df64' ...
rec['vv10_cert']   # the stock FP64 certificate on the last call
```

- **Phases.** Early calls run the pair sum with FP32 pair terms and FP64 accumulation (variant
  `f32_t1_hilo`, hi/lo-split coordinates). The switch is one way, on |ΔE_nlc| between calls below
  `vv10_switch_tol` (default 1e-5 Ha), with the same stall detector and cap. After it, **every**
  call runs a df64 kernel (`df64_f32_t1`: FP32-pair double-float arithmetic, ~48-bit). There are
  no stock FP64 tail calls. Everything outside the pair sum (density on the grid, masking, the
  potential matrix) is stock FP64.
- **Certificate.** When the SCF ends, the stock FP64 kernel re-runs on the inputs of the last
  df64 call, inside the SCF timer. The bands are 1e-10 relative on E/U/W and 1e-11 Ha on E_nlc.
  Out of band (or if certifying itself fails), the record is stored, `mf.converged` is set to
  False and `RuntimeError` is raised. A run that ends in the FP32 phase has no certificate
  (`vv10_cert` is None). An SCF that raises skips it.
- **The guard.** The two-iteration rule above treats a df64 VV10 call as clean and an FP32 one as
  not, so convergence is accepted only on a df64 (or, with the component off, stock) tail.
- **Record.** `vv10`, `vv10_n_masked`, `vv10_switch_call`, `vv10_switch_reason`, `vv10_n_fp32`,
  `vv10_n_df64`, `vv10_tol`, `vv10_kernel` (variant, entry and build status of both kernels),
  `vv10_cert` and `tail_precision`.
- **Seam.** `numint._vv10nlc` and `numint.nr_nlc_vxc` gain the keywords `uwe_kernel=` /
  `vv10_kernel=`, default None (the stock kernel). With None the stock code path is unchanged.

**Measured** (external prototype, `data/vv10_wb97mv_pro6000.csv`; RTX PRO 6000 Blackwell,
wB97M-V / def2-mTZVPP, DF, nlcgrids at the library default). Whole-SCF e2e is the stock FP64 wall
÷ the treated wall on the same pod, certificate included:

| stage | run | paracetamol | propranolol | celecoxib | worst \|ΔE\| (Ha) |
|---|---|---|---|---|---|
| FP32 + stock tail, tol 1e-4 | 36781246916 | 1.555 | 1.393 | 1.363 | 9.1e-13 |
| FP32 + stock tail, tol 1e-5 | 36792813279 | 1.690 | 1.649 | 1.592 | 4.5e-13 |
| **FP32 + df64 tail + certificate (ADOPTed)** | **36875554583** | **2.809** | **2.965** | **2.635** | **9.1e-13** |

- ADOPTed run: celecoxib 92.16 s → 34.98 s, of which the certificate is 5.07 s; cycle counts
  identical to stock on all three; certificates 7.8e-14 / 1.6e-13 / 1.5e-13 relative. Against
  stock GPU4PySCF's own warm wall, the geomean is 2.798.
- Kernel level, celecoxib: the FP32 pair sum is 30.4× faster than the stock UWE kernel (0.168 s
  vs 5.12 s); the df64 kernel 5.32× (0.959 s vs 5.10 s). Kernel error against stock: FP32 ≤ 1.16e-7
  relative and 1.17e-10 Ha in E_nlc; df64 ≤ 2.3e-13 and 1.3e-14 Ha.
- **Not measured:** other cards, other functionals or nlcgrids, gradients, open shell, molecules
  beyond these three. These numbers are the prototype's. The port's own GPU validation is a
  correctness gate, and passed: see `validation/VALIDATION-vv10.md`.

## Contents

| path | what it is |
|---|---|
| `data/r2scan_ladder_pro6000.csv` | r2SCAN, 24 molecules (S/M/L tiers), RTX PRO 6000, CuPy/einsum, `conv_tol_grad=1e-5`. Stock and treated warm walls, cycles, \|ΔE\|. The dispatch-B paracetamol row is a control, marked `canary_excluded`. |
| `data/b3lyp_cg_rebank_pro6000.csv` | B3LYP, 24 molecules, RTX PRO 6000, CuPy/einsum, library-default `conv_tol_grad`. Stock, XC-only and XC+K arms. |
| `data/b3lyp_cutensor_aba_pro6000.csv` | B3LYP, L tier, one pod, einsum → cuTENSOR → einsum legs (3 × 4 rows). |
| `data/r2scan_gradient_xc.csv` | Analytic-gradient XC experiment on RTX PRO 6000 and A100. Exploratory; not part of the proposal. |
| `data/r2scan_energy_trio_by_gpu.csv` | Three molecules on RTX PRO 6000, H100 and A100, energy path. |
| `data/vv10_wb97mv_pro6000.csv` | wB97M-V VV10, three molecules × six runs, RTX PRO 6000: two kernel commissionings (FP32, df64) and four whole-SCF runs (FP32 + stock tail at tol 1e-4 and 1e-5; df64 tail, shadowed and production). |
| `data/SOURCES.md` | Per table: the CI run, its attempt and commit, and the GPU string, pins and settings as recorded by that run. |
| `verify_aggregates.py` | Recomputes each quoted geomean, range, ratio and bound, and checks it against the quoted value. |
| `geometries/*.xyz` + `SHA256SUMS` | The 24 fixed input geometries (RDKit ETKDGv3 + MMFF, conformer 0; the recipe is in each file's comment line). |
| `requirements.lock`, `ENVIRONMENT.md` | The exact Python pins, CUDA, driver, card, and numerical settings. |
| `validation/` | GPU validation of this branch's implementation (correctness gate). |
| `stock_baseline.py` | Standalone script, no dependencies beyond GPU4PySCF, that runs the exact **stock** calculation every ratio is measured against. Use it to check the reference side on your own card. |

## Reading the numbers

- **Whole-SCF speedup** is stock warm wall ÷ treated warm wall, same pod, same run, same settings
  and contraction backend. It is not a kernel-only speedup, and not a geometry-optimization
  measurement.
- **Incremental K ratio** is XC-only wall ÷ XC+K wall. It includes any change in cycle count.
- The r2SCAN (`conv_tol_grad=1e-5`) and B3LYP (library default) tables use different SCF
  convergence settings, and should not be pooled.
- **VV10 e2e** is the spec side's own stock FP64 base arm ÷ treated, the ratio the VV10 RESULTs
  quote; the CSV also carries the stock GPU4PySCF warm wall and that ratio. The commissioning rows'
  walls are single runs with a shadow re-run inside the treated wall, and are not speed
  measurements.
- **H100 and A100 energy rows** were measured in a verification configuration: some stock re-runs
  are charged to the treated arm. The "net" columns subtract that recorded time, so they are
  reconstructions, not production measurements.
- **The A100 whole-SCF deficit** is dominated by a one-off excursion in the untreated
  density-fitting build for celecoxib. The phase columns show it.
- **Energy differences** are against stock GPU4PySCF at matched settings, not CPU PySCF.

## Known issues in the numbers, disclosed

1. **One rounding disagreement.** Our internal record quoted the A100 celecoxib `df_build` delta
   as 2.3383 s. The unrounded values give 2.33825 s (2.3382 at 4 dp); the quoted value came from
   subtracting already-rounded numbers. `verify_aggregates.py` reports it as a known discrepancy.
   It is not adjusted.
2. **A100 energy table.** In our internal record of that run, the column labelled as the stock
   wall was in fact the treated implementation's own FP64 control arm. The ratios quoted in the
   RFC (e2e geomean 0.971) are stock GPU4PySCF ÷ treated, and are correct. The CSV carries both
   ratios, and `data/SOURCES.md` explains the difference.
3. **Rounding of the worst |ΔE|.** In the cuTENSOR comparison, the worst |ΔE| is 1.71e-10 Ha
   (omeprazole), quoted as "1.7e-10".

## Relation to the implementation on this branch

The numbers in `data/` were measured with an **external prototype** built on instance-level hooks,
not with the code on this branch. The branch is a native port of the same scheme
(`gpu4pyscf/dft/mixed_precision.py` and `gpu4pyscf/dft/vv10_mixed.py`, with hooks in `dft/rks.py`,
`df/df_jk.py`, `dft/numint.py` and `scf/hf.py`). The VV10 kernel sources are byte-identical to the
prototype's.
Its own GPU validation is in `validation/`, as a correctness gate. The port's speed is measured
in [`native/`](native/README.md): 44 pods with the fork's own code. Those runs used an unpruned
grid, the FP64 AO cache and B3LYP `xc_switch_tol=3e-4`, not the library defaults; the defaults
themselves are not measured.

The source-document paths in `data/SOURCES.md` refer to the authors' own lab notebook, which is
not public. The CI run IDs and recorded settings next to them are the primary records.
