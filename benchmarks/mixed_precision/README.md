# Evidence bundle: opt-in mixed-precision DF-RKS on FP64-limited GPUs

This bundle backs the GPU4PySCF RFC "opt-in mixed-precision DF-RKS with FP64 convergence on
FP64-limited GPUs". Every aggregate the RFC quotes can be re-derived from the per-molecule tables
here:

```
python verify_aggregates.py      # stdlib only; exit 0 = every quoted aggregate reproduced
```

## Using the mode

```python
import gpu4pyscf
from gpu4pyscf.dft.mixed_precision import MixedPrecision

mf = gpu4pyscf.dft.RKS(mol, xc='b3lyp').density_fit()
mf.mixed_precision = MixedPrecision(xc=True, k=True)   # default: None, i.e. stock
mf.kernel()
print(mf.mixed_precision_record)   # the precision of each iteration, and why it switched
```

- **Who it is for.** GPUs whose FP64 throughput is a small fraction of FP32, such as RTX and
  workstation Blackwell cards. On FP64-strong data-centre GPUs (e.g. H100, A100) it may not pay
  off.
- **Opt-in only.** It is off unless `mf.mixed_precision` is set. With it unset, nothing changes.
- **What runs in FP32.**
  - XC: the density and XC-potential contractions, against an FP32 copy of the AO values that is
    built once per SCF. Functional evaluation, grid weights, the electron count, the XC energy and
    the potential-matrix accumulator stay FP64.
  - K, for global hybrids with density fitting only: the two exchange contractions. J and the K
    accumulator stay FP64.
- **Accuracy contract.** Each component switches one way, FP32 → FP64. The switch comes from an
  XC-energy-change threshold, a stall detector or an iteration cap. The SCF cannot report
  convergence until two consecutive iterations were built entirely in FP64; if the convergence
  tests pass earlier, the switch is forced and the SCF continues.
- **Refused with `NotImplementedError` at `kernel()`, never silently ignored:**
  - UKS / ROKS / GKS, and RKS with spin ≠ 0;
  - range-separated functionals;
  - NLC (e.g. VV10);
  - multi-GPU;
  - `k=True` without density fitting.
- **Memory.** The AO cache is capped at a fraction of free GPU memory (`ao_cache_mem_fraction`,
  default 0.7). If it does not fit, XC stays FP64 for that SCF, K follows it (K is never FP32 while XC is FP64), and the record says why.
- **Not covered.** Gradients and Hessians are unchanged: they run stock, in FP64.

## Contents

| path | what it is |
|---|---|
| `data/r2scan_ladder_pro6000.csv` | r2SCAN, 24 molecules (S/M/L tiers), RTX PRO 6000, CuPy/einsum, `conv_tol_grad=1e-5`. Stock and treated warm walls, cycles, \|ΔE\|. The dispatch-B paracetamol row is a control, marked `canary_excluded`. |
| `data/b3lyp_cg_rebank_pro6000.csv` | B3LYP, 24 molecules, RTX PRO 6000, CuPy/einsum, library-default `conv_tol_grad`. Stock, XC-only and XC+K arms. |
| `data/b3lyp_cutensor_aba_pro6000.csv` | B3LYP, L tier, one pod, einsum → cuTENSOR → einsum legs (3 × 4 rows). |
| `data/r2scan_gradient_xc.csv` | Analytic-gradient XC experiment on RTX PRO 6000 and A100. Exploratory; not part of the proposal. |
| `data/r2scan_energy_trio_by_gpu.csv` | Three molecules on RTX PRO 6000, H100 and A100, energy path. |
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
(`gpu4pyscf/dft/mixed_precision.py`, with hooks in `dft/rks.py`, `df/df_jk.py` and `scf/hf.py`).
Its own GPU validation is in `validation/`. That validation is a correctness gate, not a speed
measurement: re-measuring speed with the port is future work.

The source-document paths in `data/SOURCES.md` refer to the authors' own lab notebook, which is
not public. The CI run IDs and recorded settings next to them are the primary records.
