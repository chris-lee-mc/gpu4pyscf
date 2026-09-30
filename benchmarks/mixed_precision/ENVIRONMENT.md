# Environment of the banked measurements

**Python packages.** Every run installed exactly `requirements.lock` (14 pins), from a prebuilt
wheelhouse keyed on its hash. The ones that matter numerically:
- `gpu4pyscf-cuda12x==1.8.1`
- `gpu4pyscf-libxc-cuda12x==0.8.1`
- `pyscf==2.14.0`
- `cupy-cuda12x==14.1.1`
- `numpy==2.2.6`
- `cuda-pathfinder==1.6.0`

**The cuTENSOR leg only.** It additionally installed `cutensor-cu12==2.3.1` and put that wheel's
`cutensor/lib` directory on `LD_LIBRARY_PATH` before Python started. Without that, GPU4PySCF 1.8.1
fell back to CuPy/einsum in this stack. That leg verified cuTENSOR was loaded and selected.

**CUDA and GPU.**
- CUDA 12.8 container image.
- RTX PRO 6000 Blackwell Workstation Edition (sm_120, 96 GB), driver 580.126.20.
- The H100 80GB HBM3 and A100-SXM4-80GB rows used the same lock.
- Each CSV's GPU string is read from its run's own record (`data/SOURCES.md`).

**Numerical settings, both arms.**
- def2-mTZVPP basis, def2-TZVPP-JKFIT auxiliary basis.
- `grids.level=3`, `grids.prune=None`.
- `conv_tol=1e-9`, `max_cycle=60`.
- `conv_tol_grad`:
  - `1e-5`, set explicitly, for the r2SCAN ladder and the three-molecule energy runs on each GPU;
  - the library default (about 3.162e-5 at `conv_tol=1e-9`) for the B3LYP re-bank and the
    cuTENSOR comparison;
  - the library default for the gradient experiment (its record reads `"conv_tol_grad": null`).

**Timing.** Stock and treated arms ran on the same pod, in the same run. Warm-wall ratios are
stock / treated.
