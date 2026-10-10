# GPU validation of the master-based code branch, built from source (g4psrc)

The branch proposed upstream, `chris-lee-mc/gpu4pyscf@mixed-precision-upstream`, was validated on
upstream master with gpu4pyscf **built from source** for sm_120 on one RTX PRO 6000 Blackwell
Workstation Edition. The protocol, gates and pre-dispatch sections are in the engine repository:
`docs/upstream/PORT-PLAN-upstream-source-build.md`, in `chris-lee-mc/gpu-conformer-engine`.

The method is one build and two trees:
- gpu4pyscf's CUDA libraries are built once, from the base commit, with
  `-DCUDA_ARCHITECTURES=120-real`, then copied byte for byte into the branch tree. The branch
  changes only `.py` files, and the pod checks this.
- Stock master and the branch then run the same tests with the same libraries.
- The released wheel is uninstalled first, and every test run records which tree it imported.

## Runs

| run | fork commit | upstream master | result |
|---|---|---|---|
| r1 (`37723300240`) | `5b9c87c0` | `eef5f5b3` | PASS: regression 43/43 on both trees, 49/49 new tests, trio \|ΔE\| ≤ 5.9e-12 Ha |
| r2 (`37881608304`) | `122e2d5e` | `82bc7028` | PASS: regression 44 passed + 1 skipped on both trees, 49/49, \|ΔE\| ≤ 6.8e-12 Ha |
| **r3 (`38017282503`)** | **`ac4c668e`** | **`c1a6e371`** | **PASS: regression 44 passed + 1 skipped on both trees, 49/49, \|ΔE\| ≤ 6.4e-12 Ha** |

r3 is the current head of the branch. It is the same four commits as r2, rebased with no conflicts
onto two newer upstream commits, one of which changed `numint.eval_ao`. The last commit is amended
with two docstring corrections. The records are `g4psrc_r2.json` and `g4psrc_r3.json`: the pod's
JSON, with the build log tail dropped and the run id and status added.

## Gates (all hold on r3)

| gate | r3 |
|---|---|
| G1 commits | HEAD is the named SHA; the base is an ancestor |
| G2 scope | 8 files under `gpu4pyscf/`, all `.py`, plus `README.md`; no compiled input differs |
| G3 build | rc 0, 12 libraries, branch copies byte-identical |
| G4 provenance | the wheel was removed; each tree imported itself; libxc came from its wheel |
| G5 regression | upstream `test_df_rks`, `test_df_jk`, `test_rks`, `test_scf`: no test that passed on master failed or skipped on the branch |
| G6 new tests | `test_mixed_precision.py` (31) and `test_mixed_precision_vv10.py` (18): 49/49, no skips |
| G7 VV10 identity | the FP32 and df64 launchers are bitwise identical to the NumPy emulation |
| G8 trio | paracetamol, propranolol and celecoxib × r2SCAN, B3LYP, wB97M-V: \|ΔE\| ≤ 1e-8 Ha, cycles ±1, an FP64 tail, real FP32 use; VV10 certificate in band |

**Not covered:** other architectures, upstream's full test suite, and speed. The contraction
engine on these builds was `cupy`, not cuTENSOR; the walls in the JSON are one run each and are
not speed readings.
