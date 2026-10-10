# GPU validation of the master-based code branch, built from source (g4psrc)

The branch proposed upstream, `chris-lee-mc/gpu4pyscf@mixed-precision-upstream`, was validated on
upstream master with gpu4pyscf **built from source**: for sm_120 on one RTX PRO 6000 Blackwell
Workstation Edition (r1–r4), and in r4 also for sm_70 on one Tesla V100-SXM2-16GB. The protocol,
gates, profiles and pre-dispatch sections are in the engine repository:
`docs/upstream/PORT-PLAN-upstream-source-build.md`, in `chris-lee-mc/gpu-conformer-engine`.

The method is one build and two trees:
- gpu4pyscf's CUDA libraries are built once, from the base commit, with
  `-DCUDA_ARCHITECTURES=<arch>-real`, then copied byte for byte into the branch tree. The branch
  changes only `.py` files, and the pod checks this.
- Stock master and the branch then run the same tests with the same libraries.
- The released wheel is uninstalled first, and every test run records which tree it imported.

## Runs

| run | fork commit | upstream master | profile (card; packages after the lock) | result |
|---|---|---|---|---|
| r1 (`37723300240`) | `5b9c87c0` | `eef5f5b3` | PRO 6000; the lock | PASS: regression 43/43 on both trees, 49/49 new tests, trio \|ΔE\| ≤ 5.9e-12 Ha |
| r2 (`37881608304`) | `122e2d5e` | `82bc7028` | PRO 6000; the lock | PASS: regression 44 passed + 1 skipped on both trees, 49/49, \|ΔE\| ≤ 6.8e-12 Ha |
| r3 (`38017282503`) | `ac4c668e` | `c1a6e371` | PRO 6000; the lock | PASS: regression 44 passed + 1 skipped on both trees, 49/49, \|ΔE\| ≤ 6.4e-12 Ha |
| **r4 `pro6000-lock`** (`38022477235`) | **`4d2f3de4`** | **`c1a6e371`** | PRO 6000; the lock (pyscf 2.14.0, cupy 14.1.1, libxc 0.8.1, no cutensor) | **PASS G1–G8**: regression 44 passed + 1 skipped on both trees, 50/50, \|ΔE\| ≤ 5.9e-12 Ha |
| **r4 `pro6000-pyscf28`** (`38023769024`) | **`4d2f3de4`** | **`c1a6e371`** | PRO 6000; upstream's `requirements.txt`, then pyscf 2.8.0 and libxc 0.8.1 (cupy 13.4.1, scipy 1.15.3, cutensor 2.2.0) | **PASS G1–G8**: regression 44 passed + 1 skipped on both trees, 50/50, \|ΔE\| ≤ 2.7e-12 Ha |
| **r4 `v100-ci`** (`38024957471`) | **`4d2f3de4`** | **`c1a6e371`** | Tesla V100-SXM2-16GB (sm_70, `70-real`); upstream's `requirements.txt`, then libxc 0.9.0 (pyscf 2.14.0, cupy 13.4.1, cutensor 2.2.0) | **PASS G1–G7**; G8 skipped by profile, no verdict. Regression 44 passed + 1 failed on both trees (upstream's own, below); 50/50 new tests in 133 s |

r4 validates the current head of the branch, `4d2f3de4`, on the same base as r3. The change since
r3 is tests only: `@unittest.skipIf(num_devices > 1, ...)` on the test classes that run an SCF with
a policy, and a `MultiGPU` test that patches the device count and asserts the refusal, so the new
tests number 50. The earlier heads are kept on the fork as `mixed-precision-upstream-on-eef5f5b`
(r1), `-on-82bc702` (r2) and `-on-c1a6e37` (r3, `ac4c668e`). The `v100-ci` run is the one
infrastructure re-run of `38023715438`, which found no V100 capacity and created no pod. The
records are `g4psrc_r2.json`, `g4psrc_r3.json`, `g4psrc_r4_pro6000-lock.json`,
`g4psrc_r4_pro6000-pyscf28.json` and `g4psrc_r4_v100-ci.json`: the pod's JSON, with the build log
tail dropped and the run id and status added.

**Why three profiles.** Upstream's `unittest.yml` runs its single-GPU job on a V100 with libxc
0.9.0 and its multi-GPU job on two T4s with pyscf 2.8; r1–r3 ran neither card nor package set. A
profile names the card and the packages installed after the lock, and the gate set: `v100-ci`
omits G8 because its bands and walls are banked on the PRO 6000 and the trio's tier-L estimate does
not fit a 16 GB card beside another tenant.

## Gates

| gate | r3 | r4 `pro6000-lock` | r4 `pro6000-pyscf28` | r4 `v100-ci` |
|---|---|---|---|---|
| G1 commits | HEAD is the named SHA; the base is an ancestor | same | same | same |
| G2 scope | 8 files under `gpu4pyscf/`, all `.py`, plus `README.md`; no compiled input differs | same | same | same |
| G3 build | rc 0, 12 libraries, branch copies byte-identical | 12 libraries, 124 s | 12 libraries, 138 s | 12 libraries for `70-real`, 362 s |
| G4 provenance | the wheel was removed; each tree imported itself; libxc came from its wheel | same, and the installed versions equal the profile's (new in r4) | same | same |
| G5 regression | upstream `test_df_rks`, `test_df_jk`, `test_rks`, `test_scf`: no test that passed on master failed or skipped on the branch | 44 passed + 1 skipped / same | 44 passed + 1 skipped / same | 44 passed + 1 failed / same (the failure is upstream's, below) |
| G6 new tests | `test_mixed_precision.py` (31) and `test_mixed_precision_vv10.py` (18): 49/49, no skips | 32 + 18: 50/50, no skips | 50/50, no skips | 50/50, no skips, 133 s |
| G7 VV10 identity | the FP32 and df64 launchers are bitwise identical to the NumPy emulation | bitwise | bitwise | bitwise, on sm_70 |
| G8 trio | paracetamol, propranolol and celecoxib × r2SCAN, B3LYP, wB97M-V: \|ΔE\| ≤ 1e-8 Ha, cycles ±1, an FP64 tail, real FP32 use; VV10 certificate in band | \|ΔE\| ≤ 5.9e-12 Ha, cycles identical | \|ΔE\| ≤ 2.7e-12 Ha, cycles identical | not run (skipped by profile) |

**Disclosed, not gated (r4).**
- **The V100 regression failure is upstream's.** `dft/tests/test_rks.py::KnownValues::test_nr_coach`
  fails on master and on the branch alike with `KeyError: "LibXCFunctional: name 'COACH' not
  found."`: the libxc 0.9.0 wheel that upstream's V100 job installs does not name the functional
  that upstream #938 added. G5 records base failures and does not gate them; on this pin set no
  test is skipped, which is why the base's skip count is 0 rather than 1.
- **V100 durations, new tests only** (`pytest --durations=20`, in the JSON): 133 s in all. The
  slowest single test is 18.3 s (`test_vv10_paracetamol`), then 17.5 s
  (`test_kernels_against_stock_vv10nlc`), 11.8 s and 10.0 s; everything else is under 10 s. One
  RunPod V100, serially, not upstream's runner with `-n 4`.
- **The contraction engine** read `cupy` under the lock and `None` under both `requirements.txt`
  pin sets, on both trees. Recorded only.

**Not covered:** multi-GPU (no T4 or multi-GPU pod was rented; the `skipIf(num_devices > 1)`
decorators never fire on one card, and the skip path is exercised only by the patched refusal
test), scipy 1.17 (it declares `requires_python >= 3.11` and the image ships 3.10; r4 Amendment 1
kept the lock's 1.15.3), upstream's full test suite, other architectures than sm_120 and sm_70,
and speed. The walls in the JSON are one run each and are not speed readings.
