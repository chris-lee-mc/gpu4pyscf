# Provenance of the CSVs in this directory

Every row was extracted from the JSON line of a banked sentinel on the `ci-bus` branch, read with
`git show origin/ci-bus:<path>` (never checked out). Line 2 of each sentinel was compared with the
run id the source doc names; all matched. Every field below (RUN_ID, GPU model, pins, prewarm) is
quoted from the sentinel as read, not from the doc.

Common to every run below: `gpu4pyscf-cuda12x==1.8.1`, `pyscf==2.14.0`, `cupy-cuda12x==14.1.1`,
`numpy==2.2.6` (scfbench runs also record `scipy==1.15.3`). Timings are seconds, energies Hartree,
gradient deviations Hartree/Bohr. Ratios are dimensionless and computed from the unrounded sentinel
values.

## `r2scan_ladder_pro6000.csv`

Source doc: `docs/scf-perf/PREREG-m6-pro6000-ladder.md` (RESULT, dispatch A and dispatch B).
r2SCAN / def2-mTZVPP / def2-TZVPP-JKFIT, `spec_xc_mode=mp-prod`, `conv_tol_grad=1e-05`, `sides=g4p`.

| dispatch | run | attempt | sha | sentinel path | ci-bus commit | line 2 as read | `meta.gpu` as read | prewarm |
|---|---|---|---|---|---|---|---|---|
| A (S, M) | 35550452536 | 1 | `ea4f774a2b85` | `ci-results/scfbench-ea4f774a2b85.txt` | `785a2020` | `RUN_ID=35550452536-1-ea4f774a` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `WH=HIT KC=HIT arch=sm_120 reqhash=434eb1e7` |
| B (L) | 35551843564 | 1 | `561b6a205acd` | `ci-results/scfbench-561b6a205acd.txt` | `929ee3f8` | `RUN_ID=35551843564-1-561b6a20` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `WH=HIT KC=HIT arch=sm_120 reqhash=434eb1e7` |

JSON tag: `SCFBENCH_JSON`. `ratio_g4p_over_spec` = `g4p.wall_warm / spec.wall_warm`. The `xc_bucket_*`
columns come from the `spec.xcspeedup.<mol>` assertion (spec-side fp64 base arm XC bucket over the
treated XC bucket). The dispatch B paracetamol row has `role=canary_excluded` and is not part of any
tier aggregate.

## `b3lyp_cg_rebank_pro6000.csv`

Source doc: `docs/scf-perf/PREREG-cg-rebank.md` (RESULT S/M; RESULT L repeat).
B3LYP, `spec_xc_mode=mp-prod`, speck (fp32 K) `forced`, `conv_tol_grad=None` (gpu4pyscf's own).

| tiers | run | attempt | sha | sentinel path | ci-bus commit | line 2 as read | `meta.gpu` as read | STATUS |
|---|---|---|---|---|---|---|---|---|
| S, M | 36134049978 | 3 | `259f99de086c` | `ci-results/runs/scfbench-259f99de086c-36134049978-3-259f99de.txt` | `d4eb652a` | `RUN_ID=36134049978-3-259f99de` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | FAIL (2 FAILs: `spec.timer.metformin`, `spec.timer.paracetamol`) |
| L | 36152138663 | 1 | `0d9b7b32abac` | `ci-results/runs/scfbench-0d9b7b32abac-36152138663-1-0d9b7b32.txt` | `35c67335` | `RUN_ID=36152138663-1-0d9b7b32` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | PASS |

JSON tag: `SCFBENCH_JSON`. Arms: stock = `g4p.wall_warm`; stock fp64 base = `spec.wall_base`;
XC-only = `spec.wall_xconly`; XC+K = `spec.wall_warm`. `vs_stock` = stock / XC+K;
`k_ratio` (R_K) = XC-only / XC+K. `under_1s_timing_floor=yes` marks the rows whose `spec.timer`
record FAILed on the harness's 1 s wall floor; their walls are below 1 s and are the least reliable
timings in the table.

## `b3lyp_cutensor_aba_pro6000.csv`

Source doc: `docs/scf-perf/PREREG-cutensor-aba.md` (RESULT). Same configuration as the L re-bank,
three legs on one pod.

| run | attempt | sha | sentinel path | ci-bus commit | line 2 as read | `meta.gpu` as read (all three legs) | extra pin |
|---|---|---|---|---|---|---|---|
| 36214605134 | 1 | `76594d7e93d4` | `ci-results/runs/scfbench-76594d7e93d4-36214605134-1-76594d7e.txt` | `8b05a2ac` | `RUN_ID=36214605134-1-76594d7e` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `cutensor-cu12==2.3.1` |

JSON tags: `ABA_JSON_einsum1` (leg E1), `ABA_JSON_cutensor` (leg C), `ABA_JSON_einsum2` (leg E2).
`contract_engine_selected` is `meta.contract_engine.engine` of that leg (`cupy` / `cutensor` /
`cupy`). Columns as in the cg re-bank CSV.

## `r2scan_gradient_xc.csv`

Source doc: `docs/scf-perf/ARCH-GRADMP.md` §12 (PRO 6000) and §14.7 (A100).
SCF `conv_tol_grad`: `null` (gpu4pyscf's own default), as read from the PRO 6000 run's `GRADMP_JSON` config.
r2SCAN / def2-mTZVPP / def2-TZVPP-JKFIT, grid level 3, 3 warm repeats + 1 cold per arm.

| class | run | attempt | sha | sentinel path | ci-bus commit | line 2 as read | `meta.gpu` as read |
|---|---|---|---|---|---|---|---|
| pro6000 | 32322726163 | 1 | `7160b89d9a95` | `ci-results/gradmp-7160b89d9a95.txt` | `fc464705` | `RUN_ID=32322726163-1-7160b89d` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` |
| a100 | 32373234782 | 1 | `07a76faf10ee` | `ci-results/gradmp-07a76faf10ee.txt` | `679bcb15` | `RUN_ID=32373234782-1-07a76faf` | `NVIDIA A100-SXM4-80GB` |

JSON tag: `GRADMP_JSON`. `xc_excl_*` = `arms.<arm>.bucket_excl.xc`; `t_grad_*` =
`arms.<arm>.t_grad_warm_median`; `S_xc`, `S_grad`, `max_abs_dgrad_*` and `gshare_stock` are the
sentinel's own per-molecule fields.

## `r2scan_energy_trio_by_gpu.csv`

Source docs: `docs/scf-perf/SPEC-v2.md` §7.2 (H100), §7.3.2 and §7.3.3 (A100); PRO 6000 production
trio as quoted in `docs/scf-perf/ARCH-GRADMP.md` §1 and §12.1 (`s_energy=1.697`).
r2SCAN, `conv_tol_grad=1e-05`, `sides=g4p`.

| class | run | attempt | sha | sentinel path | ci-bus commit | line 2 as read | `meta.gpu` as read | `spec_xc_mode` | prewarm | STATUS |
|---|---|---|---|---|---|---|---|---|---|---|
| h100 | 32079364413 | 1 | `f50fb8d02f78` | `ci-results/scfbench-f50fb8d02f78.txt` | `6858a36d` | `RUN_ID=32079364413-1-f50fb8d0` | `NVIDIA H100 80GB HBM3` | `mp` (shadowed) | `WH=HIT KC=MISS arch=sm_90` | FAIL (speedup gates) |
| a100 | 32372635923 | 1 | `07a76faf10ee` | `ci-results/scfbench-07a76faf10ee.txt` | `56bbb9b8` | `RUN_ID=32372635923-1-07a76faf` | `NVIDIA A100-SXM4-80GB` | `mp` (shadowed) | `WH=HIT KC=MISS arch=sm_80` | FAIL (speedup gates) |
| pro6000 | 31978102688 | 1 | `d78a6340b31e` | `ci-results/scfbench-d78a6340b31e.txt` | `c44c5bfa` | `RUN_ID=31978102688-1-d78a6340` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `mp-prod` | not recorded | PASS |

SPEC-v2 names no sentinel path for the H100 run; the path was found by its RUN_ID in `ci-bus` history.
It is the tip of that path, and the only other version (`2da7d234`) is the aborted attempt
`32075407566`, not used. The d78a6340b31e path holds four PASS runs in history; the tip
(`c44c5bfa`) is the one named, `31978102688`.

JSON tag: `SCFBENCH_JSON`. `xc_base_s`, `xc_spec_s`, `xcspeedup`, `shadow_wall_s`,
`xc_spec_net_of_shadows_s` and `xcspeedup_net_of_shadows` are from the `spec.xcspeedup.<mol>`
assertion. Two end-to-end ratios are given because the docs use both walls:

- `e2e_g4p_over_spec` = `g4p.wall_warm / spec.wall_warm`. This is the harness's `spec.speedup` and
  the "e2e" the docs quote (A100 1.127 / 1.086 / 0.748; PRO 6000 geomean 1.697).
- `e2e_base_over_spec` = `spec.wall_base / spec.wall_warm`, using the spec side's own fp64 base arm.
  SPEC-v2 §7.3.2's table prints this `wall_base` beside the g4p-based e2e, so its rows do not divide
  out; the column is here so a reader can see which wall is which.

The `*_df_build_*` and `*_scf_iter_*` columns are `phases_warm` (g4p and spec) and `phases_base`
(spec fp64 base arm).

## `vv10_wb97mv_pro6000.csv`

Source docs: `docs/scf-perf/PREREG-vv10-commission.md`, `PREREG-vv10-production.md`,
`PREREG-vv10-tol5.md`, `PREREG-vv10-df64-commission.md` and `PREREG-vv10-df64-production.md`
(the RESULT section of each). wB97M-V / def2-mTZVPP / def2-TZVPP-JKFIT, grids level 3 with
`prune=None`, nlcgrids at the library default (level 3, `nwchem_prune`), `conv_tol=1e-9`,
`conv_tol_grad=1e-05`, cuTENSOR loaded on every run (engine `cutensor`). Besides the common pins
above, every run records `cutensor-cu12==2.3.1` and `scipy==1.15.3`; the two commissioning runs
also record `gpu4pyscf-libxc-cuda12x==0.8.1`.

| stage | run | attempt | sha | sentinel path | ci-bus commit | line 2 as read | `meta.gpu` as read | JSON tag |
|---|---|---|---|---|---|---|---|---|
| `fp32_commission` | 36769392379 | 1 | `e09687c1f732` | `ci-results/runs/vv10comm-e09687c1f732-36769392379-1-e09687c1.txt` | `a5d84509` | `RUN_ID=36769392379-1-e09687c1` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `VV10COMM_JSON` |
| `fp32_stock_tail_tol1e-4` | 36781246916 | 1 | `98d52242eafa` | `ci-results/runs/scfbench-98d52242eafa-36781246916-1-98d52242.txt` | `36720b5b` | `RUN_ID=36781246916-1-98d52242` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `SCFBENCH_JSON` |
| `fp32_stock_tail_tol1e-5` | 36792813279 | 1 | `95c93b39b5e7` | `ci-results/runs/scfbench-95c93b39b5e7-36792813279-1-95c93b39.txt` | `fd09b2dd` | `RUN_ID=36792813279-1-95c93b39` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `SCFBENCH_JSON` |
| `df64_commission` | 36869739118 | 1 | `e87884eb7077` | `ci-results/runs/vv10df64comm-e87884eb7077-36869739118-1-e87884eb.txt` | `c42a9a40` | `RUN_ID=36869739118-1-e87884eb` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `VV10DF64COMM_JSON` |
| `df64_tail_shadowed` | 36873230133 | 1 | `96d4a735a979` | `ci-results/runs/scfbench-96d4a735a979-36873230133-1-96d4a735.txt` | `223b798f` | `RUN_ID=36873230133-1-96d4a735` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `SCFBENCH_JSON` |
| `df64_tail_production` | 36875554583 | 1 | `96d4a735a979` | `ci-results/runs/scfbench-96d4a735a979-36875554583-1-96d4a735.txt` | `42f72efa` | `RUN_ID=36875554583-1-96d4a735` | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | `SCFBENCH_JSON` |

All six read `STATUS=PASS` on line 1 and `WH=HIT KC=HIT arch=sm_120 reqhash=434eb1e7` as prewarm.
The ADOPTed configuration is the last row (`spec_vv10=df64-prod`): FP32 pair sum `f32_t1_hilo`,
switch tolerance 1e-5, a df64 (`df64_f32_t1`) tail with no stock tail call, and a stock FP64
certificate on the last call. The R1 run of `PREREG-vv10-production.md` (`36777172798`, shadowed,
speed records `n/a-by-config`) is not tabulated; its R2 (`36781246916`) is.

Columns, by kind of run:

- **scfbench rows** (`fp32_stock_tail_*`, `df64_tail_*`). `wall_base_s` / `wall_spec_s` /
  `e2e_*` / `nlc_*` / `mechanism_*` are the `spec.vv10speedup.<mol>` assertion's own fields
  (`wall_base` is the spec side's stock FP64 base arm, `wall_spec` its warm treated run; e2e =
  `wall_base / wall_spec`, the RESULT tables' "e2e"). `g4p_wall_warm_s` is the stock g4p side's warm
  wall; `spec_speedup_g4p_over_spec` is the run-level `spec.speedup` per-molecule value
  (`g4p.wall_warm / spec.wall_warm`). Switch, call counts and tolerance are from
  `spec.vv10precision.<mol>`, |ΔE| and cycles from `spec.vv10energy.<mol>`, and the certificate from
  `spec.vv10shadow.<mol>`. `n_tail` is the stock FP64 calls (`n_fp64`) for a stock tail and the df64
  calls (`n_df64`) for a df64 tail. In `df64_tail_shadowed` the speed fields are recorded but the
  records are `n/a-by-config`: a stock shadow ran on every df64 call, so those walls are disclosed
  only.
- **Commissioning rows** (`fp32_commission`, `df64_commission`). `wall_base_s` / `wall_spec_s` are
  one stock SCF and one treated SCF (`runs.stock` / `runs.treated`), each a single run with the
  shadow included in the treated wall (`shadowed=yes`); `e2e_*` is left empty because neither
  RESULT quotes one. `abs_dE_spec_vs_base_Ha` is |E_treated − E_stock| of those two runs. The
  `fp32_variant` column is the variant the treated SCF ran: in `fp32_commission` that was the
  pre-commission default `f16_t2_hilo`. The `matrix_*` columns are the per-call kernel matrix of
  the variant that commissioning selected (`f32_t1_hilo`, `df64_f32_t1`):
  `matrix_kernel_median_s` and `matrix_stock_uwe_median_s` are the medians over all timed samples
  (calls 1, 2, mid and last, three repeats each) of that variant and of the stock UWE kernel on the
  same snapshot; `matrix_max_rel` is the worst of E/U/W over every call, and
  `matrix_max_abs_dEnlc_Ha` the worst |ΔE_nlc|. `30.4×` and `5.32×` are
  `matrix_stock_uwe_median_s / matrix_kernel_median_s` on celecoxib.
- **Certificate columns** are the `cert` record (`call`, `rel.E/U/W`, `denlc`, `wall_s`). The
  RESULTs quote the certificate's relative error as the worst of E/U/W.

Values are the unrounded sentinel numbers. The RESULT tables print rounded walls, and
recomputing a ratio from them can miss the printed ratio in the last digit (paracetamol,
`df64_tail_production`: 19.14 / 6.81 = 2.811, printed 2.809 from the unrounded 19.138 / 6.813).
`verify_aggregates.py` therefore checks each printed number, walls and ratios alike, against the
unrounded values at its printed precision.
