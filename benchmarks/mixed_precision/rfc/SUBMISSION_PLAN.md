# Submission plan: the mixed-precision RFC to pyscf/gpu4pyscf

Planning document for the owner. Nothing here has been posted, and nothing here edits
`RFC_ISSUE.md`, `PR_BODY.md` or the code branch; section 6 lists the edits to make, for review
first. Every number is from `native/CLAIMS.md`, `validation/`, or `git` on the two branches, unless
it is marked **estimate**. Written 2026-10-10 against code branch `mixed-precision-upstream` @
`ac4c668e` (four commits on upstream master `c1a6e371`, which is still `upstream/master`'s head
after `git fetch upstream` today) and package branch `mixed-precision-rfc-package-v2` @ `ea3d748`.
**Later the same day** PREREG-9 (M1) and g4psrc r4 (G1, M2, G8) ran and the §6 edits were applied;
the branch head is now `4d2f3de4`. The commit SHAs and numstats below are updated to it; everything
else is left as planned, and the **status addendum at the end** says what is done and what is not.

## At a glance

| | |
|---|---|
| top blocker | the new tests **error** on upstream's 2×T4 CI runner (`num_devices > 1` refusal, no skip); fix before the issue (G1) |
| biggest evidence gap | no speed figure at library defaults (pruned grid, no cache); one pod answers it (M1) |
| recommended PR shape | a stack: XC+K first (`920bf64`, +928/−17, applies on master alone), VV10 and the AO cache as follow-ups; the cache commit only applies after VV10 unless re-authored |
| cheaper form of the `_kernel` intrusion | PySCF's `check_convergence` hook, which `soscf` already honours and `_kernel` does not (G3) |
| GPU budget | 2 measurement pods (M1, M2) + 1 g4psrc re-validation before the issue; 1 g4psrc per later rebase or stack member |
| fallback if declined | ask for two seams only (`check_convergence` parity, the `uwe_kernel` keyword) and ship the rest as a fork package |

## What upstream looks like (read from the repo, not from memory)

| fact | source |
|---|---|
| CI: `unittest.yml` runs `pytest -m 'not slow and not benchmark and not special'` on **one V100** (sm_70) **and on a 2×T4 runner** (`[self-hosted, Linux, X64, 2T4]`), on every PR to master; `lint.yml` runs ruff 0.15 + flake8; the nightly adds `--benchmark` on the V100 | `.github/workflows/unittest.yml`, `lint.yml`, `nightly_build.yml` |
| CI environment: `requirements.txt` pins cupy-cuda12x 13.4.1, **cutensor-cu12 2.2.0**, pyscf 2.14.0 | `requirements.txt` |
| 38 upstream test files guard single-GPU-only tests with `from gpu4pyscf.lib.multi_gpu import num_devices` + `@unittest.skipIf(num_devices > 1, '')` | `git grep num_devices upstream/master -- '*/tests/*.py'` |
| `CONTRIBUTING.md`: sm_70 required, no sm_80-only features; tests against CPU PySCF values; examples "highly recommended"; `to_cpu`/`to_gpu` where applicable | `CONTRIBUTING.md` |
| Reduced-precision precedents in tree: `mp/dfmp2.py` `fp_type` (class attribute from `__config__` key `gpu_mp_dfmp2_fp_type`, in `_keys`, forwarded as a kernel kwarg); `tdscf/ris.py` `single=True` constructor flag that **warns** when `conv_tol < 1e-5`; cmake `ENABLE_FP32_MULTIGRID` (PBC); `lib/cuest_wrapper.py` with int8 precision-control parameters and `examples/46-cuest.py` | `git grep -i 'float32\|fp32\|single'` |
| Convergence hook: `SCF.check_convergence = None` exists (`scf/hf.py:850`) and `scf/soscf.py:492` honours it (`if callable(mf.check_convergence): scf_conv = mf.check_convergence(locals())`, as PySCF 2.14 `hf.kernel` does), but **`hf._kernel` does not call it** | `git grep check_convergence upstream/master` |
| Merging: PRs land as merge commits made through the GitHub UI (committer `GitHub`, two parents). Of the last 120 commits, 17 PR-authored commits are Qiming Sun's; the last 25 PR commits range from 2 to 1668 insertions; the ≥ 800-line features (#917, #869, #942, #937) are all by collaborators. External `CONTRIBUTOR` PRs merged since June (#949, #931, #929, #923, #914) were merged within about a day and are all small fixes | `git log upstream/master`; read-only PR search |
| `CHANGELOG` is written at release time (one of the last 30 commits touched it); no issue or PR template | `git log -- CHANGELOG`, `.github/` |
| Master moved 25 commits in the last 30 days; 2 between r2 (`82bc702`) and r3 (`c1a6e37`) | `git log` |

## 1. Maintainer-perspective review of RFC_ISSUE.md / PR_BODY.md

Ranked by how likely the objection is to block, then by how costly it is to answer.

| # | likely objection | answered by | gap |
|---|---|---|---|
| 1 | **"Your tests will fail on our 2×T4 runner."** `check_supported` raises `NotImplementedError` when `num_devices > 1`; neither test file imports `num_devices` or skips, and the whole suite runs on the 2T4 job | nothing | **Hard CI red on every PR. Most of the 49 tests (every one that runs an SCF with a policy) would error there (estimate; exact count needs a 2-GPU host).** Fix: G1 below |
| 2 | **Maintenance burden**: +2931/−39 over nine files, two new ~950-line modules, NVRTC C++ in a `.py`, hooks in `scf/hf.py` (`_kernel` and `scf()`), `df_jk.get_veff`/`get_jk`, `rks.get_veff`, `numint` | PR_BODY "Default-off guarantee" + `test_both_components_off_is_stock`; r3 (and r4) regression 44 passed + 1 skipped on both trees; the split in §3 (first PR +928/−17, five files) | No external PR of this size has landed (table above). The `_kernel` intrusion (13 added lines) has a cheaper form: G3 |
| 3 | **"Does it pay at the library defaults?"** Measured with `grids.prune = None`, `ao_cache_fp64=True`, B3LYP `xc_switch_tol=3e-4`, cuTENSOR 2.3.1 | Disclosed in both texts and CLAIMS "configuration" table; "expect smaller gains on pruned grids. How much smaller is not measured" | **No number at defaults.** Also no quotable PRO 6000 figure for the FP32 switch *without* the cache (CLAIMS C3: the PRO 6000 precision effect exists only on CONTENTION-UNKNOWN C0 and is not claimed); the only clean no-cache reading is the L40S precision effect 1.655 / 1.463 / 2.981. Fix: M1 |
| 4 | **Tests on a V100**: 129.0 s for the 49 tests on a PRO 6000 (r3 `new_tests.wall_s`), unpruned level-3 grids, paracetamol def2-mTZVPP, `conv_tol=1e-10`; the NVRTC kernels have never been compiled for sm_70; the 16 GB T4 | sm_70: no `__CUDA_ARCH__` branching, intrinsics are `__fadd_rn/__fsub_rn/__fmul_rn/__syncthreads/__longlong_as_double` only (CHECKLIST §3); AO-cache tests fit the budget by construction (tier chosen before allocation) | **V100/T4 runtime and sm_70 compile unmeasured.** Fix: M2 (only if a V100/T4-class pod exists) or G1's skip plus a note |
| 5 | **FP64-strong cards are slower and nothing warns** | CLAIMS C3 (H100 wB97M-V 0.577–0.615; A100 11 of 12 clean cells neutral or slower); both texts say so; question 2 asks | Precedent for *warning* exists (`ris.py:894`). Compute capability alone cannot separate A100 (8.0) from L40S (8.9); a (major, minor) list is a heuristic. Fix: G4, only if asked |
| 6 | **One card family for the headline** (PRO 6000 Workstation; L12 is one pod, replicates DEGRADED) | CLAIMS C1 replication note (+0.05–0.09 on clean cells), C3 (L40S, H100, A100), "Not transferable" | Accept; say "one card family" in the issue's first table caption (E6) |
| 7 | **cuTENSOR vs cupy** | Campaign: cuTENSOR 2.3.1 gated on every pod (speed + accuracy). g4psrc r1–r3: contraction engine **cupy** (VALIDATION-g4psrc.md), so correctness holds on both engines. Upstream CI installs cutensor 2.2.0 | Speed without cuTENSOR unmeasured; a one-line disclosure (E4) is enough |
| 8 | **Convergence contract inside `_kernel`** | PR_BODY "Design points"; question 4 | Alternative with in-tree precedent: G3 |
| 9 | **The record dict**: 42 keys written, two `_keys` entries | question 6 | Offer a trimmed default (G6), only if asked |
| 10 | **API fit**: policy object vs `fp_type`-style attribute | Precedents paragraph is missing from the issue (E2); `_keys` usage matches `df_jk.py` / `dfmp2.py` | Offer a `__config__` default (`gpu_dft_mixed_precision`, default `None`) like `dfmp2_addons.CONFIG_FP_TYPE` (E2) |
| 11 | **No example** (CONTRIBUTING recommends one) | — | G5 |
| 12 | **Grid pruning, basis, `conv_tol_grad` recommendations** (CLAIMS C4: set `conv_tol_grad ≈ 3e-6` for gradients, stock and mixed alike) | CLAIMS C4; issue "Accuracy" | Fine as is |

## 2. Gap list with closures

Code changes alter the validated SHA; every code change below is bundled into **one** re-validation
(g4psrc r4, one pod) immediately before posting, which the CHECKLIST §1 rebase step requires anyway
if master moves. G2 scope (only `.py` under `gpu4pyscf/` plus README) still holds for all of them;
an example under `examples/` is outside the gate's scope.

| id | gap | closure | files / functions | re-validate? | when |
|---|---|---|---|---|---|
| G1 | tests error on multi-GPU runners | Code: `from gpu4pyscf.lib.multi_gpu import num_devices`; `@unittest.skipIf(num_devices > 1, '')` on every test that calls `kernel()` or `mp.begin()` with a policy (the upstream pattern), **and** one new test asserting the refusal message on `num_devices > 1` by patching `mixed_precision.check_supported`'s import, so the refusal is still tested | `gpu4pyscf/dft/tests/test_mixed_precision.py`, `test_mixed_precision_vv10.py` | yes (r4) | **before issue** (the issue links the branch; a reader may run it on two GPUs) |
| G2 | copyright year on the four new files (`2021-2026` vs upstream's `2026` for new 2026 files) | Text on the code branch; CHECKLIST §3 | the two new modules, two new test files | yes (r4, trivially) | before PR |
| G3 | `_kernel` intrusion | Code, **only after the maintainers answer Q4**: add PySCF/soscf parity to `hf._kernel` (`if callable(mf.check_convergence): scf_conv = mf.check_convergence(locals())` at the convergence test, 3 generic lines), have `mixed_precision.begin` install `mf.check_convergence` (force the switch, return False; raise if `call == 0`) and `end` restore it. The `scf()` begin/end wrapper stays | `gpu4pyscf/scf/hf.py::_kernel`, `dft/mixed_precision.py::begin/end/_SCFState` | yes (full g4psrc) | **before PR, if Q4 answered "not in `_kernel`"**; otherwise offer it in the issue (E3) |
| G4 | no device guard | Code, only if Q2 answered "warn": `logger.warn` in `check_supported` when `cupy.cuda.runtime.getDeviceProperties(0)` has `(major, minor)` in `{(7,0),(8,0),(9,0)}` (the measured FP64-strong class plus V100; B200 = 10.0 unmeasured, so not listed), citing C3's numbers; never refuse. Test with a patched property dict | `dft/mixed_precision.py::check_supported`, tests | yes | only if asked |
| G5 | no example | Doc: `examples/dft/NN-mixed_precision.py` (next free number; `11-dft_with_nlc.py` is the closest neighbour), 30 lines: B3LYP `MixedPrecision(xc=True, k=True)`, wB97M-V `MixedPrecision(vv10=True)`, print `mixed_precision_record['xc']`, a comment on FP64-strong cards | new file | no (outside `gpu4pyscf/`) | before PR |
| G6 | 42-key record | Code, only if Q6 asks: keep `xc`, `k`, `vv10`, `*_switch_call`, `*_switch_reason`, `ao_cache_tier`, `fp64_tail`, `vv10_cert` on the object; move the rest to `log.debug` | `dft/mixed_precision.py::_SCFState.finish` | yes | only if asked |
| G7 | per-PR gate profiles in the validation harness | Harness (engine repo, no GPU): `g4psrc` currently requires both test files (G6) and nine trio cells with cache and VV10 gates (G8); a split PR needs a profile per stack member (§3). Not a code-branch change | `.github/scripts/runpod_g4psrc.py`, `g4psrc_pod.py` in the engine repo | n/a | before PR |
| M1 | no speed at library defaults; no no-cache PRO 6000 figure | **Measurement, 1 pod** (estimate: under the rfcbench `WORK_CAP` of 4440 s from the C0 calibration, where the trio's largest cell, celecoxib wB97M-V stock, is 162 s per run): trio × {r2SCAN, B3LYP, wB97M-V}, R = 3, arms `stock-default` (pruned grid), `mixed-default` (`MixedPrecision(xc=True, k=True)` / `(vv10=True)` exactly as the README, pruned grid, no cache), and `mixed-campaign` (current arm) for a bridge. Needs a harness amendment first: `rfcbench_cells.py` hard-codes `grids.prune = None` (line 1054), gates on it (line 143) and fixes the policies (lines 62–66); add a `defaults` mode under a dated PREREG amendment, on the campaign fork tree the harness admits (`VALIDATED_FORK_SHAS`). **Falsification written first:** if r2SCAN `mixed-default/stock-default` < 1.15 (the PAYS threshold), the issue must say the gain needs unpruned grids and the cache, and Q3 should lead with the cache | engine repo harness; no code-branch change | no | **before issue** (it is the first question a maintainer asks, and one pod answers it and gap 3's second half together) |
| M2 | V100/T4 runtime and sm_70 NVRTC compile | Measurement, **0 pods first**: `rfcbench mode=gputypes` (launches no pod) to see whether RunPod lists a V100 or T4. If yes, 1 pod running only the 49 tests + `bind_kernels()` on it (the g4psrc pod minus the source build, or the wheel + overlay path from g4pport, since the 1.8.1 wheel has sm_70 code). If no such card, close by text: state the 129.0 s PRO 6000 figure, the intrinsics list, and that sm_70 is untested | engine repo | no | before PR (or only if asked, when no card exists) |

Total GPU spend proposed: **2 pods** (M1, M2) plus **1 g4psrc r4** for G1/G2 before the issue, and
one g4psrc per later rebase or per stack member. Nothing else.

## 3. PR split strategy

Measured facts (throwaway worktree on `c1a6e37`, removed afterwards; re-tested on the current SHAs
in a throwaway clone after the tests-only change, same outcome on every row):

| commit | content | numstat | files | tests | applies standalone on master? |
|---|---|---|---|---|---|
| `920bf64` | XC + K + convergence contract | +928 / −17 | `df/df_jk.py`, `dft/mixed_precision.py`, `dft/rks.py`, `dft/tests/test_mixed_precision.py`, `scf/hf.py` | 22 (21 `KnownValues` + `MultiGPU`) | **yes** (`cherry-pick` clean) |
| `39300f2` | VV10 component | +1549 / −75 | adds `dft/vv10_mixed.py`, `dft/tests/test_mixed_precision_vv10.py`; edits `numint.py` (+29/−20), `mixed_precision.py`, `df_jk.py`, `rks.py`, `hf.py` (+11/−2) | 18 | on `920bf64`: **yes** |
| `9755115` | FP64 AO cache | +541 / −39 | `mixed_precision.py` (+362/−39), `df_jk.py` (+5), `rks.py` (+5), `test_mixed_precision.py` (+169) | 10 (`AOCache`) | on `920bf64` **without** `39300f2`: **no** — conflicts, all in `mixed_precision.py` (7 hunks / 104 lines when measured on the pre-r4 SHA; not re-counted); on `920bf64`+`39300f2`: yes |
| `4d2f3de` | docstring + README | +9 / −4 | `mixed_precision.py`, `README.md` | — | on the three above: yes; the replayed stack's tree equals `4d2f3de4^{tree}` |

So the stack **a → c → b** (XC+K, VV10, cache) replays cleanly; **a → b** needs the cache commit
re-authored on `920bf64` (estimate: an hour of merge work in one file; the tests do not conflict).

| option | PRs | g4psrc gate set per PR | for | against |
|---|---|---|---|---|
| **One PR** | the four commits, +2931/−39, nine files, 50 tests | r4's gates as they are (one pod per rebase) | one review, one re-validation loop, evidence maps onto the PR as written | largest external PR in the sample by a wide margin; review stalls on the NVRTC module block the XC+K core; CI red from G1 hits all of it at once |
| **Stack of three** | (a) `920bf64` (+ the docstring/README parts of `4d2f3de` that concern it); (b) `9755115` re-authored on (a); (c) `39300f2` | (a): G1–G5 as now, G6 = `test_mixed_precision.py::KnownValues` 21/21 + `MultiGPU` 1/1, G8 = trio × {r2SCAN, B3LYP} 6 cells, `|ΔE| ≤ 1e-8`, cycles ±1, FP64 tail, real FP32, no cache or VV10 gates. (b): + `AOCache` 10/10, cache-tier gates on the 6 cells. (c): + VV10 18/18, G7 identity, trio wB97M-V 3 cells with the certificate | each PR is reviewable in one sitting; (a) contains the only two intrusions a maintainer must think hard about (`_kernel`, `get_jk`); the two follow-ups wait for (a) to merge, so their rebases are cheap | three re-validation loops (one pod each, serialised); (a) alone has only PREREG-9's no-cache and library-default trio figures as PRO 6000 speed evidence (the 24-molecule headline 1.657/1.783 needs the cache); (b) must be re-authored, so its validated SHA changes |
| Stack in commit order (a → c → b) | same commits, no re-authoring | as above with (b) and (c) swapped | zero conflicts; (c) is the component with the largest gain (2.914) | puts the NVRTC module second and the small, bitwise-tested cache last, the reverse of what Q3 in the issue proposes |

**Recommendation: the stack of three, (a) first, with (b) and (c) ordered by the maintainers'
answer to Q3** (default a → c → b because it replays cleanly; re-author (b) only if they want the
cache before VV10). Open (a) only; keep (b) and (c) as validated branches on the fork
(`mixed-precision-upstream-vv10`, `mixed-precision-upstream-aocache`, each rebased on the merged
(a)) and link them from the PR. Reason: the sample shows no external PR above ~30 lines landing,
so the first PR should be the smallest piece that is useful on its own, and (a) is that piece.
If the maintainers answer Q3 with "send it all", fall back to the one PR: the texts are already
written for it.

## 4. Sequenced timeline

**Phase 0, pre-issue (no posting).**
1. Engine repo: harness amendment for M1 (`defaults` mode), dated PREREG amendment with the
   falsification criterion written before dispatch. Dispatch M1 (1 pod, serialised).
2. `rfcbench mode=gputypes` (0 pods). If a V100/T4 is listed, dispatch M2 (1 pod) after M1 finishes.
3. Code branch: G1 (skips) and G2 (years). `git fetch upstream`; if `c1a6e37..upstream/master`
   is non-empty, rebase. Lint as CHECKLIST §2. Push as `mixed-precision-upstream` (the issue links
   the branch name, so the link survives).
4. g4psrc r4 on the new head (1 pod; `BASE_SHA` updated in `runpod_g4psrc.py` with a dated port-plan
   section first). Update CHECKLIST, `VALIDATION-g4psrc.md`, `g4psrc_r4.json`.
5. Apply the §6 edits to `RFC_ISSUE.md` / `PR_BODY.md` after the owner's review; add the new numbers
   (M1 results, r4 run id, any line numbers quoted) to `verify_rfc.py`'s `BOUND`/`ALLOWLIST` with
   their sources; run the three verifiers in CHECKLIST §5; run the forbidden-string greps.
6. Post the issue (CHECKLIST §6). Pin nothing else.

**Phase 1, the six questions. Likely answers and the follow-up for each:**

| Q | likely answer | follow-up |
|---|---|---|
| 1 API | "policy object is fine" | nothing. "Prefer a flag like `fp_type`" | add `mixed_precision = getattr(__config__, 'gpu_dft_mixed_precision', None)` as the class default and accept `mf.mixed_precision = 'xc,k'` strings parsed into a policy; keep the object. Code, (a) only, re-validate |
| 2 guard | "document only" | nothing. "warn" | G4. "refuse on sm_80/sm_90" | implement as refusal with the same list; say in the PR that L40S (8.9) must stay allowed (C3: 1.729 / 1.524 / 2.994) |
| 3 scope | "XC+K first" | open (a). "cache before VV10" | re-author (b) on (a), validate. "everything" | one PR |
| 4 contract | "`_kernel` is fine" | nothing. "not in `_kernel`" | G3; mention that soscf already honours `check_convergence`. "in the RKS classes" | move the check into `rks.get_veff`/`df_jk.get_veff` via `end_call` raising a flag that `scf()` reads; harder, propose G3 first |
| 5 tolerances | "fine" | nothing. "1e-3 too loose / 1e-8 too tight" | thresholds are controller settings (CLAIMS); tolerance change = one constant + test; re-validate |
| 6 record | "fine" | nothing. "logger only" | G6 |

If there is no answer after two weeks (estimate of a reasonable wait, given the one-day merges of
small external PRs), post one follow-up comment with the M1 number; after four weeks, open (a) as a
draft PR referencing the issue, since the texts assume an answered issue.

**Phase 2, PR (a).** Rebase onto current master, g4psrc (a)-profile, lint, open from
`mixed-precision-upstream` with `PR_BODY.md` trimmed to (a) (reviewer's guide rows 1–4 only;
remove the VV10 section; `verify_rfc.py` `BOUND` loses the VV10 and cache entries, deliberately).
Watch for: lint, the V100 job's `--durations=50` (if the new tests appear in the top 50, offer to
move the paracetamol cells to default pruned grids), and the 2T4 job (should show skips).

**Phase 3, keeping current.** The re-validation loop has run three times (r1 `eef5f5b`,
r2 `82bc702`, r3 `c1a6e37`, all PASS, no conflicts; the `eval_ao` API change in `a2f3c36` did not
touch the branch). Rule: rebase and re-run g4psrc only when (i) a maintainer asks, (ii) a merge
conflict appears, or (iii) a reviewed-and-approved state needs a final green; not on every master
commit (25 in 30 days). One pod per rebase, serialised with everything else.

**Phase 4, (c) and (b)** after (a) merges: rebase each onto master, validate with its profile, open.

## 5. Risks and stop conditions

| risk | signal | action |
|---|---|---|
| Maintainers decline the feature | explicit "no" or "not in core" | **Fallback A:** ask for two small seams only — `check_convergence` parity in `hf._kernel` (3 lines, soscf already has it) and the `vv10_kernel`/`uwe_kernel` keywords in `numint` (+29/−20, default `None`, stock path unchanged). With those, the rest ships as a package on the fork (`gpu4pyscf_mixed_precision`) that subclasses RKS (`get_veff`, `get_jk` via `with_df`) and installs `check_convergence`. **Fallback B:** keep the fork branch, rebased at the three triggers above |
| Plugin feasibility without any upstream change | — | Hooks used: `scf()` begin/end (wrap `mf.kernel`: feasible), `rks.get_veff` / `df_jk._DFHF.get_veff` (subclass override: feasible, but `nr_rks_fp32` and `nr_rks_fp64_cached` call `numint` internals and `_sparse_index`-style private helpers, so they break on upstream refactors), `df_jk.get_jk` FP32 path (override `mf.get_jk` with a copy of `_k_block_fp32`: feasible), **`_kernel` convergence veto: not feasible today**, because `_kernel` ignores `check_convergence`; the only plugin route is replacing `mf.kernel` with a copy of `_kernel`, and `_vv10nlc` needs a monkeypatch. So a plugin is feasible for XC+K+cache only with a private copy of the SCF loop; Fallback A removes exactly those two obstacles |
| M1 shows no gain at defaults | r2SCAN `mixed-default/stock-default` < 1.15 | Reframe the issue: the mode is for users who already run unpruned grids (and say why: XC share), lead Q3 with the cache; do not post the headline without that caveat |
| CI red on the 2T4 runner | the first PR run | G1 before the issue removes it; if any test still errors there, convert the module to skip at `setUpModule` |
| V100 runtime complaint | `--durations` | cut paracetamol cells to default grids; keep one unpruned cell for the FP32-phase assertions |
| Capacity flake on the validation pod | `STATUS=FAIL` with `ModuleNotFoundError`/`NVML`/empty `out.txt`, or no sentinel | re-dispatch once, unchanged (the engine repository's dispatch rules); never alongside another GPU run |
| The issue's branch link goes stale | branch re-pushed for r4 | links name the branch, not the SHA; keep `mixed-precision-upstream-on-c1a6e37` as a frozen alias of r3 like the r1/r2 branches |
| A review asks for UKS, RSH or multi-GPU | PR comment | out of scope by refusal; say so once, with the refusal test as the evidence; do not extend the PR |

## 6. Edits recommended to RFC_ISSUE.md and PR_BODY.md (not applied)

Every new number below goes into `verify_rfc.py`'s `ALLOWLIST` with its source (line numbers
quoted from upstream files are numbers too: prefer naming the function instead).

**RFC_ISSUE.md**
- E1 (**before the first table**, after the API block): add a "Fit with the test suite" paragraph:
  "The 49 tests are unmarked, run in 129 s on one RTX PRO 6000 (g4psrc r3), skip on multi-GPU hosts
  (`num_devices > 1`, the refusal is tested separately), and need no compiled code: the VV10 kernels
  are NVRTC `RawModule`s with `--std=c++14` and no `__CUDA_ARCH__` branching; they have not been
  compiled for sm_70." Write it after G1; if M2 runs, replace the last clause with its result.
- E2 (**under "Proposed API"**, replacing the one-line precedent sentence): "Precedents in tree:
  `mp/dfmp2.py` `fp_type` (a `__config__`-backed attribute in `_keys`), `tdscf/ris.py` `single`
  (warns below `conv_tol` 1e-5), the `ENABLE_FP32_MULTIGRID` build option and the cuEST wrapper's
  precision parameters. We can give the policy a `__config__` default
  (`gpu_dft_mixed_precision`, default `None`) in the same way." Then Q1 becomes concrete.
- E3 (**Q4**): add the alternative: "`SCF.check_convergence` exists and `soscf` honours it;
  `_kernel` does not. We can instead add that parity call to `_kernel` and have the policy install
  `mf.check_convergence`, which removes the mixed-precision branch from `_kernel` entirely."
- E4 (**"What is not claimed", cuTENSOR bullet**): add "correctness was validated on both engines:
  cuTENSOR 2.3.1 in the campaign, cupy in the source-build validation."
- E5 (**Q3**): state the split facts: "XC+K (`+915/−17`, five files, 21 tests) applies on master
  alone; VV10 applies on it; the AO cache applies cleanly only after VV10 (one file conflicts) and
  would be re-authored if you want it second." After M1: give the no-cache and library-default
  numbers here, since (a) alone is what they would get first.
- E6 (**first speed table caption**): "RTX PRO 6000 Blackwell Workstation (one card family; one
  pod for r2SCAN/B3LYP)".
- E7 (**length**): move the "MIG"/"MPS" and "Cold start" bullets out of the issue into a one-line
  pointer to CLAIMS ("deployment notes and cold-start analysis: CLAIMS.md"); maintainers do not
  need deployment advice to answer the six questions. Optional.
- E8 (**Problem**, first paragraph): add the size up front: "The code is +2915/−39 over nine files;
  the first PR we would propose is +915/−17 over five."

**PR_BODY.md**
- P1: add a "CI fit" subsection under "Test plan" with E1's content and the 2T4 skip behaviour.
- P2: "Reviewer's guide" row 2 (`scf/hf.py`): if G3 is adopted, rewrite to "adds the
  `check_convergence` call PySCF's `kernel` and `soscf` already make; the policy installs it".
- P3: if G5 is adopted, "nine files" becomes "ten files" and a row 9 for the example is added
  (the `ALLOWLIST` entry for `9` must change with it, by design).
- P4: when the PR is (a) only, drop rows 5–7, the VV10 section, and every VV10/cache number;
  `verify_rfc.py` will then fail on stale `BOUND` entries until they are removed, which is the
  intended check.
- P5: under "Limitations", after the FP64-strong bullet, add the M1 result at library defaults,
  whatever it is, with the same PAYS/NEUTRAL/HARMS reading CLAIMS uses.

## Addendum (2026-10-10, owner review): dependency pins in upstream CI

Upstream's `unittest.yml` installs different versions from the ones every validation here used.

| CI job | pyscf | gpu4pyscf-libxc | validated here |
|---|---|---|---|
| `single-gpu` (V100) | from `requirements.txt` | **0.9.0** | libxc 0.8.1 |
| `multi-gpu` (2×T4) | **2.8** (`pip3 install pyscf==2.8`) | 0.8.1 | pyscf 2.14.0 |

**G8, a gap.** The branch has never run against pyscf 2.8 or libxc 0.9.0. Nothing here shows that
`mixed_precision.py`, `vv10_mixed.py` or the tests use a pyscf API newer than 2.8. The libxc 0.9.0
change could also shift XC values the |ΔE| ≤ 1e-8 Ha tests compare against stock. Both arms use the
same libxc, so this is less likely to matter, but it is unmeasured.

Closure, before the PR:
- **On CPU.** List every `pyscf.*` symbol the four new files import or call, and check each exists
  in pyscf 2.8. `pip download pyscf==2.8` and grep; no GPU is needed.
- **On the GPU.** Add a pin option to the g4psrc pod (pyscf, gpu4pyscf-libxc) and run r4 twice:
  - once at the lock's pins, as r1–r3;
  - once at the CI pins: libxc 0.9.0 for the V100 job, and pyscf 2.8 for the 2T4 job.
  That is 1 extra pod. Under 2T4's pins, the multi-GPU skip (G1) is what runs, so pyscf 2.8 only
  needs the single-GPU import and test path.
- **If the pins differ** by the time of the PR, use whatever `unittest.yml` says then.

## Status addendum (2026-10-10, after PREREG-9 and g4psrc r4)

Against code branch head `4d2f3de4` (commits `920bf64`, `39300f2`, `9755115`, `4d2f3de` on
`c1a6e371`; the r3 head `ac4c668e` is kept as `mixed-precision-upstream-on-c1a6e37`) and the package
branch with PREREG-9 banked (`native/CLAIMS.md` C7, `verify_native.py` N11, `native/data/ld1.csv`
and `ld2.csv`).

| item | status |
|---|---|
| G1 multi-GPU skips | **done**, tests only: `@unittest.skipIf(num_devices > 1, ...)` on every test class that runs an SCF with a policy, plus `MultiGPU::test_refuses_multi_gpu` with a patched device count (50 tests now). Validated by r4. Lint re-run clean at `4d2f3de4` (CHECKLIST §2) |
| G2 copyright years | **not done**: the four new files still carry `2021-2026` |
| G3 `check_convergence` parity | not done, by design: offered in the issue (E3, Q4) |
| G4 device guard | not done; only if asked |
| G5 example | not done |
| G6 record trim | not done; only if asked |
| G7 per-PR gate profiles | **not done.** r4 added *card/package* profiles to g4psrc (`pro6000-lock`, `v100-ci`, `pro6000-pyscf28`), which is a different axis from per-stack-member gate sets |
| M1 speed at library defaults | **done: PREREG-9**, as its own protocol with a new `defaults` mode (two pods, LD1 and LD2, five arms per cell rather than the three sketched above). r2SCAN **1.577**, B3LYP **1.636**, wB97M-V **3.094** at the defaults; without the cache on the unpruned grid 1.627 / 1.954 / 2.884. The falsifier (r2SCAN < 1.15) did not fire; B3LYP no-cache missed its band above (MODEL-MISS); the same-pod campaign anchor read +0.108 (B3LYP) and +0.141 (wB97M-V) over the banked trio, both faster (REPLICATION-FLAG) |
| M2 V100 / sm_70 | **done: r4 `v100-ci`** (run `38024957471`, the infrastructure re-run of `38023715438`, which found no V100 capacity): built from source for sm_70 with upstream's single-GPU CI package set (libxc 0.9.0); 50/50 in 133 s, slowest 18.3 s; VV10 bitwise on sm_70; G1–G7, no trio. RunPod lists no T4 |
| G8 pins | **partly done**: pyscf 2.8 (r4 `pro6000-pyscf28`, run `38023769024`, G1–G8, trio \|ΔE\| ≤ 2.7e-12 Ha) and libxc 0.9.0 (r4 `v100-ci`, G1–G7). **scipy 1.17 not tested** (no Python 3.10 wheel; r4 Amendment 1 kept 1.15.3). The CPU-side symbol check was superseded by the GPU run. Found in passing: upstream's own `test_rks::test_nr_coach` fails under libxc 0.9.0 on master and branch alike |
| g4psrc r4 | **done**, three profiles, all PASS (`pro6000-lock` run `38022477235`: 44 + 1 skipped on both trees, 50/50, \|ΔE\| ≤ 5.9e-12 Ha); `validation/VALIDATION-g4psrc.md` and `g4psrc_r4_*.json` |
| §3 split facts | re-tested on the new SHAs: (a) clean on master; (b) conflicts on (a) alone, in `mixed_precision.py` only; a → c → b → d replays to exactly `4d2f3de4^{tree}` |
| §4 Phase 0 | steps 1–5 done; step 6 (post the issue) not done |
| §5 risk "M1 shows no gain" | did not materialise (1.577 ≥ 1.15); the issue leads its speed section with the defaults beside the campaign configuration |

**§6 edits.** RFC_ISSUE: E1 applied (with M2's measured result in place of the "not compiled for
sm_70" clause), E2, E3, E4, E5 (with M1's numbers), E6, E7 (MIG/MPS and cold start reduced to a
pointer; their `verify_rfc.py` bindings were removed for the issue and kept for the PR body) and E8
(with the current numstat) applied. PR_BODY: P1 and P5 applied; P2 not applied (G3 not adopted,
pending Q4); P3 not applicable (G5 not done); P4 not applicable yet (the body still describes the
full branch). Every new number is bound to `verify_native.py` (C7) or allow-listed with its source
in `verify_rfc.py`, which passes with no `UNVERIFIED` line.
