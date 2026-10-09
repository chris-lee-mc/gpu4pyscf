# Staging checklist for submitting the mixed-precision RFC to pyscf/gpu4pyscf

For the person who will post. Nothing in this directory has been posted. The order matters:
**the issue first, the PR only after the maintainers have answered.**

State of play when this checklist was written (2026-10-09):

| item | value |
|---|---|
| code branch | `chris-lee-mc/gpu4pyscf@mixed-precision-upstream`, head `122e2d5ee2bb20a7d6fa611e07a3628fc4242b01`, four commits on upstream master `82bc70289268193aacfc225aabce6b7221d01632` |
| evidence package | `chris-lee-mc/gpu4pyscf@mixed-precision-rfc-package-v2`, `benchmarks/mixed_precision/` |
| last GPU validation of the code branch | g4psrc r2, engine-repo run `37881608304`, on master `82bc702`, from source for sm_120 |
| upstream master at the time of writing | `c1a6e37` (2026-10-10): **two commits past the base**, one of which (`a2f3c36`, "Change eval_ao API. Reject argument shls_slice (#955)") edits `gpu4pyscf/dft/numint.py`, a file this branch also edits |

## 1. Base: re-verify, rebase if needed, re-validate

- [ ] `git fetch upstream && git log --oneline 82bc702..upstream/master`. If the list is non-empty,
      the base is stale (it already is: see the table).
- [ ] Rebase `mixed-precision-upstream` onto `upstream/master`. Expect to resolve `dft/numint.py`
      by hand: the branch's hunks are the `uwe_kernel=` / `vv10_kernel=` keywords in `_vv10nlc`
      and `nr_nlc_vxc`; `a2f3c36` changed `eval_ao`. `mixed_precision.py` and `vv10_mixed.py` do
      not call `eval_ao` directly (they use `ni.block_loop`, `numint.eval_rho`, `_eval_rho2`,
      `_scale_ao`, `_tau_dot`), so check that those private helpers still exist with the same
      signatures on the new master.
- [ ] Re-run the g4psrc validation on the rebased head, from source for sm_120, and require the
      same gates as r2: upstream `test_df_rks`, `test_df_jk`, `test_rks`, `test_scf` identical on
      master and branch (r2: 44 passed, 1 skipped); new tests 49/49; VV10 kernels bitwise equal to
      their emulations; trio \|ΔE\| ≤ 1e-8 Ha with cycles ±1 (r2: ≤ 6.8e-12 Ha, identical cycles).
- [ ] Update the base SHA, the run id and the numbers in `PR_BODY.md` (Test plan) and in
      `RFC_ISSUE.md` ("Correctness of the master-based branch"), then re-run
      `python3 benchmarks/mixed_precision/rfc/verify_rfc.py`: the new run id and any changed count
      must be added to its `ALLOWLIST` with the run as the source, or the checker fails.
- [ ] Add a validation record for the run the PR cites. The package's `validation/` directory holds
      only the v1.8.1-overlay runs; `VALIDATION-vv10.md` still ends "Not validated: the
      master-rebased copy", which `37881608304` has since superseded. Write
      `validation/VALIDATION-g4psrc.md` (+ the JSON record) and fix that line.

## 2. Lint, exactly as upstream's `.github/workflows/lint.yml` runs it

From the root of the code branch:

```
pip install ruff==0.15 "flake8>=3.7.0"
ruff check --config .ruff.toml --unsafe-fixes gpu4pyscf
ruff check --select NPY --ignore NPY002 gpu4pyscf
flake8 --config .flake8 gpu4pyscf
```

- [ ] All three clean on the rebased head. On `122e2d5e` all three passed on the eight changed
      Python files (ruff 0.15.8, flake8 7.3.0), checked on a scratch export of the tree.

## 3. Licence headers and CLA

- [ ] Upstream has no CLA file and `CONTRIBUTING.md` mentions none. Watch for a CLA bot on the PR.
- [ ] Every file upstream carries the Apache-2.0 header `# Copyright <years> The PySCF Developers.
      All Rights Reserved.` The four new files on the branch carry it with `2021-2026`. Files added
      to upstream in 2026 (for example `gpu4pyscf/df/tests/test_df_ao2mo.py`,
      `gpu4pyscf/pbc/dft/gks.py`) use `# Copyright 2026 The PySCF Developers`. **Mismatch to
      decide:** either leave `2021-2026` (the existing edited files use `2021-2024` or
      `2021-2026`) or change the four new files to `2026`. This checklist does not edit the code
      branch; make the change there if you want it, then re-run lint.
- [ ] `CONTRIBUTING.md` asks for sm_70 compatibility and for tests against CPU PySCF values. The
      VV10 kernels are NVRTC-compiled C++14 (`cupy.RawModule`, options `--std=c++14`, no fast
      math); the only device intrinsics in `vv10_mixed.py` are `__fadd_rn`, `__fsub_rn`,
      `__fmul_rn`, `__syncthreads` and `__longlong_as_double`, and there is no `__CUDA_ARCH__`
      branching, so nothing requires sm_80 or newer. They were run only on sm_120 (and the L40S,
      H100 and A100 pods of the campaign). `test_against_cpu_pyscf` covers the second point for
      `xc`/`k`. State both in the PR if asked.

## 4. Consistency of the staged texts (found while preparing them; fix before posting)

- [ ] `PR_DESCRIPTION.md` is the older draft and is superseded by `rfc/PR_BODY.md`. Delete
      `PR_DESCRIPTION.md` from the package branch, or keep it clearly marked as superseded. Its
      stale points: "21 tests" for `test_mixed_precision.py` (it has 31 on `122e2d5e`; 21 + 10
      AO-cache tests), "+914 / −18" (the v1.8.1 code diff; the master-based diff is +2913 / −39),
      a reference to `DISCUSSION-DRAFT-precision-option.md`, which is not in the package, and
      prototype speed numbers that `rfc/` does not quote.
- [ ] `benchmarks/mixed_precision/README.md` says "41 pods" in its last section;
      `native/README.md` and `native/SOURCES.md` say 44. Make them agree (SOURCES lists 44 CSVs).
- [ ] `gpu4pyscf/dft/mixed_precision.py`'s module docstring says "about 1.7-2x slower for wB97M-V"
      on the H100/A100; `CLAIMS.md` C3 says 1.6–2× (H100 0.577–0.615, A100 clean cells
      0.488–0.596). Harmless, but align the docstring on the code branch if you touch it.
- [ ] That docstring also says "K is never FP32 on an iteration where XC is FP64". `begin_call`
      enforces that only when `xc=True`; a k-only policy runs FP32 K under FP64 XC, and
      `test_k_only_paracetamol` asserts exactly that. `rfc/` says "with `xc` and `k` both on, K
      returns to FP64 no later than XC". Align the docstring on the code branch.
- [ ] The usage snippet in that docstring uses `gpu4pyscf.dft.RKS` (a factory function in
      `gpu4pyscf/dft/__init__.py` on `82bc702`); the tests and `rfc/` use `gpu4pyscf.dft.rks.RKS`.
      Both resolve; no action unless the rebase changes `dft/__init__.py`.

## 5. Final verification before posting anything

- [ ] `python3 benchmarks/mixed_precision/native/verify_native.py` ends with
      `837/845 checks reproduced; 8 known discrepancies (listed above, not adjusted); 0 failures`
      (or a higher reproduced count if checks were added) and exits 0.
- [ ] `python3 benchmarks/mixed_precision/verify_aggregates.py` ends with
      `419 checks: 418 ok, 1 known DISCREPANCY (reported, not adjusted), 0 FAIL`.
- [ ] `python3 benchmarks/mixed_precision/rfc/verify_rfc.py` exits 0 with no `UNVERIFIED` line.
- [ ] `git grep -n -i -E 'claude|opus|gpt|anthropic|runpod_api|api_key' -- benchmarks/mixed_precision`
      returns nothing on the package branch, and the code branch likewise.
- [ ] The package branch is pushed and the links in `RFC_ISSUE.md` resolve (they point at the
      `mixed-precision-upstream` tree and at `benchmarks/mixed_precision/` on
      `mixed-precision-rfc-package-v2`).

## 6. Post the issue

- [ ] Open a GitHub issue on `pyscf/gpu4pyscf` with the body of `RFC_ISSUE.md` (drop the HTML
      comment at the top). Title: "RFC: opt-in mixed-precision SCF for density-fitted RKS on
      FP64-limited GPUs". Upstream has no issue template.
- [ ] Attach nothing to the issue beyond the two links; the package is the attachment.
- [ ] **Wait for maintainer feedback.** The six questions at the end of the issue decide the
      shape of the PR (policy object vs flag, device guard, whether VV10 and the AO cache go in
      the first PR, where the convergence check lives, the default thresholds, the record).

## 7. Only then: the PR

- [ ] Apply the maintainers' answers to the code branch first (API shape, scope split, guard),
      re-run lint and the GPU validation, and update `PR_BODY.md` and `verify_rfc.py` to match.
- [ ] Open the PR from `mixed-precision-upstream` (rebased) against `pyscf/gpu4pyscf:master` with
      the body of `PR_BODY.md`, `#<issue>` filled in, and the HTML comment removed. Upstream has no
      PR template. The code branch contains only the nine files in the reviewer's guide; the
      evidence package stays on the fork's package branch and is linked, not included.
- [ ] If the maintainers asked for XC + K only, drop the VV10 commit (`a717a11`) and the AO-cache
      commit (`4b38ec1`) from the PR branch, remove the VV10 section and the AO-cache rows from
      `PR_BODY.md`, and re-run `verify_rfc.py` (its `BOUND` table will then need the VV10 and
      cache entries removed, which is deliberate: every edit to a number goes through the checker).
