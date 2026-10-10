# Staging checklist for submitting the mixed-precision RFC to pyscf/gpu4pyscf

For the person who will post. Nothing in this directory has been posted. The order matters:
**the issue first, the PR only after the maintainers have answered.**

State of play (updated 2026-10-10):

| item | value |
|---|---|
| code branch | `chris-lee-mc/gpu4pyscf@mixed-precision-upstream`, head `4d2f3de4c244a6f11debbf44e17d61900c845fc6`, four commits (`920bf64`, `39300f2`, `9755115`, `4d2f3de`) on upstream master `c1a6e371d1a46afc9932c618108a4bc70e895edf` |
| evidence package | `chris-lee-mc/gpu4pyscf@mixed-precision-rfc-package-v2`, `benchmarks/mixed_precision/` |
| last GPU validation of the code branch | g4psrc r4, engine-repo runs `38022477235` (`pro6000-lock`), `38023769024` (`pro6000-pyscf28`) and `38024957471` (`v100-ci`), on master `c1a6e37`, from source for sm_120 and sm_70: PASS (`validation/VALIDATION-g4psrc.md`) |
| earlier heads, kept as branches | `mixed-precision-upstream-on-eef5f5b` (r1), `mixed-precision-upstream-on-82bc702` (r2), `mixed-precision-upstream-on-c1a6e37` (r3, `ac4c668e`) |

## 1. Base: re-verify, rebase if needed, re-validate

- [x] Rebased onto `c1a6e37` (no conflicts; the branch adds no `eval_ao` call, so `a2f3c36`'s API
      change does not touch it) and re-validated as g4psrc r3: regression 44 passed + 1 skipped on
      both trees, 49/49 new tests, VV10 bitwise identity, trio \|ΔE\| ≤ 6.4e-12 Ha with
      identical cycles.
- [x] Tests-only change for upstream's CI (multi-GPU skips, `MultiGPU` refusal test; head
      `4d2f3de4`, same base) re-validated as g4psrc r4 on three profiles: `pro6000-lock` (as r3:
      44 passed + 1 skipped on both trees, 50/50, trio \|ΔE\| ≤ 5.9e-12 Ha), `pro6000-pyscf28`
      (pyscf 2.8: the same gates, \|ΔE\| ≤ 2.7e-12 Ha) and `v100-ci` (one V100, sm_70, libxc
      0.9.0: regression, 50/50 in 133 s, VV10 bitwise identity; no trio). `PR_BODY.md`,
      `RFC_ISSUE.md` and `verify_rfc.py` cite r4.
- [x] `validation/VALIDATION-g4psrc.md` with `g4psrc_r2.json`, `g4psrc_r3.json` and the three
      `g4psrc_r4_*.json` records added; `VALIDATION-vv10.md`'s "not validated" line points to it.
- [ ] Immediately before posting: `git fetch upstream && git log --oneline c1a6e37..upstream/master`.
      If the list is non-empty, rebase again, re-run g4psrc (the engine repo pins `BASE_SHA` in
      `.github/scripts/runpod_g4psrc.py`, with a dated section in the port plan before dispatch),
      update the run id, base and numbers in `PR_BODY.md` and `RFC_ISSUE.md`, add them to
      `verify_rfc.py`'s `ALLOWLIST`, and re-run it.

## 2. Lint, exactly as upstream's `.github/workflows/lint.yml` runs it

From the root of the code branch:

```
pip install ruff==0.15 "flake8>=3.7.0"
ruff check --config .ruff.toml --unsafe-fixes gpu4pyscf
ruff check --select NPY --ignore NPY002 gpu4pyscf
flake8 --config .flake8 gpu4pyscf
```

- [x] All three clean on the whole `gpu4pyscf` tree at `ac4c668e` (ruff 0.15, flake8 7.3.0), and
      again at `4d2f3de4` after the tests-only change (ruff 0.15.0, flake8 7.3.0, from a
      `git archive` of that commit).
- [ ] Re-run if the branch changes again.

## 3. Licence headers and CLA

- [ ] Upstream has no CLA file and `CONTRIBUTING.md` mentions none. Watch for a CLA bot on the PR.
- [ ] Every file upstream carries the Apache-2.0 header `# Copyright <years> The PySCF Developers.
      All Rights Reserved.` The four new files on the branch carry it with `2021-2026` (still so at `4d2f3de4`). Files added
      to upstream in 2026 (for example `gpu4pyscf/df/tests/test_df_ao2mo.py`,
      `gpu4pyscf/pbc/dft/gks.py`) use `# Copyright 2026 The PySCF Developers`. **Mismatch to
      decide:** either leave `2021-2026` (the existing edited files use `2021-2024` or
      `2021-2026`) or change the four new files to `2026`. This checklist does not edit the code
      branch; make the change there if you want it, then re-run lint and g4psrc.
- [ ] `CONTRIBUTING.md` asks for sm_70 compatibility and for tests against CPU PySCF values. The
      VV10 kernels are NVRTC-compiled C++14 (`cupy.RawModule`, options `--std=c++14`, no fast
      math); the only device intrinsics in `vv10_mixed.py` are `__fadd_rn`, `__fsub_rn`,
      `__fmul_rn`, `__syncthreads` and `__longlong_as_double`, and there is no `__CUDA_ARCH__`
      branching, so nothing requires sm_80 or newer. They were run only on sm_120 (and the L40S,
      H100 and A100 pods of the campaign). `test_against_cpu_pyscf` covers the second point for
      `xc`/`k`. State both in the PR if asked.

## 4. Consistency of the staged texts (found while preparing them; fix before posting)

- [x] `PR_DESCRIPTION.md` is marked as superseded by `rfc/PR_BODY.md` (it describes the v1.8.1
      branch: "21 tests", "+914 / −18"); its dead `DISCUSSION-DRAFT` reference now points to
      `rfc/RFC_ISSUE.md`. Delete it from the package branch before posting if you prefer.
- [x] `README.md` now says 46 pods (44 plus PREREG-9's LD1 and LD2), matching `native/README.md`
      and `native/SOURCES.md`.
- [x] The `mixed_precision.py` docstring (amended into the fourth commit, now `4d2f3de4`) now gives
      CLAIMS C3's figures (wB97M-V 1.6–2× slower; H100 B3LYP neutral to about 1.1× slower; A100
      11 of 12 clean cells neutral or slower), and says K is held to XC's switch only when `xc` and
      `k` are both on.
- [ ] The usage snippet in that docstring uses `gpu4pyscf.dft.RKS` (a factory function in
      `gpu4pyscf/dft/__init__.py` on `c1a6e37`); the tests and `rfc/` use `gpu4pyscf.dft.rks.RKS`.
      Both resolve; no action unless the rebase changes `dft/__init__.py`.

## 5. Final verification before posting anything

- [ ] `python3 benchmarks/mixed_precision/native/verify_native.py` ends with
      `939/947 checks reproduced; 8 known discrepancies (listed above, not adjusted); 0 failures`
      (or a higher reproduced count if checks were added) and exits 0.
- [ ] `python3 benchmarks/mixed_precision/verify_aggregates.py` ends with
      `419 checks: 418 ok, 1 known DISCREPANCY (reported, not adjusted), 0 FAIL`.
- [ ] `python3 benchmarks/mixed_precision/rfc/verify_rfc.py` exits 0 with no `UNVERIFIED` line.
- [ ] On the package branch, this returns nothing:

      ```
      git grep -n -i -E 'claude|opus|gpt|anthropic|runpod_api|api_key' -- benchmarks/mixed_precision \
        ':!benchmarks/mixed_precision/native/provenance/harness' \
        ':!benchmarks/mixed_precision/rfc/CHECKLIST.md' \
        ':!benchmarks/mixed_precision/rfc/verify_rfc.py'
      ```

      The last two exemptions are the files that hold the pattern itself: this checklist, and
      `verify_rfc.py`'s `FORBIDDEN` list.
- [ ] On the code branch, grep only what the branch adds. Upstream's own sources already contain
      matches (for example, ten "Generated by ChatGPT" comments in `lib/ecp/cart2sph.cu` and
      `lib/gdft/nr_eval_gto.cu` on master). This returns nothing:

      ```
      git diff upstream/master mixed-precision-upstream | grep -i -E '^\+.*(claude|opus|gpt|anthropic|runpod_api|api_key)'
      ```
- [ ] **The harness copy is exempt on purpose.** `native/provenance/harness/` is a byte-for-byte
      snapshot of the benchmark harness at the commit that produced the data, so it is not edited.
      It holds 7 matches in 4 files, and none is a credential:
      - `rfcbench.yml` 1: the Actions secret's name, `${{ secrets.RUNPOD_API_KEY }}`;
      - `runpod_rfcbench.py` 4: the docstring naming the required variable, the line that reads it
        from the environment (`os.environ["RUNPOD_API_KEY"]`), and two comments citing the harness
        repository's `CLAUDE.md`;
      - `rfcbench_sets.py` 1 and `scfbench_requirements.lock` 1: comments citing that
        repository's `.claude/skills/` notes and `CLAUDE.md`.
      Check that nothing else has appeared. This should total 7, in those four files:
      `git grep -c -i -E 'claude|opus|gpt|anthropic|runpod_api|api_key' -- benchmarks/mixed_precision/native/provenance/harness`
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
- [ ] If the maintainers asked for XC + K only, drop the VV10 commit (`39300f2`) and the AO-cache
      commit (`9755115`) from the PR branch, remove the VV10 section and the AO-cache rows from
      `PR_BODY.md`, and re-run `verify_rfc.py` (its `BOUND` table will then need the VV10 and
      cache entries removed, which is deliberate: every edit to a number goes through the checker).
