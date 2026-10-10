# GPU validation of the VV10 component

**Result: PASS**, 2026-10-01, on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

- Branch commit tested: `906043fd2c7a6583ba827e6d7348a18c78ae58de` (`mixed-precision-vv10-v1.8.1`).
- Base: v1.8.1, `5b284c258a4260baef80e3d150b4e7a81a9dbd57`.
- Machine-readable record: `g4pport_vv10_run.json`.

This is a **correctness gate, not a speed measurement.** It uses the same overlay method as
`VALIDATION.md`, with gates fixed before the run.

## Method and results

1. **Base check.** For each modified file, the wheel's copy was byte-identical to v1.8.1. The files
   are `df/df_jk.py`, `dft/numint.py`, `dft/rks.py` and `scf/hf.py`. The added files are absent from
   the wheel.
2. **Kernel identity.** The kernel sources, compile options, variants and tolerances in
   `dft/vv10_mixed.py` equal those of the implementation that was commissioned and measured. On the
   GPU, the FP32 and df64 kernels are **bitwise identical** to their NumPy emulations, on 9 probe
   points and on a 2,000-point shell grid.
3. **Regression.** Upstream `df/tests/test_df_rks.py` (including `test_rks_wb97m_v`),
   `df/tests/test_df_jk.py`, `dft/tests/test_rks.py` (including `test_rks_vv10`) and
   `scf/tests/test_scf.py`: **42/42 on the stock wheel and 42/42 after the overlay.**
4. **New tests.** `dft/tests/test_mixed_precision.py` (21) and `dft/tests/test_mixed_precision_vv10.py`
   (18): **39/39 passed.**
5. **Stock vs mixed energies** on three drug-like molecules, at the settings of `VALIDATION.md`.
   - **r2SCAN and B3LYP** (`xc`/`k`), the six cells as before: |ΔE| ≤ 6.4e-12 Ha, identical cycles.
   - **wB97M-V**, `MixedPrecision(vv10=True)`:

| molecule | \|ΔE\| (Ha) | cycles stock / mixed | FP32 / df64 VV10 calls | stock tail calls | certificate rel / \|ΔE_nlc\| |
|---|---|---|---|---|---|
| paracetamol | 0.0 | 12 / 12 | 7 / 6 | 0 | 6.7e-14 / 2.0e-15 Ha |
| propranolol | 9.1e-13 | 12 / 12 | 7 / 6 | 0 | 1.7e-13 / 1.1e-14 Ha |
| celecoxib | 0.0 | 13 / 13 | 7 / 7 | 0 | 1.8e-13 / 1.2e-14 Ha |

   The gates were: |ΔE| ≤ 1e-8 Ha, cycles ≤ +1, at least one FP32 and one df64 call, the last two
   df64, no stock tail call, and the certificate within 1e-10 relative and 1e-11 Ha.

The pod also recorded wall times: celecoxib wB97M-V 97.6 s stock vs 40.4 s mixed, with the certificate
included. These are one warm run per arm, so they are **disclosed, not claimed**. The controlled speed
measurement is in `../data/vv10_wb97mv_pro6000.csv` (2.635× on celecoxib).

**Not validated here:** the master-rebased copy, and any later commit.

**Later (2026-10-10):** the master-rebased branch (`ac4c668e` on upstream master `c1a6e37`) was
validated separately, built from source for sm_120 (engine-repo run `38017282503`, g4psrc r3; see
`VALIDATION-g4psrc.md`):
the VV10 launchers are bitwise identical to the emulation, and the three wB97M-V trio cells pass
with the VV10 certificate in band. See `../rfc/PR_BODY.md`.
