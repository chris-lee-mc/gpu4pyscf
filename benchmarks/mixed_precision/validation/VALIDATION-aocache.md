# GPU validation of the opt-in FP64 AO cache

**Result: PASS**, 2026-10-02, on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

- Branch commit tested: `63af0568d4fd19935bef51b7fd71f161a9cee56f` (`mixed-precision-aocache-v1.8.1`).
- Base: v1.8.1, `5b284c258a4260baef80e3d150b4e7a81a9dbd57`.
- Machine-readable record: `g4pport_aocache_run.json`.

## What was tested

`MixedPrecision(..., ao_cache_fp64=True)` is opt-in and off by default. With it, AO values are
evaluated once per SCF and kept in FP64, plus an FP32 mirror while XC is FP32. Every FP64 XC call
runs `numint.nr_rks`'s own kernels on the cached blocks. The cache is used only where it fits in
`ao_cache_mem_fraction` of the free memory of the device or MIG instance. Otherwise it falls back,
with the fallback recorded.

## Method and results

The overlay method of `VALIDATION.md` was used, with gates fixed before the run.

1. **Base check.** Every modified file was byte-identical to v1.8.1 in the wheel.
2. **Regression.** Upstream DF-RKS, DF-JK, RKS and SCF tests passed **42/42 on the stock wheel and
   42/42 after the overlay.**
3. **New tests.** `dft/tests/test_mixed_precision.py` and `dft/tests/test_mixed_precision_vv10.py`:
   **49/49 passed.** Among them:
   - the cached FP64 path is **bitwise equal** to stock `nr_rks` for LDA, GGA and meta-GGA, with
     tagged and plain densities;
   - a negative control shows that a 1e-12 change in the cache is detected;
   - cache on versus off gives identical schedules and cycle counts;
   - the memory fallbacks, invalidation, empty-block refusal and the opt-in default are each
     tested.
4. **VV10 kernel identity.** Bitwise, 4/4.
5. **Trio** (paracetamol, propranolol, celecoxib × r2SCAN, B3LYP, wB97M-V; every cell opts in):
   - |ΔE| ≤ 4.5e-12 Ha against stock, with identical cycle counts and an FP64 tail;
   - every FP64 XC call was served from the cache, none by stock;
   - the cache was 3.1–10.3 GiB.

## Speed (disclosed, not a claim)

Single warm runs per arm. Geomean of stock/mixed walls over the trio, against the VV10 validation
run without the cache:

| | without cache | with cache |
|---|---|---|
| r2SCAN | 1.59 | **1.81** |
| B3LYP | 1.66 | **1.97** |
| wB97M-V | 2.52 | **2.69** |

Paired, repeated measurements come from the separate benchmark campaign.
