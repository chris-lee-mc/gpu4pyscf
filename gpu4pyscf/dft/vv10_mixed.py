# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Mixed-precision VV10 pair sum for gpu4pyscf.dft.mixed_precision.

The O(N^2) U/W/E pair sum of numint._vv10nlc (VXC_vv10nlc_fock_eval_UWE) is
replaced, through the uwe_kernel keyword, by one of two NVRTC kernels:

* FP32 (FP32_VARIANT): every pair term in FP32, coordinates carried as an
  FP32 hi/lo pair, partial sums over FLUSH j points added into FP64
  accumulators. Used for the early SCF iterations.
* df64 (DF64_VARIANT): every pair op an FP32-pair (double-float) error-free
  transform, ~48 bits, FP64 accumulators. Used for every iteration after the
  switch.

The df64 result of the last SCF iteration is certified against the stock
FP64 kernel after the SCF (certify). Both modules are built once per process
per device, and every entry point is launched on a probe and compared with
an FP64 NumPy reference before use (bind_kernels); a module that does not
build or fails its probe raises.

cupy is imported lazily, so this module imports on a host without a GPU.
'''

import ctypes
import math
import time

import numpy as np

FP32_VARIANT = 'f32_t1_hilo'
DF64_VARIANT = 'df64_f32_t1'

# Per-point bands are relative to max|X| (no floor at 1); DENLC bands are on
# |sum(rho*w*dexc)| in Hartree.
VV32_REL = 2e-5             # FP32 vs FP64
VV32_DENLC = 1e-5
VV10_SWITCH_TOL = 1e-5      # PhaseController tol on |dE_nlc|
VV64_PROBE_REL = 1e-13      # FP64 reference entry vs NumPy, bind probe
VVDF_REL = 1e-10            # df64 vs stock FP64, per point (certificate)
VVDF_DENLC = 1e-11          # df64 vs stock FP64, energy (certificate)
VVDF_PROBE_REL = 1e-12      # df64 vs NumPy FP64, bind probe

FLUSHES = (8, 16, 32)
ITILES = (1, 2)
BLOCK = 128
ENTRY_F64_REF = 'vv10_uwe_f64_ref'
DF64_FLUSHES = (32, 128)
DF64_ITILES = (1, 2)
DF64_VARIANTS = ('df64_f32_t1', 'df64_f32_t2', 'df64_f128_t1', 'df64_f128_t2')

# No fast math, no ftz, no approximate division (NVRTC defaults).
NVRTC_OPTIONS = ("--std=c++14",)


def variant_name(flush, itile, hilo):
    return f"f{int(flush)}_t{int(itile)}_{'hilo' if hilo else 'nohilo'}"


def entry_name(variant):
    return "vv10_uwe_f32_" + variant


VARIANTS = tuple(variant_name(f, t, h) for h in (True, False) for f in FLUSHES for t in ITILES)
ENTRIES = tuple(entry_name(v) for v in VARIANTS) + (ENTRY_F64_REF,)


def df64_variant_name(flush, itile):
    return f"df64_f{int(flush)}_t{int(itile)}"


def df64_entry_name(variant):
    return "vv10_uwe_" + variant


DF64_ENTRIES = tuple(df64_entry_name(v) for v in DF64_VARIANTS)


def parse_df64_variant(variant):
    '''(flush, itile) for one of DF64_VARIANTS.'''
    if variant not in DF64_VARIANTS:
        raise ValueError(f"df64 variant {variant!r} is not one of {DF64_VARIANTS}")
    _df, f, t = variant.split("_")
    return int(f[1:]), int(t[1:])


def parse_variant(variant):
    '''(flush, itile, hilo) for one of VARIANTS.'''
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} is not one of {VARIANTS}")
    f, t, h = variant.split("_")
    return int(f[1:]), int(t[1:]), h == "hilo"


# -- kernel sources -----------------------------------------------------------

_KERNEL_BODY = r"""
#define VV10_BLOCK 128

// FP32 pair terms, FP64 accumulation. One thread owns ITILE i points (i = base + k*VV10_BLOCK),
// j runs over every grid point in VV10_BLOCK-wide tiles staged through shared memory, as stock
// does.
// j-data is packed { hi.x, hi.y, hi.z, omega } and { lo.x, lo.y, lo.z, kappa }, plus rho*w.
template <int FLUSH, int ITILE, bool HILO>
__device__ __forceinline__ void vv10_uwe_f32_body(double* __restrict__ U, double* __restrict__ W,
                                                  double* __restrict__ E,
                                                  const float4* __restrict__ jhi,
                                                  const float4* __restrict__ jlo,
                                                  const float* __restrict__ rw, const int n)
{
    __shared__ float4 s_hi[VV10_BLOCK];
    __shared__ float4 s_lo[VV10_BLOCK];
    __shared__ float s_rw[VV10_BLOCK];

    const int base = blockIdx.x * (VV10_BLOCK * ITILE) + threadIdx.x;
    float4 hi_i[ITILE];
    float4 lo_i[ITILE];
    float pE[ITILE], pU[ITILE], pW[ITILE];
    double aE[ITILE], aU[ITILE], aW[ITILE];
#pragma unroll
    for (int k = 0; k < ITILE; ++k) {
        const int i = base + k * VV10_BLOCK;
        if (i < n) {
            hi_i[k] = jhi[i];
            lo_i[k] = jlo[i];
        } else {
            // inactive: finite dummies (omega = kappa = 1), never written back
            hi_i[k].x = 0.0f; hi_i[k].y = 0.0f; hi_i[k].z = 0.0f; hi_i[k].w = 1.0f;
            lo_i[k].x = 0.0f; lo_i[k].y = 0.0f; lo_i[k].z = 0.0f; lo_i[k].w = 1.0f;
        }
        pE[k] = 0.0f; pU[k] = 0.0f; pW[k] = 0.0f;
        aE[k] = 0.0;  aU[k] = 0.0;  aW[k] = 0.0;
    }

    for (int j0 = 0; j0 < n; j0 += VV10_BLOCK) {
        const int j = j0 + threadIdx.x;
        if (j < n) {
            s_hi[threadIdx.x] = jhi[j];
            s_lo[threadIdx.x] = jlo[j];
            s_rw[threadIdx.x] = rw[j];
        }
        __syncthreads();
        const int jn = min(VV10_BLOCK, n - j0);
        for (int jj = 0; jj < jn; ++jj) {
            const float4 hj = s_hi[jj];
            const float4 lj = s_lo[jj];
            const float rwj = s_rw[jj];
#pragma unroll
            for (int k = 0; k < ITILE; ++k) {
                float dx = __fsub_rn(hi_i[k].x, hj.x);
                float dy = __fsub_rn(hi_i[k].y, hj.y);
                float dz = __fsub_rn(hi_i[k].z, hj.z);
                if (HILO) {
                    dx = __fadd_rn(dx, __fsub_rn(lo_i[k].x, lj.x));
                    dy = __fadd_rn(dy, __fsub_rn(lo_i[k].y, lj.y));
                    dz = __fadd_rn(dz, __fsub_rn(lo_i[k].z, lj.z));
                }
                float r2 = __fmul_rn(dx, dx);
                r2 = fmaf(dy, dy, r2);
                r2 = fmaf(dz, dz, r2);
                const float g_ij = fmaf(hi_i[k].w, r2, lo_i[k].w);
                const float g_ji = fmaf(hj.w, r2, lj.w);
                const float gs = __fadd_rn(g_ij, g_ji);
                const float q = 1.0f / __fmul_rn(__fmul_rn(g_ij, g_ji), gs);
                const float e = -__fmul_rn(rwj, q);
                const float u = __fmul_rn(__fmul_rn(__fmul_rn(e, __fadd_rn(gs, g_ij)), g_ji), q);
                const float w = __fmul_rn(u, r2);
                pE[k] = __fadd_rn(pE[k], e);
                pU[k] = __fadd_rn(pU[k], u);
                pW[k] = __fadd_rn(pW[k], w);
            }
            if (((jj + 1) % FLUSH) == 0) {
#pragma unroll
                for (int k = 0; k < ITILE; ++k) {
                    aE[k] += (double)pE[k]; aU[k] += (double)pU[k]; aW[k] += (double)pW[k];
                    pE[k] = 0.0f; pU[k] = 0.0f; pW[k] = 0.0f;
                }
            }
        }
        // VV10_BLOCK is a multiple of every FLUSH, so this only ever carries the tail of the last,
        // partial tile; flushing an all-zero partial is exact.
#pragma unroll
        for (int k = 0; k < ITILE; ++k) {
            aE[k] += (double)pE[k]; aU[k] += (double)pU[k]; aW[k] += (double)pW[k];
            pE[k] = 0.0f; pU[k] = 0.0f; pW[k] = 0.0f;
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < ITILE; ++k) {
        const int i = base + k * VV10_BLOCK;
        if (i < n) {
            E[i] = 1.5 * aE[k];
            U[i] = -1.5 * aU[k];
            W[i] = -1.5 * aW[k];
        }
    }
}

// The FP64 reference: stock `vv10_fock_eval_UWE_kernel` (vv10.cu:30-96), statement for statement.
extern "C" __global__ void vv10_uwe_f64_ref(double* __restrict__ U, double* __restrict__ W,
                                            double* __restrict__ E,
                                            const double* __restrict__ grid_coord,
                                            const double* __restrict__ rho_weight,
                                            const double* __restrict__ omega,
                                            const double* __restrict__ kappa, const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < ngrids;

    // stock initialises these to NaN; spelled as an intrinsic because NVRTC compiles this without
    // <math.h>, where the macro is not defined. Inactive threads only; never written back.
    const double vv10_nan = __longlong_as_double(0x7ff8000000000000LL);
    double omega_i = vv10_nan;
    double kappa_i = vv10_nan;
    double3 r_i = { vv10_nan, vv10_nan, vv10_nan };
    if (active) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }

    double U_i = 0;
    double W_i = 0;
    double E_i = 0;

    __shared__ double3 shared_omega_kappa_rhow_j[VV10_BLOCK];
    __shared__ double3 shared_r_j[VV10_BLOCK];

    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += VV10_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            shared_omega_kappa_rhow_j[threadIdx.x].x = omega[j];
            shared_omega_kappa_rhow_j[threadIdx.x].y = kappa[j];
            shared_omega_kappa_rhow_j[threadIdx.x].z = rho_weight[j];
            shared_r_j[threadIdx.x].x = grid_coord[j * 3 + 0];
            shared_r_j[threadIdx.x].y = grid_coord[j * 3 + 1];
            shared_r_j[threadIdx.x].z = grid_coord[j * 3 + 2];
        }
        __syncthreads();

        const int block_upper_bound = min(VV10_BLOCK, ngrids - j_block_offset);
        for (int j_in_block = 0; j_in_block < block_upper_bound; j_in_block++) {
            const double omega_j = shared_omega_kappa_rhow_j[j_in_block].x;
            const double kappa_j = shared_omega_kappa_rhow_j[j_in_block].y;
            const double3 r_j = shared_r_j[j_in_block];
            const double rho_weight_j = shared_omega_kappa_rhow_j[j_in_block].z;

            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1;

            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            const double W_ij = U_ij * r_ij2;

            U_i += U_ij;
            W_i += W_ij;
            E_i += E_ij;
        }
        __syncthreads();
    }

    if (active) {
        U[i] = -U_i * 1.5;
        W[i] = -W_i * 1.5;
        E[i] = E_i * 1.5;
    }
}
"""


def _kernel_src():
    wrappers = []
    for v in VARIANTS:
        flush, itile, hilo = parse_variant(v)
        wrappers.append(
            f'extern "C" __global__ void {entry_name(v)}(double* __restrict__ U, '
            f"double* __restrict__ W, double* __restrict__ E, const float4* __restrict__ jhi, "
            f"const float4* __restrict__ jlo, const float* __restrict__ rw, const int n)\n"
            f"{{\n    vv10_uwe_f32_body<{flush}, {itile}, {'true' if hilo else 'false'}>"
            f"(U, W, E, jhi, jlo, rw, n);\n}}\n")
    return _KERNEL_BODY + "\n" + "\n".join(wrappers)


VV10_KERNEL_SRC = _kernel_src()


# A separate module: a df64 build failure cannot touch the FP32 module. The
# DF64-ARITH region is FP32 spelled with __fadd_rn/__fsub_rn/__fmul_rn/fmaf
# only, so the compiler cannot contract it.
_DF64_BODY = r"""
#define VV10_BLOCK 128

struct vv10_df { float hi, lo; };

// DF64-ARITH-BEGIN
__device__ __forceinline__ vv10_df vv10_df_make(float hi, float lo)
{
    vv10_df r;
    r.hi = hi;
    r.lo = lo;
    return r;
}

// s + e == a + b exactly (Knuth TwoSum, no ordering precondition)
__device__ __forceinline__ vv10_df vv10_two_sum(float a, float b)
{
    const float s = __fadd_rn(a, b);
    const float bb = __fsub_rn(s, a);
    const float e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
    return vv10_df_make(s, e);
}

// s + e == a - b exactly (TwoDiff)
__device__ __forceinline__ vv10_df vv10_two_diff(float a, float b)
{
    const float s = __fsub_rn(a, b);
    const float bb = __fsub_rn(s, a);
    const float e = __fsub_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fadd_rn(b, bb));
    return vv10_df_make(s, e);
}

// renormalisation (Dekker FastTwoSum, |a| >= |b| or a == 0)
__device__ __forceinline__ vv10_df vv10_fast_two_sum(float a, float b)
{
    const float s = __fadd_rn(a, b);
    const float e = __fsub_rn(b, __fsub_rn(s, a));
    return vv10_df_make(s, e);
}

__device__ __forceinline__ vv10_df vv10_df_add(const vv10_df a, const vv10_df b)
{
    const vv10_df s = vv10_two_sum(a.hi, b.hi);
    return vv10_fast_two_sum(s.hi, __fadd_rn(s.lo, __fadd_rn(a.lo, b.lo)));
}

// the coordinate difference: TwoDiff on the hi halves, then the lo halves
__device__ __forceinline__ vv10_df vv10_df_sub(const float ah, const float al, const float bh,
                                               const float bl)
{
    const vv10_df s = vv10_two_diff(ah, bh);
    return vv10_fast_two_sum(s.hi, __fadd_rn(s.lo, __fsub_rn(al, bl)));
}

// TwoProd p + e == a.hi * b.hi exactly, then the cross terms
__device__ __forceinline__ vv10_df vv10_df_mul(const vv10_df a, const vv10_df b)
{
    const float p = __fmul_rn(a.hi, b.hi);
    float e = fmaf(a.hi, b.hi, -p);
    e = fmaf(a.hi, b.lo, e);
    e = fmaf(a.lo, b.hi, e);
    return vv10_fast_two_sum(p, e);
}

// 1/d: the IEEE FP32 seed, one Newton step on the full df64 residual; d > 0 always (kappa > 0)
__device__ __forceinline__ vv10_df vv10_df_rcp(const vv10_df d)
{
    const float xn = 1.0f / d.hi;
    float r = fmaf(-d.hi, xn, 1.0f);
    r = fmaf(-d.lo, xn, r);
    return vv10_fast_two_sum(xn, __fmul_rn(xn, r));
}

// One pair term, every op df64, in stock's association order (vv10.cu:65-75).
// i: a = {hx, hy, hz, omega.hi}, b = {lx, ly, lz, omega.lo}, c = {kappa.hi, kappa.lo, -, -}
// j: the same three float4, with c = {kappa.hi, kappa.lo, rw.hi, rw.lo}
__device__ __forceinline__ void vv10_df64_pair(const float4 ai, const float4 bi, const float4 ci,
                                               const float4 aj, const float4 bj, const float4 cj,
                                               vv10_df& e, vv10_df& u, vv10_df& w)
{
    const vv10_df dx = vv10_df_sub(ai.x, bi.x, aj.x, bj.x);
    const vv10_df dy = vv10_df_sub(ai.y, bi.y, aj.y, bj.y);
    const vv10_df dz = vv10_df_sub(ai.z, bi.z, aj.z, bj.z);
    vv10_df r2 = vv10_df_mul(dx, dx);
    r2 = vv10_df_add(r2, vv10_df_mul(dy, dy));
    r2 = vv10_df_add(r2, vv10_df_mul(dz, dz));
    const vv10_df g_ij = vv10_df_add(vv10_df_mul(vv10_df_make(ai.w, bi.w), r2),
                                     vv10_df_make(ci.x, ci.y));
    const vv10_df g_ji = vv10_df_add(vv10_df_mul(vv10_df_make(aj.w, bj.w), r2),
                                     vv10_df_make(cj.x, cj.y));
    const vv10_df gs = vv10_df_add(g_ij, g_ji);
    const vv10_df q = vv10_df_rcp(vv10_df_mul(vv10_df_mul(g_ij, g_ji), gs));
    const vv10_df rq = vv10_df_mul(vv10_df_make(cj.z, cj.w), q);
    e = vv10_df_make(-rq.hi, -rq.lo);
    u = vv10_df_mul(vv10_df_mul(vv10_df_mul(e, vv10_df_add(gs, g_ij)), g_ji), q);
    w = vv10_df_mul(u, r2);
}
// DF64-ARITH-END

// df64 pair terms, renormalised df64 partials over FLUSH consecutive j, then two FP64 adds (hi,
// then lo; both conversions exact) into FP64 accumulators. Tiling as the FP32 body.
template <int FLUSH, int ITILE>
__device__ __forceinline__ void vv10_uwe_df64_body(double* __restrict__ U, double* __restrict__ W,
                                                   double* __restrict__ E,
                                                   const float4* __restrict__ ja,
                                                   const float4* __restrict__ jb,
                                                   const float4* __restrict__ jc, const int n)
{
    __shared__ float4 s_a[VV10_BLOCK];
    __shared__ float4 s_b[VV10_BLOCK];
    __shared__ float4 s_c[VV10_BLOCK];

    const int base = blockIdx.x * (VV10_BLOCK * ITILE) + threadIdx.x;
    float4 a_i[ITILE];
    float4 b_i[ITILE];
    float4 c_i[ITILE];
    vv10_df pE[ITILE], pU[ITILE], pW[ITILE];
    double aE[ITILE], aU[ITILE], aW[ITILE];
#pragma unroll
    for (int k = 0; k < ITILE; ++k) {
        const int i = base + k * VV10_BLOCK;
        if (i < n) {
            a_i[k] = ja[i];
            b_i[k] = jb[i];
            c_i[k] = jc[i];
        } else {
            // inactive: finite dummies (omega = kappa = 1, rw = 0), never written back
            a_i[k].x = 0.0f; a_i[k].y = 0.0f; a_i[k].z = 0.0f; a_i[k].w = 1.0f;
            b_i[k].x = 0.0f; b_i[k].y = 0.0f; b_i[k].z = 0.0f; b_i[k].w = 0.0f;
            c_i[k].x = 1.0f; c_i[k].y = 0.0f; c_i[k].z = 0.0f; c_i[k].w = 0.0f;
        }
        pE[k].hi = 0.0f; pE[k].lo = 0.0f; pU[k].hi = 0.0f; pU[k].lo = 0.0f;
        pW[k].hi = 0.0f; pW[k].lo = 0.0f;
        aE[k] = 0.0;  aU[k] = 0.0;  aW[k] = 0.0;
    }

    for (int j0 = 0; j0 < n; j0 += VV10_BLOCK) {
        const int j = j0 + threadIdx.x;
        if (j < n) {
            s_a[threadIdx.x] = ja[j];
            s_b[threadIdx.x] = jb[j];
            s_c[threadIdx.x] = jc[j];
        }
        __syncthreads();
        const int jn = min(VV10_BLOCK, n - j0);
        for (int jj = 0; jj < jn; ++jj) {
            const float4 aj = s_a[jj];
            const float4 bj = s_b[jj];
            const float4 cj = s_c[jj];
#pragma unroll
            for (int k = 0; k < ITILE; ++k) {
                vv10_df e, u, w;
                vv10_df64_pair(a_i[k], b_i[k], c_i[k], aj, bj, cj, e, u, w);
                pE[k] = vv10_df_add(pE[k], e);
                pU[k] = vv10_df_add(pU[k], u);
                pW[k] = vv10_df_add(pW[k], w);
            }
            if (((jj + 1) % FLUSH) == 0) {
#pragma unroll
                for (int k = 0; k < ITILE; ++k) {
                    aE[k] += (double)pE[k].hi; aE[k] += (double)pE[k].lo;
                    aU[k] += (double)pU[k].hi; aU[k] += (double)pU[k].lo;
                    aW[k] += (double)pW[k].hi; aW[k] += (double)pW[k].lo;
                    pE[k].hi = 0.0f; pE[k].lo = 0.0f; pU[k].hi = 0.0f; pU[k].lo = 0.0f;
                    pW[k].hi = 0.0f; pW[k].lo = 0.0f;
                }
            }
        }
        // VV10_BLOCK is a multiple of every FLUSH, so this only ever carries the tail of the last,
        // partial tile; flushing an all-zero partial is exact.
#pragma unroll
        for (int k = 0; k < ITILE; ++k) {
            aE[k] += (double)pE[k].hi; aE[k] += (double)pE[k].lo;
            aU[k] += (double)pU[k].hi; aU[k] += (double)pU[k].lo;
            aW[k] += (double)pW[k].hi; aW[k] += (double)pW[k].lo;
            pE[k].hi = 0.0f; pE[k].lo = 0.0f; pU[k].hi = 0.0f; pU[k].lo = 0.0f;
            pW[k].hi = 0.0f; pW[k].lo = 0.0f;
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < ITILE; ++k) {
        const int i = base + k * VV10_BLOCK;
        if (i < n) {
            E[i] = 1.5 * aE[k];
            U[i] = -1.5 * aU[k];
            W[i] = -1.5 * aW[k];
        }
    }
}
"""


def _df64_kernel_src():
    wrappers = []
    for v in DF64_VARIANTS:
        flush, itile = parse_df64_variant(v)
        wrappers.append(
            f'extern "C" __global__ void {df64_entry_name(v)}(double* __restrict__ U, '
            f"double* __restrict__ W, double* __restrict__ E, const float4* __restrict__ ja, "
            f"const float4* __restrict__ jb, const float4* __restrict__ jc, const int n)\n"
            f"{{\n    vv10_uwe_df64_body<{flush}, {itile}>(U, W, E, ja, jb, jc, n);\n}}\n")
    return _DF64_BODY + "\n" + "\n".join(wrappers)


VV10_DF64_KERNEL_SRC = _df64_kernel_src()


# -- host helpers -------------------------------------------------------------

def hilo_split(xp, coords):
    '''(hi, lo), both float32: hi = f32(x), lo = f32(x - f64(hi)).'''
    c = xp.asarray(coords, dtype=xp.float64)
    hi = c.astype(xp.float32)
    lo = (c - hi.astype(xp.float64)).astype(xp.float32)
    return hi, lo


# j columns per vectorised step of the NumPy reference.
_CHUNK = 512


def _uwe_f64(coords, rw, om, ka, rows):
    '''(U, W, E) for the i points in rows, summed over all j: the stock kernel
    (VXC_vv10nlc_fock_eval_UWE) in its association order, j sum sequential.'''
    n = coords.shape[0]
    m = rows.shape[0]
    ri = coords[rows]
    om_i, ka_i = om[rows][:, None], ka[rows][:, None]
    acc = np.zeros((3, m), dtype=np.float64)          # E, U, W
    for j0 in range(0, n, _CHUNK):
        j1 = min(n, j0 + _CHUNK)
        dx = ri[:, 0][:, None] - coords[None, j0:j1, 0]
        dy = ri[:, 1][:, None] - coords[None, j0:j1, 1]
        dz = ri[:, 2][:, None] - coords[None, j0:j1, 2]
        r_ij2 = dx * dx + dy * dy + dz * dz
        g_ij = om_i * r_ij2 + ka_i
        g_ji = om[None, j0:j1] * r_ij2 + ka[None, j0:j1]
        g_sum = g_ij + g_ji
        g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum)
        phi_ij = -g_ij_ji_sum_1
        e_ij = rw[None, j0:j1] * phi_ij
        u_ij = e_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1
        w_ij = u_ij * r_ij2
        for s, t in enumerate((e_ij, u_ij, w_ij)):
            seq = np.concatenate([acc[s][:, None], t], axis=1)
            acc[s] = np.add.accumulate(seq, axis=1)[:, -1]
    return -acc[1] * 1.5, -acc[2] * 1.5, acc[0] * 1.5


def _uwe_ref(coords, rw, om, ka):
    coords = np.asarray(coords, dtype=np.float64)
    return _uwe_f64(coords, np.asarray(rw, dtype=np.float64),
                    np.asarray(om, dtype=np.float64), np.asarray(ka, dtype=np.float64),
                    np.arange(coords.shape[0]))


def max_rel(ref, got, xp=np):
    '''max|got - ref| / max|ref|; NaN when max|ref| is zero or anything is
    non-finite, so a band compare against it fails.'''
    try:
        den = float(xp.max(xp.abs(ref))) if ref.size else float("nan")
        num = float(xp.max(xp.abs(got - ref))) if ref.size else float("nan")
    except Exception:
        return float("nan")
    if not (math.isfinite(den) and math.isfinite(num)) or den <= 0.0:
        return float("nan")
    return num / den


def _host(x):
    get = getattr(x, "get", None)
    return np.asarray(get() if callable(get) else x)


def _cupy():
    import cupy
    return cupy


# -- the FP32 module ----------------------------------------------------------

class VV10Kernels:
    '''The FP32 module and its 13 entry points. prep packs the j data once;
    launch runs one FP32 entry on it.'''

    def __init__(self, xp, module, entries):
        self.xp = xp
        self.module = module
        self.entries = dict(entries)

    def prep(self, coords, rho_weight, omega, kappa):
        xp = self.xp
        c = xp.ascontiguousarray(xp.asarray(coords, dtype=xp.float64))
        n = int(c.shape[0])
        hi, lo = hilo_split(xp, c)
        jhi = xp.empty((n, 4), dtype=xp.float32)
        jlo = xp.empty((n, 4), dtype=xp.float32)
        jhi[:, :3] = hi
        jhi[:, 3] = xp.asarray(omega, dtype=xp.float64).astype(xp.float32)
        jlo[:, :3] = lo
        jlo[:, 3] = xp.asarray(kappa, dtype=xp.float64).astype(xp.float32)
        rw32 = xp.ascontiguousarray(xp.asarray(rho_weight, dtype=xp.float64).astype(xp.float32))
        return jhi, jlo, rw32, n

    def launch(self, variant, packed):
        xp = self.xp
        _flush, itile, _hilo = parse_variant(variant)
        jhi, jlo, rw32, n = packed
        U = xp.empty(n, dtype=xp.float64)
        W = xp.empty(n, dtype=xp.float64)
        E = xp.empty(n, dtype=xp.float64)
        if n:
            per_block = BLOCK * itile
            self.entries[entry_name(variant)](((n + per_block - 1) // per_block,), (BLOCK,),
                                              (U, W, E, jhi, jlo, rw32, np.int32(n)))
        return U, W, E

    def uwe(self, variant, coords, rho_weight, omega, kappa):
        return self.launch(variant, self.prep(coords, rho_weight, omega, kappa))

    def uwe_f64_ref(self, coords, rho_weight, omega, kappa):
        xp = self.xp
        c = xp.ascontiguousarray(xp.asarray(coords, dtype=xp.float64))
        n = int(c.shape[0])
        args = [xp.ascontiguousarray(xp.asarray(a, dtype=xp.float64))
                for a in (rho_weight, omega, kappa)]
        U = xp.empty(n, dtype=xp.float64)
        W = xp.empty(n, dtype=xp.float64)
        E = xp.empty(n, dtype=xp.float64)
        if n:
            self.entries[ENTRY_F64_REF](((n + BLOCK - 1) // BLOCK,), (BLOCK,),
                                        (U, W, E, c, args[0], args[1], args[2], np.int32(n)))
        return U, W, E


def probe_inputs():
    '''7-point probe (coords, rho_weight, omega, kappa): |x| ~ 10 with one pair
    1e-3 Bohr apart, the others 0.3-3 Bohr away.'''
    base = np.array([10.25, -7.125, 4.375])
    offs = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 0.31, -0.2], [1.1, -0.4, 0.7],
                     [-2.3, 0.9, 1.4], [0.6, 2.2, -1.9], [-0.8, -1.3, -0.45]])
    coords = base + offs
    rw = np.array([0.021, 0.019, 0.0073, 0.0011, 3.1e-4, 5.2e-5, 2.4e-3])
    omega = np.array([2.9, 2.85, 1.7, 0.95, 0.61, 0.48, 1.2])
    kappa = np.array([1.9, 1.88, 1.5, 1.1, 0.83, 0.66, 1.3])
    return coords, rw, omega, kappa


def bind_vv10_kernel(xp):
    '''(VV10Kernels, "built") or (None, "unbuilt: <reason>"); never raises.

    Every one of the 13 entries is launched on probe_inputs() and compared
    with the NumPy FP64 reference: the FP64 entry within VV64_PROBE_REL, every
    FP32 entry within VV32_REL.'''
    try:
        module = xp.RawModule(code=VV10_KERNEL_SRC, options=NVRTC_OPTIONS)
    except Exception as e:
        return None, f"unbuilt: {type(e).__name__} building the RawModule: {e}"[:400]
    entries = {}
    for name in ENTRIES:
        try:
            entries[name] = module.get_function(name)
        except Exception as e:
            return None, f"unbuilt: {type(e).__name__} fetching {name}: {e}"[:400]
    kernels = VV10Kernels(xp, module, entries)
    coords, rw, omega, kappa = probe_inputs()
    ref = _uwe_ref(coords, rw, omega, kappa)
    try:
        got = {ENTRY_F64_REF: kernels.uwe_f64_ref(xp.asarray(coords), xp.asarray(rw),
                                                  xp.asarray(omega), xp.asarray(kappa))}
        packed = kernels.prep(xp.asarray(coords), xp.asarray(rw), xp.asarray(omega),
                              xp.asarray(kappa))
        for v in VARIANTS:
            got[entry_name(v)] = kernels.launch(v, packed)
    except Exception as e:
        return None, f"unbuilt: {type(e).__name__} in the forced probe launch: {e}"[:400]
    for name in ENTRIES:
        band = VV64_PROBE_REL if name == ENTRY_F64_REF else VV32_REL
        rels = [max_rel(r, _host(g)) for r, g in zip(ref, got[name])]
        if not all(x <= band for x in rels):            # NaN fails this too
            return None, (f"unbuilt: probe {name} max_rel U/W/E = "
                          f"{', '.join(f'{x:.3e}' for x in rels)} against band {band:g}")
    return kernels, "built"


# -- the df64 module ----------------------------------------------------------

class VV10DF64Kernels:
    '''The df64 module and its 4 entry points. prep packs the j data into
    {hx, hy, hz, omega.hi}, {lx, ly, lz, omega.lo}, {kappa.hi, kappa.lo,
    rw.hi, rw.lo}, every FP64 input split by hilo_split.'''

    def __init__(self, xp, module, entries):
        self.xp = xp
        self.module = module
        self.entries = dict(entries)

    def prep(self, coords, rho_weight, omega, kappa):
        xp = self.xp
        c = xp.ascontiguousarray(xp.asarray(coords, dtype=xp.float64))
        n = int(c.shape[0])
        hi, lo = hilo_split(xp, c)
        omh, oml = hilo_split(xp, omega)
        kah, kal = hilo_split(xp, kappa)
        rwh, rwl = hilo_split(xp, rho_weight)
        ja = xp.empty((n, 4), dtype=xp.float32)
        jb = xp.empty((n, 4), dtype=xp.float32)
        jc = xp.empty((n, 4), dtype=xp.float32)
        ja[:, :3] = hi
        ja[:, 3] = omh
        jb[:, :3] = lo
        jb[:, 3] = oml
        jc[:, 0] = kah
        jc[:, 1] = kal
        jc[:, 2] = rwh
        jc[:, 3] = rwl
        return ja, jb, jc, n

    def launch(self, variant, packed):
        xp = self.xp
        _flush, itile = parse_df64_variant(variant)
        ja, jb, jc, n = packed
        U = xp.empty(n, dtype=xp.float64)
        W = xp.empty(n, dtype=xp.float64)
        E = xp.empty(n, dtype=xp.float64)
        if n:
            per_block = BLOCK * itile
            self.entries[df64_entry_name(variant)](((n + per_block - 1) // per_block,), (BLOCK,),
                                                   (U, W, E, ja, jb, jc, np.int32(n)))
        return U, W, E

    def uwe(self, variant, coords, rho_weight, omega, kappa):
        return self.launch(variant, self.prep(coords, rho_weight, omega, kappa))


def probe_inputs_df64():
    '''probe_inputs() plus two points at the dynamic-range edges of the
    nlcgrids: rho at the mask threshold 1e-10, and rho ~ 500 about 30 Bohr
    away, so the largest g and the smallest q meet in one pair.'''
    coords, rw, omega, kappa = probe_inputs()
    kpref = 6.0 * 1.5 * np.pi * (9 * np.pi) ** (-1.0 / 6.0)
    rho_lo, rho_hi = 1e-10, 500.0
    x_lo = coords[0] + np.array([8.0, -6.0, 5.0])
    x_hi = np.array([-14.5, 9.75, -12.0])
    om_lo = math.sqrt(0.01 * 2.0 ** 4 + 4.0 / 3.0 * math.pi * rho_lo)
    om_hi = math.sqrt(0.01 * 16.0 ** 4 + 4.0 / 3.0 * math.pi * rho_hi)
    coords = np.vstack([coords, x_lo, x_hi])
    rw = np.concatenate([rw, [rho_lo * 5.0, rho_hi * 1e-7]])
    omega = np.concatenate([omega, [om_lo, om_hi]])
    kappa = np.concatenate([kappa, [kpref * rho_lo ** (1.0 / 6.0), kpref * rho_hi ** (1.0 / 6.0)]])
    return coords, rw, omega, kappa


def bind_vv10_df64_kernel(xp):
    '''(VV10DF64Kernels, "built") or (None, "unbuilt: <reason>"); never raises.

    Every one of the 4 entries is launched on probe_inputs_df64() and compared
    with the NumPy FP64 reference within VVDF_PROBE_REL.'''
    try:
        module = xp.RawModule(code=VV10_DF64_KERNEL_SRC, options=NVRTC_OPTIONS)
    except Exception as e:
        return None, f"unbuilt: {type(e).__name__} building the df64 RawModule: {e}"[:400]
    entries = {}
    for name in DF64_ENTRIES:
        try:
            entries[name] = module.get_function(name)
        except Exception as e:
            return None, f"unbuilt: {type(e).__name__} fetching {name}: {e}"[:400]
    kernels = VV10DF64Kernels(xp, module, entries)
    coords, rw, omega, kappa = probe_inputs_df64()
    ref = _uwe_ref(coords, rw, omega, kappa)
    got = {}
    try:
        packed = kernels.prep(xp.asarray(coords), xp.asarray(rw), xp.asarray(omega),
                              xp.asarray(kappa))
        for v in DF64_VARIANTS:
            got[df64_entry_name(v)] = kernels.launch(v, packed)
    except Exception as e:
        return None, f"unbuilt: {type(e).__name__} in the df64 forced probe launch: {e}"[:400]
    for name in DF64_ENTRIES:
        rels = [max_rel(r, _host(g)) for r, g in zip(ref, got[name])]
        if not all(x <= VVDF_PROBE_REL for x in rels):  # NaN fails this too
            return None, (f"unbuilt: probe {name} max_rel U/W/E = "
                          f"{', '.join(f'{x:.3e}' for x in rels)} against band {VVDF_PROBE_REL:g}")
    return kernels, "built"


# -- the launchers mixed_precision uses ---------------------------------------

class _Bound:
    '''Both modules, built and probed on one device.'''
    def __init__(self, fp32, df64, record):
        self.fp32 = fp32
        self.df64 = df64
        self.record = record

    def uwe_fp32(self, coords, rw, om, ka):
        return self.fp32.uwe(FP32_VARIANT, coords, rw, om, ka)

    def uwe_df64(self, coords, rw, om, ka):
        return self.df64.uwe(DF64_VARIANT, coords, rw, om, ka)


_BOUND = {}     # device id -> _Bound


def bind_kernels():
    '''Build and probe both modules on the current device, once per process.

    Returns an object with uwe_fp32(coords, rw, om, ka) and
    uwe_df64(coords, rw, om, ka), each -> (U, W, E), and record, the
    {'fp32': {variant, entry, status}, 'df64': {...}} of the build. Raises
    RuntimeError unless both modules are built.'''
    cupy = _cupy()
    dev = cupy.cuda.Device().id
    bound = _BOUND.get(dev)
    if bound is not None:
        return bound
    try:
        k32, s32 = bind_vv10_kernel(cupy)
    except Exception as e:
        k32, s32 = None, f'unbuilt: {type(e).__name__}: {e}'[:400]
    try:
        k64, s64 = bind_vv10_df64_kernel(cupy)
    except Exception as e:
        k64, s64 = None, f'unbuilt: {type(e).__name__}: {e}'[:400]
    record = {'fp32': {'variant': FP32_VARIANT, 'entry': entry_name(FP32_VARIANT),
                       'status': str(s32)},
              'df64': {'variant': DF64_VARIANT, 'entry': df64_entry_name(DF64_VARIANT),
                       'status': str(s64)}}
    if k32 is None or s32 != 'built' or k64 is None or s64 != 'built':
        raise RuntimeError(f'vv10_mixed: VV10 kernels not available on device {dev}: '
                           f'fp32 {s32}; df64 {s64}')
    bound = _Bound(k32, k64, record)
    _BOUND[dev] = bound
    return bound


def uwe_stock(coords, rw, om, ka):
    '''The stock FP64 UWE kernel (numint.libgdft.VXC_vv10nlc_fock_eval_UWE).'''
    cupy = _cupy()
    from gpu4pyscf.dft import numint
    n = coords.shape[0]
    coords, rw, om, ka = (cupy.ascontiguousarray(a, dtype=np.float64)
                          for a in (coords, rw, om, ka))
    U = cupy.empty(n)
    W = cupy.empty(n)
    E = cupy.empty(n)
    stream = cupy.cuda.get_current_stream()
    err = numint.libgdft.VXC_vv10nlc_fock_eval_UWE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U.data.ptr, ctypes.c_void_p),
        ctypes.cast(W.data.ptr, ctypes.c_void_p),
        ctypes.cast(E.data.ptr, ctypes.c_void_p),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(rw.data.ptr, ctypes.c_void_p),
        ctypes.cast(om.data.ptr, ctypes.c_void_p),
        ctypes.cast(ka.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n),
    )
    if err != 0:
        raise RuntimeError('CUDA Error in vv10 Fock kernel')
    return U, W, E


def _cert_record(call, error=None):
    '''A certificate that fails until every field is filled.'''
    nan = float('nan')
    return {'call': call, 'rel': {'E': nan, 'U': nan, 'W': nan}, 'denlc': nan,
            'band_rel': VVDF_REL, 'band_denlc': VVDF_DENLC, 'ok': False,
            'wall_s': nan, 'error': error}


def certify(call, coords, rw, om, ka, U, W, E):
    '''Re-run the stock FP64 UWE kernel on the masked inputs of one df64 call
    and compare with that call's (U, W, E): every max_rel within VVDF_REL and
    |sum(rw * (E - E_stock) / 2)| within VVDF_DENLC. wall_s is the whole
    check, sync to sync. Never raises: an exception gives ok = False and
    error. Returns the certificate record.'''
    cert = _cert_record(call)
    try:
        cupy = _cupy()
        stream = cupy.cuda.get_current_stream()
        stream.synchronize()
        t0 = time.perf_counter()
        U_r, W_r, E_r = uwe_stock(coords, rw, om, ka)
        rel = {'E': max_rel(E_r, E, cupy),
               'U': max_rel(U_r, U, cupy),
               'W': max_rel(W_r, W, cupy)}
        try:
            denlc = abs(float(cupy.dot(rw, 0.5 * (E - E_r))))
        except Exception:
            denlc = float('nan')
        stream.synchronize()
        cert['wall_s'] = time.perf_counter() - t0
        cert['rel'] = rel
        cert['denlc'] = denlc
        cert['ok'] = bool(all(v <= VVDF_REL for v in rel.values())
                          and denlc <= VVDF_DENLC
                          and math.isfinite(cert['wall_s']) and cert['wall_s'] > 0.0)
    except Exception as e:
        cert['ok'] = False
        cert['error'] = f'{type(e).__name__}: {e}'[:400]
    return cert
