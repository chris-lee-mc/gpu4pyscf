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
Opt-in mixed-precision SCF for closed-shell RKS on GPUs with limited FP64
throughput.

Early SCF iterations evaluate selected large contractions in FP32; the SCF
then returns to FP64 before convergence can be accepted.

* XC quadrature (``xc=True``): the density (rho) and XC-potential (vxc)
  contractions run in FP32 against an FP32 copy of the AO values, evaluated
  once per SCF. Functional evaluation (``eval_xc_eff``), grid weights, the
  electron count, the XC-energy reduction and the potential-matrix
  accumulator stay FP64. Every block result is promoted before accumulation.
* DF exchange for global hybrids (``k=True``): the two K contractions of
  ``df_jk.get_jk`` run in FP32 on FP32 copies of the density-fitting
  intermediates. J and the K accumulator stay FP64.
* VV10 nonlocal correlation (``vv10=True``): the O(N^2) pair sum of
  ``numint._vv10nlc`` runs with FP32 pair terms and FP64 accumulation
  (gpu4pyscf.dft.vv10_mixed). After the switch every call runs a df64
  (FP32-pair, ~48-bit) kernel instead; there is no stock FP64 tail.

AO cache (``ao_cache_fp64=True``, opt-in, off by default): the AO values of
every grid block are evaluated once per SCF and kept in FP64, with an FP32
mirror while XC is FP32. Every FP64 XC call, including those of k- or
vv10-only policies, then runs numint.nr_rks's own kernels on the cached
blocks instead of re-evaluating them, with a bitwise-identical result. It
costs 8 bytes per AO value (12 during the FP32 phase) of device memory, so it
is used only when it fits the budget (ao_cache_mem_fraction of the free
memory of the device or MIG instance); otherwise those calls go through
numint.nr_rks unchanged, and the record says which.

Each component switches one way when the change in its energy (E_xc for XC
and K, E_nlc for VV10) between successive iterations falls below its
threshold. A stall detector and an iteration cap back that up. K is never
FP32 on an iteration where XC is FP64.

Convergence contract: convergence is accepted only after two consecutive
iterations built without FP32-phase arithmetic; the VV10 tail is df64 and is
certified against stock FP64 after the SCF. If the convergence tests pass
earlier, the switch is forced and the SCF continues. The density-fitting
get_veff (df_jk._DFHF.get_veff) builds J/K from the full density every
iteration, so no FP32 K contribution survives the FP32 phase. The non-DF
rks.get_veff builds J incrementally; there the first FP64 build after an
FP32 phase is a full rebuild. K itself is only FP32 with density fitting.
The VV10 certificate re-runs the stock FP64 pair sum on the inputs of the
last df64 call; out of band, the SCF is marked not converged and
RuntimeError is raised.

Supported: single GPU, closed-shell RKS. XC and K: no NLC, no
range-separated functionals; K additionally requires density fitting. VV10:
a functional with exactly one VV10 term, run alone (not with XC or K), and a
NumInt whose nr_nlc_vxc accepts vv10_kernel; range-separated functionals and
the non-DF path are allowed. Anything else
raises NotImplementedError when the SCF starts; a VV10 kernel that does not
build or fails its probe raises RuntimeError. Nothing falls back silently.

This is an opt-in performance mode. On FP64-strong GPUs (e.g. H100, A100) it
may not pay off.

Usage::

    mf = gpu4pyscf.dft.RKS(mol, xc='b3lyp').density_fit()
    mf.mixed_precision = MixedPrecision(xc=True, k=True)
    mf.kernel()
    mf.mixed_precision_record   # which iterations ran in which precision

    mf = gpu4pyscf.dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.mixed_precision = MixedPrecision(vv10=True)
'''

import contextlib
import inspect
import math

import numpy as np
import cupy

from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    add_sparse, contract, release_gpu_stack, take_last2d, transpose_sum)
from gpu4pyscf.dft import vv10_mixed
from gpu4pyscf.dft.vv10_mixed import VV10_SWITCH_TOL

FP32 = 'fp32'
FP64 = 'fp64'
DF64 = 'df64'
# record['xc_path'] entries: how each call's XC was evaluated
XC_PATH_FP32 = 'fp32'
XC_PATH_CACHED = 'fp64-cached'
XC_PATH_STOCK = 'fp64-stock'

# Default switch thresholds on |E_xc(n) - E_xc(n-1)| (Hartree). Measured
# settings: 1e-3 for XC on r2SCAN; 3e-4 for XC with 1e-3 for K on B3LYP.
# VV10 switches on |E_nlc(n) - E_nlc(n-1)| at VV10_SWITCH_TOL (1e-5).
# They are controller settings, not error bounds.
XC_SWITCH_TOL = 1e-3
K_SWITCH_TOL = 1e-3
SWITCH_STALL = 2
SWITCH_CALL_CAP = 30
# Largest fraction of free device memory the AO cache may take. With XC
# FP32 the cache holds an FP32 copy of every AO block (4 B per value); with
# ao_cache_fp64=True (opt-in) also an FP64 copy (8 B), kept for the FP64
# calls. The tier is decided from the predicted size before anything is
# allocated: if the FP64 copy does not fit it is not made; if the FP32 copy
# does not fit, XC stays FP64 for the whole SCF. All recorded.
AO_CACHE_MEM_FRACTION = 0.7
# Fraction of free device memory for one FP32 chunk of a cderi block.
K_CHUNK_MEM_FRACTION = 0.25


class MixedPrecision:
    '''Opt-in mixed-precision policy for one SCF object. Every component
    defaults to off; enable them explicitly.'''
    def __init__(self, xc=False, k=False, vv10=False, xc_switch_tol=XC_SWITCH_TOL,
                 k_switch_tol=K_SWITCH_TOL, vv10_switch_tol=VV10_SWITCH_TOL,
                 stall=SWITCH_STALL, call_cap=SWITCH_CALL_CAP,
                 ao_cache_mem_fraction=AO_CACHE_MEM_FRACTION, ao_cache_fp64=False,
                 diis_reset_at_switch=False, warm_start_gorb=None):
        self.xc = bool(xc)
        self.k = bool(k)
        self.vv10 = bool(vv10)
        self.xc_switch_tol = float(xc_switch_tol)
        self.k_switch_tol = float(k_switch_tol)
        self.vv10_switch_tol = float(vv10_switch_tol)
        self.stall = int(stall)
        self.call_cap = int(call_cap)
        self.ao_cache_mem_fraction = float(ao_cache_mem_fraction)
        self.ao_cache_fp64 = bool(ao_cache_fp64)
        # Restart the DIIS subspace on the first iteration built entirely in FP64, so no
        # FP32-phase Fock or error vector enters the tail's extrapolation.
        self.diis_reset_at_switch = bool(diis_reset_at_switch)
        # Warm start: when the SCF starts from a supplied density with orbitals (for example a
        # scanner's previous geometry) and the orbital-gradient norm of the initial Fock is below
        # this value, every later call runs in FP64. None disables the rule.
        if warm_start_gorb is not None:
            warm_start_gorb = float(warm_start_gorb)
            if not warm_start_gorb > 0:
                raise ValueError('warm_start_gorb must be positive or None')
        self.warm_start_gorb = warm_start_gorb

    def __repr__(self):
        extra = ''
        if self.diis_reset_at_switch:
            extra += ', diis_reset_at_switch=True'
        if self.warm_start_gorb is not None:
            extra += f', warm_start_gorb={self.warm_start_gorb:g}'
        return (f'MixedPrecision(xc={self.xc}, k={self.k}, vv10={self.vv10}, '
                f'xc_switch_tol={self.xc_switch_tol:g}, '
                f'k_switch_tol={self.k_switch_tol:g}, '
                f'vv10_switch_tol={self.vv10_switch_tol:g}, '
                f'ao_cache_fp64={self.ao_cache_fp64}{extra})')


class PhaseController:
    '''One-way FP32 -> FP64 switch driven by an energy trace (E_xc, or E_nlc
    for VV10; ``label`` names it in the switch reason).

    It never decides convergence. It only decides the precision of the next
    iteration.'''
    def __init__(self, tol, stall=SWITCH_STALL, cap=SWITCH_CALL_CAP, label='xc'):
        self.tol = float(tol)
        self.label = str(label)
        self.stall = int(stall)
        self.cap = int(cap)
        self.switched = False
        self.switch_call = None
        self.switch_reason = ''
        self.prev_exc = None
        self.prev_dexc = None
        self.no_decrease = 0

    def precision_for(self, call):
        if not self.switched and call >= self.cap:
            self.switch(call, f'call cap {self.cap}')
        return FP64 if self.switched else FP32

    def observe(self, call, exc):
        e = float(exc)
        d = None if self.prev_exc is None else abs(e - self.prev_exc)
        self.prev_exc = e
        if self.switched or d is None:
            return
        if not math.isfinite(d):
            self.switch(call + 1, f'non-finite |dE_{self.label}|')
            return
        if d < self.tol:
            self.switch(call + 1, f'|dE_{self.label}|={d:.3e} < {self.tol:g}')
        elif self.prev_dexc is not None and d >= self.prev_dexc:
            self.no_decrease += 1
            if self.no_decrease >= self.stall:
                self.switch(call + 1, f'|dE_{self.label}| stalled at {d:.3e}')
        else:
            self.no_decrease = 0
        self.prev_dexc = d

    def switch(self, call, reason):
        if self.switched:
            return
        self.switched = True
        self.switch_call = int(call)
        self.switch_reason = reason


def check_supported(mf, policy):
    '''Raise NotImplementedError for a configuration the mode does not support.'''
    from gpu4pyscf.dft import rks
    from gpu4pyscf.__config__ import num_devices
    if not isinstance(policy, MixedPrecision):
        raise TypeError(f'mixed_precision must be a MixedPrecision instance, '
                        f'not {type(policy).__name__}')
    if not isinstance(mf, rks.RKS):
        raise NotImplementedError(
            f'mixed_precision supports closed-shell RKS only, not {type(mf).__name__}')
    if mf.mol.spin != 0:
        raise NotImplementedError('mixed_precision supports closed-shell (spin=0) only')
    if num_devices > 1:
        raise NotImplementedError('mixed_precision supports a single GPU only')
    ni = mf._numint
    if policy.xc or policy.k:
        if mf.do_nlc():
            raise NotImplementedError(
                'mixed_precision xc/k do not support NLC functionals; '
                'use MixedPrecision(vv10=True) alone')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
        if omega != 0 or getattr(mf, 'omega', 0):
            raise NotImplementedError(
                'mixed_precision does not support range-separated functionals')
        if policy.k and hyb != 0 and getattr(mf, 'with_df', None) is None:
            raise NotImplementedError('mixed_precision K requires density fitting')
    if policy.vv10:
        if not mf.do_nlc():
            raise NotImplementedError(
                'mixed_precision vv10=True: the functional has no NLC term, nothing to treat')
        # get_veff passes vv10_kernel=; an override that lacks it would raise
        # TypeError mid-SCF, after the kernels are built
        params = inspect.signature(ni.nr_nlc_vxc).parameters
        if not ('vv10_kernel' in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())):
            raise NotImplementedError(
                f'mixed_precision vv10=True: {type(ni).__name__}.nr_nlc_vxc does not '
                f'accept vv10_kernel')
        # the xc code get_veff hands to nr_nlc_vxc
        code = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
        coefs = ni.nlc_coeff(code)
        if len(coefs) != 1:
            raise NotImplementedError(
                f'mixed_precision vv10=True supports exactly one VV10 term, '
                f'not {len(coefs)}')
        nlc_pars, fac = coefs[0]
        b, c = nlc_pars[0], nlc_pars[1]
        if not all(math.isfinite(float(x)) for x in (b, c, fac)) or b <= 0:
            raise NotImplementedError(
                f'mixed_precision vv10=True: unsupported VV10 coefficients '
                f'b={b}, C={c}, factor={fac}')


class _SCFState:
    '''Per-SCF state: controllers, the FP32 AO copy, the VV10 kernels, and
    the per-iteration record.'''
    def __init__(self, mf, policy):
        self.policy = policy
        self.log = logger.new_logger(mf)
        ni = mf._numint
        self.hybrid = bool(ni.libxc.is_hybrid_xc(mf.xc))
        self.xc_on = policy.xc
        self.k_on = policy.k and self.hybrid
        self.vv10_on = policy.vv10
        self.xc_ctrl = PhaseController(policy.xc_switch_tol, policy.stall, policy.call_cap)
        self.k_ctrl = PhaseController(policy.k_switch_tol, policy.stall, policy.call_cap)
        self.vv10_ctrl = PhaseController(policy.vv10_switch_tol, policy.stall, policy.call_cap,
                                         label='nlc')
        if not self.xc_on:
            self.xc_ctrl.switch(0, 'XC mixed precision not requested')
        if not self.k_on:
            self.k_ctrl.switch(0, 'K mixed precision not requested or not a hybrid')
        self.vv10_kernels = None
        if self.vv10_on:
            # raises RuntimeError if a module does not build or fails its probe
            self.vv10_kernels = vv10_mixed.bind_kernels()
        else:
            self.vv10_ctrl.switch(0, 'VV10 mixed precision not requested')
        self.vv10_kept = None
        self.ao_cache = None
        self.ao_cache_note = ''
        self.ao_cache_tier = None
        self.ao_cache_tried = False
        self.ao_cache_bytes = (0, 0)
        self.mirror_released_call = None
        self.cderi_prebuilt = False
        self.call = 0
        self.k_ran_fp32 = False
        self.k_rebuilt = False
        self.clean_streak = 0
        self.just_clean = False
        self.forced = ''
        self.record = {'policy': repr(policy), 'xc': [], 'k': [],
                       'diis_reset_call': None, 'warm_start': None,
                       'k_full_rebuild_call': None, 'ao_cache': '', 'forced': '',
                       'vv10': [], 'vv10_n_masked': [], 'vv10_cert': None,
                       'xc_path': []}
        self._cur = None

    # -- per-iteration protocol, driven by rks.get_veff ------------------------
    def begin_call(self):
        self.call += 1
        xc_prec = self.xc_ctrl.precision_for(self.call) if self.xc_on else FP64
        cache = self.ao_cache
        if xc_prec == FP64 and cache is not None and cache.has32:
            # the FP32 phase is over: release the FP32 copy, keep any FP64 one
            if cache.has64:
                cache.drop_mirror()
            else:
                self.ao_cache = None
            self.mirror_released_call = self.call
            cache = None
            _release_pool()
        k_prec = FP64
        if self.k_on:
            if self.xc_on and xc_prec == FP64:
                self.k_ctrl.switch(self.call, 'XC returned to FP64')
            k_prec = self.k_ctrl.precision_for(self.call)
        full_rebuild = False
        if self.k_on and k_prec == FP64 and self.k_ran_fp32 and not self.k_rebuilt:
            full_rebuild = True
            self.k_rebuilt = True
            self.record['k_full_rebuild_call'] = self.call
        vv10_prec = FP64
        if self.vv10_on:
            vv10_prec = DF64 if self.vv10_ctrl.precision_for(self.call) == FP64 else FP32
        self._cur = {'xc': xc_prec, 'k': k_prec, 'k_applied': FP64,
                     'vv10': vv10_prec, 'vv10_launches': 0, 'vv10_n_masked': None,
                     'xc_path': XC_PATH_STOCK}
        return xc_prec, k_prec, full_rebuild, vv10_prec

    def vv10_kernel(self):
        '''The uwe_kernel for this call's nr_nlc_vxc: None (stock FP64), the
        FP32 launcher or the df64 launcher. Call after begin_call.'''
        prec = self._cur['vv10']
        if prec == FP32:
            return self._uwe_fp32
        if prec == DF64:
            return self._uwe_df64
        return None

    def _uwe_fp32(self, coords, rw, om, ka):
        cur = self._cur
        cur['vv10_launches'] += 1
        cur['vv10_n_masked'] = int(coords.shape[0])
        return self.vv10_kernels.uwe_fp32(coords, rw, om, ka)

    def _uwe_df64(self, coords, rw, om, ka):
        cur = self._cur
        cur['vv10_launches'] += 1
        cur['vv10_n_masked'] = int(coords.shape[0])
        U, W, E = self.vv10_kernels.uwe_df64(coords, rw, om, ka)
        # kept on the device for the certificate; the next df64 call replaces it
        self.vv10_kept = (self.call, coords, rw, om, ka, U, W, E)
        return U, W, E

    def end_call(self, exc_xc, enlc=None):
        cur = self._cur
        if self.vv10_on:
            if enlc is None:
                raise RuntimeError(
                    f'mixed_precision: call {self.call} did not report E_nlc to the '
                    f'VV10 policy; this get_veff path is not supported')
            if cur['vv10_launches'] != 1:
                raise RuntimeError(
                    f'mixed_precision: call {self.call} ran the {cur["vv10"]} VV10 '
                    f'kernel {cur["vv10_launches"]} times, not once')
        self.xc_ctrl.observe(self.call, exc_xc)
        self.k_ctrl.observe(self.call, exc_xc)
        if enlc is not None:
            self.vv10_ctrl.observe(self.call, enlc)
        if cur['k_applied'] == FP32:
            self.k_ran_fp32 = True
        k_clean = (not self.k_ran_fp32) or self.k_rebuilt
        clean = (cur['xc'] == FP64 and cur['k_applied'] == FP64 and k_clean
                 and cur['vv10'] != FP32)
        self.clean_streak = self.clean_streak + 1 if clean else 0
        # The first clean call after at least one call that was not: the switch has landed.
        self.just_clean = clean and self.clean_streak == 1 and self.call > 1
        self.record['xc'].append(cur['xc'])
        self.record['xc_path'].append(cur['xc_path'])
        self.record['k'].append(cur['k_applied'])
        self.record['vv10'].append(cur['vv10'])
        self.record['vv10_n_masked'].append(cur['vv10_n_masked'])
        self._cur = None

    def fp64_tail(self):
        '''True when the last two iterations were built without FP32-phase
        arithmetic: XC and K in FP64, VV10 in df64 (or stock FP64 when off).'''
        return self.clean_streak >= 2

    def force_fp64(self, reason):
        self.xc_ctrl.switch(self.call + 1, reason)
        self.k_ctrl.switch(self.call + 1, reason)
        self.vv10_ctrl.switch(self.call + 1, reason)
        if not self.forced:
            self.forced = reason
            self.record['forced'] = reason

    @contextlib.contextmanager
    def k_scope(self, with_df, k_prec):
        '''Request FP32 K from df_jk.get_jk for the duration of one get_k call.'''
        if with_df is None or k_prec != FP32:
            yield
            return
        with_df._k_precision = FP32
        with_df._k_precision_applied = 0
        try:
            yield
        finally:
            applied = getattr(with_df, '_k_precision_applied', 0)
            with_df._k_precision = None
            with_df._k_precision_applied = 0
            if applied:
                self._cur['k_applied'] = FP32

    def drop_ao_cache(self, why):
        '''Release the whole AO cache for the rest of this SCF; later XC calls
        run as stock does.'''
        self.ao_cache = None
        _release_pool()
        self.ao_cache_note += f'; dropped at call {self.call}: {why}'
        self.log.warn('mixed_precision: AO cache dropped: %s', why)

    def finish(self, mf):
        if self.ao_cache is not None or self.mirror_released_call is not None:
            self.ao_cache = None
            _release_pool()
        rec = self.record
        rec['cderi_prebuilt'] = self.cderi_prebuilt
        rec['ao_cache'] = self.ao_cache_note
        rec['ao_cache_tier'] = self.ao_cache_tier
        rec['ao_cache_bytes64'], rec['ao_cache_bytes32'] = self.ao_cache_bytes
        rec['ao_cache_mirror_released_call'] = self.mirror_released_call
        rec['xc_fp64_cached'] = rec['xc_path'].count(XC_PATH_CACHED)
        rec['xc_fp64_stock'] = rec['xc_path'].count(XC_PATH_STOCK)
        rec['xc_switch_call'] = self.xc_ctrl.switch_call
        rec['xc_switch_reason'] = self.xc_ctrl.switch_reason
        rec['k_switch_call'] = self.k_ctrl.switch_call
        rec['k_switch_reason'] = self.k_ctrl.switch_reason
        rec['vv10_switch_call'] = self.vv10_ctrl.switch_call
        rec['vv10_switch_reason'] = self.vv10_ctrl.switch_reason
        rec['vv10_n_fp32'] = rec['vv10'].count(FP32)
        rec['vv10_n_df64'] = rec['vv10'].count(DF64)
        rec['vv10_tol'] = self.policy.vv10_switch_tol
        rec['vv10_kernel'] = None
        if self.vv10_kernels is not None:
            rec['vv10_kernel'] = {k: dict(v) for k, v in self.vv10_kernels.record.items()}
        rec['tail_precision'] = {'xc': FP64, 'k': FP64,
                                 'vv10': DF64 if self.vv10_on else FP64}
        rec['fp64_tail'] = self.fp64_tail()
        mf.mixed_precision_record = rec

    def certify(self):
        '''Certify the last df64 VV10 call against stock FP64; the record, or
        None when no df64 call ran.'''
        kept, self.vv10_kept = self.vv10_kept, None
        if kept is None:
            return None
        cert = vv10_mixed.certify(*kept)
        self.record['vv10_cert'] = cert
        return cert


def begin(mf):
    '''Create the per-SCF state if mf.mixed_precision is set; None otherwise.'''
    policy = getattr(mf, 'mixed_precision', None)
    if policy is None:
        return None
    check_supported(mf, policy)
    state = _SCFState(mf, policy)
    if policy.ao_cache_fp64:
        state.cderi_prebuilt = _prebuild_cderi(mf)
    mf._mixed_precision_state = state
    return state


def _prebuild_cderi(mf):
    '''With the AO cache opted in, build the DF tensor before the first XC call
    builds the cache, so that DF places it (device or host) as it would
    without the cache. Only where the first get_jk would build it anyway: a
    hybrid functional with density fitting. A non-hybrid computes J without
    it. The same with_df.build() call get_jk makes. Returns True if built.'''
    with_df = getattr(mf, 'with_df', None)
    if with_df is None or getattr(with_df, '_cderi', None) is not None:
        return False
    if not mf._numint.libxc.is_hybrid_xc(mf.xc):
        return False
    with_df.build()
    return True


def end(mf, failed=False):
    '''Close the SCF: certify the VV10 tail (unless the SCF raised), store
    mf.mixed_precision_record and clear the state. Raises RuntimeError, with
    mf.converged set to False, when the certificate is out of band.'''
    state = getattr(mf, '_mixed_precision_state', None)
    if state is None:
        return
    cert = None
    try:
        if not failed:
            cert = state.certify()
        state.vv10_kept = None
        state.finish(mf)
    finally:
        mf._mixed_precision_state = None
    if cert is not None and not cert['ok']:
        mf.converged = False
        rel = cert['rel']
        raise RuntimeError(
            f'mixed_precision: VV10 certificate out of band on call {cert["call"]}: '
            f'max_rel E/U/W = {rel["E"]:.3e}/{rel["U"]:.3e}/{rel["W"]:.3e} '
            f'(band {cert["band_rel"]:g}), |dE_nlc| = {cert["denlc"]:.3e} '
            f'(band {cert["band_denlc"]:g}), error = {cert["error"]}')


# -- XC quadrature ----------------------------------------------------------

class _AOCache:
    '''AO values of every non-empty grid block for one SCF, as an FP64 copy,
    an FP32 copy, or both, with each block's AO indices and grid offsets and
    what the cache was built for.'''
    def __init__(self, blocks, has64, has32, opt, coords, ao_deriv, n_empty):
        self.blocks = blocks        # list of [ao64, ao32, idx, p0, p1]
        self.has64 = has64
        self.has32 = has32
        self.opt = opt
        self.coords = coords
        self.ao_deriv = ao_deriv
        self.n_empty = n_empty

    def blocks32(self):
        for ao64, ao32, idx, p0, p1 in self.blocks:
            yield ao32, idx, p0, p1

    def blocks64(self):
        for ao64, ao32, idx, p0, p1 in self.blocks:
            yield ao64, idx, p0, p1

    def drop_mirror(self):
        for b in self.blocks:
            b[1] = None
        self.has32 = False

    def mismatch(self, opt, grids, ao_deriv):
        '''Why this cache cannot serve a call with these inputs, or None.'''
        if opt is not self.opt:
            return 'ni.gdftopt is not the one the cache was built for'
        if grids.coords is not self.coords:
            return 'grids.coords is not the array the cache was built for'
        if ao_deriv != self.ao_deriv:
            return f'ao_deriv {ao_deriv}, cache built for {self.ao_deriv}'
        return None


def _free_device_bytes():
    return (cupy.cuda.runtime.memGetInfo()[0]
            + cupy.get_default_memory_pool().free_bytes())


def _release_pool():
    '''Return the memory of released AO copies to the device, so that later
    consumers that exclude the memory pool (e.g. a DF build) see it.'''
    cupy.get_default_memory_pool().free_all_blocks()


def _predict_ao_values(opt, grids, ao_deriv):
    '''(number of AO values block_loop will produce, number of empty blocks),
    from the grid's AO sparsity index, without evaluating any AO.'''
    from gpu4pyscf.dft.numint import MIN_BLK_SIZE
    ngrids = grids.coords.shape[0]
    comp = 1 if ao_deriv == 0 else (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) // 6
    nvals = n_empty = 0
    for block_id, entry in enumerate(grids.get_non0ao_idx(opt)):
        ip0 = block_id * MIN_BLK_SIZE
        if ip0 >= ngrids:
            break
        ng = min(ip0 + MIN_BLK_SIZE, ngrids) - ip0
        nao_sub = len(entry[1])
        if nao_sub == 0:
            n_empty += 1
        nvals += comp * nao_sub * ng
    return nvals, n_empty


def _build_ao_cache(ni, opt, grids, ao_deriv, budget, want64, want32):
    '''Evaluate every AO block once. Returns (cache, tier, note); cache is None
    when nothing requested fits. The tier is decided from the predicted size
    before any copy is allocated: with both copies requested and only the
    FP32 one fitting, no FP64 copy is made (tier 'fp32'). A grid with an empty
    block gets no FP64 copy: numint.nr_rks skips such a block without
    advancing its grid offset, which the cached FP64 path cannot reproduce.'''
    nvals, n_empty = _predict_ao_values(opt, grids, ao_deriv)
    keep64 = want64
    why64 = ''
    if keep64 and n_empty:
        keep64, why64 = False, f'{n_empty} empty grid blocks'
    if keep64 and nvals * (8 + 4 * want32) > budget:
        keep64, why64 = False, ('FP64 copy and FP32 mirror do not fit' if want32
                                else 'FP64 copy does not fit')
    if not want32:
        if not keep64:
            return None, None, (f'not built: {why64} (needs {nvals * 8 / 2**30:.2f} GiB, '
                                f'budget {budget / 2**30:.2f} GiB); FP64 XC as stock')
    elif nvals * 4 > budget:
        return None, None, (f'did not fit: needs > {nvals * 4 / 2**30:.2f} GiB, '
                            f'budget {budget / 2**30:.2f} GiB; XC stays FP64')

    sorted_mol = opt._sorted_mol
    nao = sorted_mol.nao
    blocks = []
    ngrids = grids.coords.shape[0]
    got = 0
    p0 = p1 = 0
    for ao, idx, weight, _ in ni.block_loop(sorted_mol, grids, nao, ao_deriv,
                                            max_memory=None,
                                            grid_range=(0, ngrids),
                                            strict_grid_order=True):
        p0, p1 = p1, p1 + weight.size
        if len(idx) == 0:
            continue
        got += ao.size
        if got > nvals:
            break
        blocks.append([ao.copy() if keep64 else None,
                       ao.astype(np.float32) if want32 else None,
                       cupy.asarray(idx).copy(), p0, p1])
    if got != nvals or p1 != ngrids:
        # the prediction disagrees with block_loop: keep nothing rather than guess
        blocks = None
        _release_pool()
        return None, None, (f'not built: block_loop produced {got} values over {p1} grid '
                            f'points, predicted {nvals} over {ngrids}')
    tier = {(True, True): 'fp64+fp32', (False, True): 'fp32', (True, False): 'fp64'}[
        (keep64, want32)]
    cache = _AOCache(blocks, keep64, want32, opt, grids.coords, ao_deriv, n_empty)
    cache.nvals = nvals
    gib = nvals * (8 * keep64 + 4 * want32) / 2**30
    note = f'built: {gib:.2f} GiB, tier {tier}'
    if want64 and not keep64:
        note += f' (no FP64 copy: {why64})'
    return cache, tier, note


def _nr_rks_overridden(ni):
    from gpu4pyscf.dft import numint
    return (getattr(type(ni), 'nr_rks', None) is not numint.nr_rks
            or 'nr_rks' in vars(ni))


def _ensure_ao_cache(state, ni, opt, grids, ao_deriv, want32):
    '''Build the SCF's AO cache on its first XC call; never retried.'''
    if state.ao_cache_tried:
        return state.ao_cache
    state.ao_cache_tried = True
    want64 = state.policy.ao_cache_fp64 and not _nr_rks_overridden(ni)
    if not (want64 or want32):
        return None
    budget = state.policy.ao_cache_mem_fraction * _free_device_bytes()
    cache, tier, note = _build_ao_cache(ni, opt, grids, ao_deriv, budget, want64, want32)
    state.ao_cache = cache
    state.ao_cache_tier = tier
    state.ao_cache_note = note
    if cache is not None:
        state.ao_cache_bytes = (8 * cache.nvals * cache.has64, 4 * cache.nvals * cache.has32)
    if cache is None or (want64 and not cache.has64):
        state.log.warn('mixed_precision: AO cache %s', note)
    return cache


def _rho_dot_unfused(a, b, axis=0):
    return (a * b).sum(axis=axis)


_rho_dot_impl = None


def _build_rho_dot():
    '''sum(a*b, axis) as one fused reduction (no A-sized temporary), after
    a small value check; the unfused form otherwise. Accumulates in the
    input dtype.'''
    try:
        kern = cupy.ReductionKernel('T x, T y', 'T out', 'x * y', 'a + b',
                                    'out = a', '0', 'mp_rho_dot')

        def fused(a, b, axis=0):
            if a.shape != b.shape:
                a = cupy.broadcast_to(a, b.shape)
            return kern(a, b, axis=axis)
        x = cupy.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
        stack = cupy.stack([x, 2 * x])
        ok = (fused(x, x, axis=0).get().tolist() == [17., 29., 45.] and
              fused(x[None], stack, axis=1).get().tolist()
              == [[17., 29., 45.], [34., 58., 90.]])
        if ok:
            return fused
    except Exception:
        pass
    return _rho_dot_unfused


def _rho_dot(a, b, axis=0):
    global _rho_dot_impl
    if _rho_dot_impl is None:
        _rho_dot_impl = _build_rho_dot()
    return _rho_dot_impl(a, b, axis=axis)


def _eval_rho_dm32(ao, dm, xctype):
    '''numint.eval_rho for hermi=1, FP32 inputs.'''
    if xctype == 'LDA':
        return _rho_dot(dm.dot(ao), ao)
    ng = ao.shape[-1]
    c = cupy.matmul(dm, ao)                      # (4, nao, ng)
    rho = cupy.zeros((5 if xctype == 'MGGA' else 4, ng), dtype=ao.dtype)
    rho[:4] = _rho_dot(c[0][None], ao, axis=1)
    rho[1:4] *= 2
    if xctype == 'MGGA':
        rho[4] = _rho_dot(c[1:4], ao[1:4], axis=1).sum(axis=0) * .5
    return rho


def _eval_rho_mo32(ao, cpos, xctype):
    '''numint._eval_rho2, FP32 inputs.'''
    if xctype == 'LDA':
        c0 = cpos.T.dot(ao)
        return _rho_dot(c0, c0)
    ng = ao.shape[-1]
    c = cupy.matmul(cpos.T, ao)                  # (4, nocc, ng)
    rho = cupy.zeros((5 if xctype == 'MGGA' else 4, ng), dtype=ao.dtype)
    rho[:4] = _rho_dot(c[0][None], c, axis=1)
    rho[1:4] *= 2
    if xctype == 'MGGA':
        rho[4] = _rho_dot(c[1:4], c[1:4], axis=1).sum(axis=0) * .5
    return rho


def _vxc_block32(ao, wv, xctype):
    '''One block of the vxc contraction, FP32; mirrors _nr_rks_task.'''
    if xctype == 'LDA':
        aow = ao * wv[0]
        return ao.dot(aow.T)
    aow = ao[0] * wv[0]
    for n in range(1, 4):
        aow += ao[n] * wv[n]
    if xctype == 'GGA':
        return ao[0].dot(aow.T)
    w = .5 * wv[4]
    block = ao[1].dot((ao[1] * w).T)
    block += ao[2].dot((ao[2] * w).T)
    block += ao[3].dot((ao[3] * w).T)
    block += ao[0].dot(aow.T)
    return block


def nr_rks_fp32(state, ni, mol, grids, xc_code, dms):
    '''FP32-contraction counterpart of numint.nr_rks (single device, spin 0).

    Returns (nelec, excsum, vmat) like nr_rks, or None when the FP32 AO copy
    does not fit, in which case the caller uses the stock FP64 path.'''
    xctype = ni._xc_type(xc_code)
    if xctype == 'HF':
        return None
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    sorted_mol = opt._sorted_mol
    nao = sorted_mol.nao
    ao_deriv = 0 if xctype == 'LDA' else 1

    cache = _ensure_ao_cache(state, ni, opt, grids, ao_deriv, want32=True)
    if cache is None or not cache.has32:
        if not state.xc_ctrl.switched:
            state.xc_ctrl.switch(state.call, 'FP32 AO copy did not fit')
        return None

    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms, 'mo_occ', None)
    if mo_coeff is not None:
        mo_coeff = opt.sort_orbitals(cupy.asarray(mo_coeff), axis=[0])
        mo_occ = cupy.asarray(mo_occ)
        cpos = cupy.asarray(mo_coeff[:, mo_occ > 0], order='C')
        cpos *= mo_occ[mo_occ > 0] ** .5
        cpos32 = cpos.astype(np.float32)
        dm = None
    else:
        dm = opt.sort_orbitals(cupy.asarray(dms), axis=[0, 1])
        cpos32 = None

    ngrids = grids.coords.shape[0]
    nvar = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    rho_tot = cupy.zeros((nvar, ngrids))
    for ao, idx, p0, p1 in cache.blocks32():
        if cpos32 is not None:
            rho_tot[:, p0:p1] = _eval_rho_mo32(ao, cupy.take(cpos32, idx, axis=0), xctype)
        else:
            dm_mask = take_last2d(dm, idx).astype(np.float32)
            rho_tot[:, p0:p1] = _eval_rho_dm32(ao, dm_mask, xctype)

    weights = cupy.asarray(grids.weights)
    den = rho_tot[0] * weights
    nelec = float(den.sum())
    exc, vxc = ni.eval_xc_eff(xc_code, rho_tot, deriv=1, xctype=xctype, spin=0)[:2]
    exc = cupy.asarray(exc, order='C')
    wv = cupy.asarray(vxc, order='C')
    excsum = float(cupy.dot(den, exc).get())
    wv *= weights
    if xctype == 'GGA':
        wv[0] *= .5
    if xctype == 'MGGA':
        wv[[0, 4]] *= .5
    wv32 = wv.astype(np.float32)
    rho_tot = den = exc = vxc = wv = None

    vmat = cupy.zeros((nao, nao))
    for ao, idx, p0, p1 in cache.blocks32():
        block = _vxc_block32(ao, wv32[:, p0:p1], xctype)
        add_sparse(vmat, block.astype(np.float64), idx)
    vmat = opt.unsort_orbitals(vmat, axis=[0, 1])
    if xctype != 'LDA':
        transpose_sum(vmat)
    if state._cur is not None:
        state._cur['xc_path'] = XC_PATH_FP32
    return nelec, excsum, vmat


def nr_rks_fp64_cached(state, ni, mol, grids, xc_code, dms):
    '''numint.nr_rks (single device, spin 0, hermi=1) on the cached FP64 AO
    blocks: the statements of nr_rks and _nr_rks_task with block_loop
    replaced by the cache, so the result is bitwise that of ni.nr_rks.

    Returns (nelec, excsum, vmat), or None when the cache cannot serve this
    call; the caller then runs ni.nr_rks.'''
    from gpu4pyscf.dft import numint
    if not state.policy.ao_cache_fp64:
        return None
    if _nr_rks_overridden(ni):
        if not state.ao_cache_tried:
            state.ao_cache_tried = True
            state.ao_cache_note = f'not built: {type(ni).__name__} overrides nr_rks'
        return None
    xctype = ni._xc_type(xc_code)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        return None
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    ao_deriv = 0 if xctype == 'LDA' else 1
    cache = _ensure_ao_cache(state, ni, opt, grids, ao_deriv, want32=False)
    if cache is None or not cache.has64:
        return None
    why = cache.mismatch(opt, grids, ao_deriv)
    if why is None and cache.n_empty:
        why = f'{cache.n_empty} empty grid blocks'
    ngrids_glob = grids.coords.shape[0]
    if why is None and numint.gen_grid_range(ngrids_glob, 0) != (0, ngrids_glob):
        why = 'grid range is not the whole grid'
    if why is not None:
        state.drop_ao_cache(why)
        return None

    # numint.nr_rks
    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms, 'mo_occ', None)
    if mo_coeff is not None:
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    else:
        assert dms.ndim == 2
        dms = cupy.asarray(dms)
        dms = opt.sort_orbitals(dms, axis=[0, 1])
    release_gpu_stack()
    cupy.cuda.get_current_stream().synchronize()

    # numint._nr_rks_task, device 0, with_lapl=False, hermi=1
    hermi = 1
    dm = dms
    if isinstance(dm, cupy.ndarray):
        assert dm.ndim == 2
        dm = cupy.asarray(dm)
    if mo_coeff is not None:
        mo_coeff = cupy.asarray(mo_coeff)
    if mo_occ is not None:
        mo_occ = cupy.asarray(mo_occ)
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    ngrids_local = ngrids_glob
    if xctype == 'LDA':
        rho_tot = cupy.empty([1, ngrids_local])
    elif xctype == 'GGA':
        rho_tot = cupy.empty([4, ngrids_local])
    else:
        rho_tot = cupy.empty([5, ngrids_local])

    if mo_coeff is None:
        buf = cupy.empty(numint.MIN_BLK_SIZE * nao)
        dm_mask_buf = cupy.empty(nao * nao)
    else:
        mo_coeff = cupy.asarray(mo_coeff[:, mo_occ > 0], order='C')
        mo_coeff *= mo_occ[mo_occ > 0]**.5
        nocc = mo_coeff.shape[1]
        mo_buf = cupy.empty(nao * nocc)
        buf = cupy.empty(numint.MIN_BLK_SIZE * max(2 * nocc, nao))

    for ao_mask, idx, p0, p1 in cache.blocks64():
        nao_sub = len(idx)
        if mo_coeff is None:
            dm_mask = dm_mask_buf[:nao_sub**2].reshape(nao_sub, nao_sub)
            dm_mask = take_last2d(dm, idx, out=dm_mask)
            rho_tot[:, p0:p1] = numint.eval_rho(_sorted_mol, ao_mask, dm_mask,
                                                xctype=xctype, hermi=hermi,
                                                with_lapl=False, buf=buf)
        else:
            cpos = mo_buf[:nao_sub * nocc].reshape(nao_sub, nocc)
            cpos = cupy.take(mo_coeff, idx, axis=0, out=cpos)
            rho_tot[:, p0:p1] = numint._eval_rho2(ao_mask, cpos, xctype, False, buf=buf)
    dm_mask_buf = mo_buf = mo_coeff = None

    weights = cupy.asarray(grids.weights[0:ngrids_local])
    den = rho_tot[0] * weights
    nelec = float(den.sum())
    exc, vxc = ni.eval_xc_eff(xc_code, rho_tot, deriv=1, xctype=xctype, spin=0)[:2]
    vxc = cupy.asarray(vxc, order='C')
    exc = cupy.asarray(exc, order='C')
    excsum = float(cupy.dot(den, exc).get())
    wv = vxc
    wv *= weights
    if xctype == 'GGA':
        wv[0] *= .5
    if xctype == 'MGGA':
        wv[[0, 4]] *= .5
    exc = den = vxc = rho_tot = weights = None

    vtmp_buf = cupy.empty(nao * nao)
    vmat = cupy.zeros((nao, nao))
    for ao_mask, idx, p0, p1 in cache.blocks64():
        nao_sub = len(idx)
        vtmp = cupy.ndarray((nao_sub, nao_sub), memptr=vtmp_buf.data)
        if xctype == 'LDA':
            aow = numint._scale_ao(ao_mask, wv[0, p0:p1], out=buf)
            add_sparse(vmat, ao_mask.dot(aow.T, out=vtmp), idx)
        elif xctype == 'GGA':
            aow = numint._scale_ao(ao_mask, wv[:, p0:p1], out=buf)
            add_sparse(vmat, ao_mask[0].dot(aow.T, out=vtmp), idx)
        else:
            vtmp = numint._tau_dot(ao_mask, ao_mask, wv[4, p0:p1], buf=buf, out=vtmp)
            aow = numint._scale_ao(ao_mask, wv[:4, p0:p1], out=buf)
            vtmp = contract('ig,jg->ij', ao_mask[0], aow, beta=1., out=vtmp)
            add_sparse(vmat, vtmp, idx)

    # numint.nr_rks, after the device reduction (one device: the array itself)
    vmat = opt.unsort_orbitals(vmat, axis=[0, 1])
    nelec = sum([nelec])
    excsum = sum([excsum])
    if xctype != 'LDA':
        transpose_sum(vmat)
    if numint.FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()
    if state._cur is not None:
        state._cur['xc_path'] = XC_PATH_CACHED
    return nelec, excsum, vmat
