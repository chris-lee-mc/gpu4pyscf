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

import math
import unittest
from unittest import mock
import numpy as np
import cupy
import pyscf
from gpu4pyscf.dft import rks, uks, numint
from gpu4pyscf.dft import mixed_precision as mp
from gpu4pyscf.dft import vv10_mixed
from gpu4pyscf.dft.mixed_precision import MixedPrecision
from gpu4pyscf.lib.multi_gpu import num_devices

water = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

# paracetamol, RDKit ETKDGv3 + MMFF conformer (Angstrom)
paracetamol = '''
C     -3.61957390    -0.76505205     0.45411191
C     -2.17647166    -0.68875405     0.88773105
O     -1.84182055    -1.04492074     2.01150591
N     -1.26945862    -0.13868633    -0.00497117
C      0.12708389     0.01669001     0.15254787
C      0.80014497     1.10398722    -0.41780981
C      2.17938264     1.25799958    -0.29233311
C      2.91091993     0.32286474     0.43757926
O      4.25891208     0.51111223     0.52776961
C      2.25206883    -0.76224096     1.00766218
C      0.86818015    -0.91087225     0.87128879
H     -4.15937393    -1.26299012     1.26558244
H     -3.72053779    -1.33045005    -0.47716418
H     -4.02961045     0.23823223     0.29637124
H     -1.59855082     0.20234542    -0.86099637
H      0.25294934     1.84115627    -0.99640914
H      2.68245632     2.10549939    -0.75191426
H      4.61640898    -0.09648598     1.19863126
H      2.81302339    -1.49467862     1.58023604
H      0.37793711    -1.76083924     1.33609043
'''

XC = 'wb97m-v'
ETOL = 1e-8       # |E_mixed - E_stock|, Hartree
# Two stock runs of the same SCF are not guaranteed to be bit-identical (some
# GPU reductions use atomicAdd), so "unchanged" is asserted to this bound.
SAME = 1e-11


def setUpModule():
    global mol_w, mol_p, mol_o, _stock
    mol_w = pyscf.M(atom=water, basis='def2-tzvpp', verbose=0, output='/dev/null')
    mol_p = pyscf.M(atom=paracetamol, basis='def2-mtzvpp', verbose=0,
                    output='/dev/null')
    mol_o = pyscf.M(atom='O 0 0 0; O 0 0 1.21', basis='def2-svp', spin=2,
                    verbose=0, output='/dev/null')
    _stock = {}


def tearDownModule():
    global mol_w, mol_p, mol_o, _stock
    for m in (mol_w, mol_p, mol_o):
        m.stdout.close()
    del mol_w, mol_p, mol_o, _stock


def make_mf(mol, xc=XC, policy=None, df=True):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = 3
    mf.grids.prune = None
    mf.conv_tol = 1e-10
    if df:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.mixed_precision = policy
    return mf


def run(mol, xc=XC, policy=None, df=True):
    mf = make_mf(mol, xc, policy, df)
    e = mf.kernel()
    assert mf.converged
    return mf, e


def stock(mol, df=True):
    '''Stock SCF (mixed_precision unset), run once per module.'''
    key = (id(mol), df)
    if key not in _stock:
        _stock[key] = run(mol, df=df)
    return _stock[key]


def bits(a):
    return cupy.asarray(a, dtype=np.float64).view(np.int64)


class _Boom(Exception):
    pass


class _NoKernelNumInt(numint.NumInt):
    '''A NumInt override whose nr_nlc_vxc predates the vv10_kernel keyword.'''
    def nr_nlc_vxc(self, mol, grids, xc_code, dms, relativity=0, hermi=1,
                   max_memory=2000, verbose=None):
        return numint.nr_nlc_vxc(self, mol, grids, xc_code, dms, relativity,
                                 hermi, max_memory, verbose)


@unittest.skipIf(num_devices > 1, 'mixed_precision supports a single GPU only')
class KnownValues(unittest.TestCase):
    # -- policy ------------------------------------------------------------------
    def test_policy_positional_and_repr(self):
        p = MixedPrecision(True, True)
        self.assertTrue(p.xc and p.k)
        self.assertFalse(p.vv10)
        self.assertEqual(p.vv10_switch_tol, vv10_mixed.VV10_SWITCH_TOL)
        self.assertEqual(vv10_mixed.VV10_SWITCH_TOL, 1e-5)
        self.assertIn('vv10=False', repr(p))
        self.assertIn('vv10_switch_tol=1e-05', repr(p))

    def test_switch_reason_names_the_traced_energy(self):
        for label, want in (('xc', '|dE_xc|'), ('nlc', '|dE_nlc|')):
            c = mp.PhaseController(1e-5, label=label)
            c.observe(1, -1.0)
            c.observe(2, -1.0 - 1e-6)
            self.assertEqual(c.switch_call, 3)
            self.assertTrue(c.switch_reason.startswith(want + '='), c.switch_reason)
            c = mp.PhaseController(1e-12, stall=1, label=label)
            for i, e in enumerate((0., 1., 3., 6.)):
                c.observe(i + 1, e)
            self.assertTrue(c.switch_reason.startswith(want + ' stalled'), c.switch_reason)
            c = mp.PhaseController(1e-5, label=label)
            c.observe(1, 0.)
            c.observe(2, float('nan'))
            self.assertEqual(c.switch_reason, f'non-finite {want}')
        self.assertTrue(mp.PhaseController(1e-5).switch_reason == '')
        mf = make_mf(mol_w, policy=MixedPrecision(vv10=True))
        state = mp._SCFState(mf, mf.mixed_precision)
        self.assertEqual(state.vv10_ctrl.label, 'nlc')
        self.assertEqual(state.xc_ctrl.label, 'xc')
        self.assertEqual(state.k_ctrl.label, 'xc')

    # -- the kernels ---------------------------------------------------------------
    def test_bind(self):
        bound = vv10_mixed.bind_kernels()
        rec = bound.record
        self.assertEqual(rec['fp32']['status'], 'built')
        self.assertEqual(rec['df64']['status'], 'built')
        self.assertEqual(rec['fp32']['variant'], vv10_mixed.FP32_VARIANT)
        self.assertEqual(rec['df64']['variant'], vv10_mixed.DF64_VARIANT)
        self.assertEqual(rec['fp32']['entry'],
                         vv10_mixed.entry_name(vv10_mixed.FP32_VARIANT))
        self.assertEqual(rec['df64']['entry'],
                         vv10_mixed.df64_entry_name(vv10_mixed.DF64_VARIANT))
        self.assertIs(vv10_mixed.bind_kernels(), bound)     # once per device

    def test_kernels_against_stock_vv10nlc(self):
        mf, _ = stock(mol_p)
        dm = mf.make_rdm1()
        ni = mf._numint
        # the arguments nr_nlc_vxc hands to _vv10nlc
        with mock.patch.object(numint, '_vv10nlc', wraps=numint._vv10nlc) as spy:
            ni.nr_nlc_vxc(mol_p, mf.nlcgrids, XC, dm)
        self.assertEqual(spy.call_count, 1)
        rho, coords, weights, nlc_pars = spy.call_args[0]

        # (d) the keyword seam: None and the stock launcher are bit for bit equal
        kept = []

        def capture(c, rw, om, ka):
            kept.append((c, rw, om, ka))
            return vv10_mixed.uwe_stock(c, rw, om, ka)
        exc0, vxc0 = numint._vv10nlc(rho, coords, weights, nlc_pars)
        exc1, vxc1 = numint._vv10nlc(rho, coords, weights, nlc_pars, uwe_kernel=capture)
        self.assertEqual(len(kept), 1)
        self.assertTrue(bool((bits(exc0) == bits(exc1)).all()))
        self.assertTrue(bool((bits(vxc0) == bits(vxc1)).all()))

        c, rw, om, ka = kept[0]
        self.assertGreater(c.shape[0], 1000)
        ref = vv10_mixed.uwe_stock(c, rw, om, ka)
        bound = vv10_mixed.bind_kernels()

        # (a) the FP64 reference entry is bit-identical to stock
        f64 = bound.fp32.uwe_f64_ref(c, rw, om, ka)
        for r, g in zip(ref, f64):
            self.assertTrue(bool((bits(r) == bits(g)).all()))

        # (b) FP32 and (c) df64, against stock, per point and in energy
        for launch, band_rel, band_denlc in (
                (bound.uwe_fp32, vv10_mixed.VV32_REL, vv10_mixed.VV32_DENLC),
                (bound.uwe_df64, vv10_mixed.VVDF_REL, vv10_mixed.VVDF_DENLC)):
            got = launch(c, rw, om, ka)
            for r, g in zip(ref, got):
                self.assertEqual(g.dtype, np.float64)
                rel = vv10_mixed.max_rel(r, g, cupy)
                self.assertLessEqual(rel, band_rel)
            self.assertGreater(vv10_mixed.max_rel(ref[2], got[2], cupy), 0)
            denlc = abs(float(cupy.dot(rw, 0.5 * (got[2] - ref[2]))))
            self.assertLessEqual(denlc, band_denlc)
            self.assertGreater(denlc, 0)

    # -- SCF against stock -------------------------------------------------------
    def _check_scf(self, mol, df=True):
        mf0, e0 = stock(mol, df)
        mf1, e1 = run(mol, policy=MixedPrecision(vv10=True), df=df)
        self.assertAlmostEqual(e1, e0, delta=ETOL)
        self.assertLessEqual(abs(mf1.cycles - mf0.cycles), 1)
        self.assertIsNone(getattr(mf1, '_mixed_precision_state', None))
        rec = mf1.mixed_precision_record
        self.assertIn('fp32', rec['vv10'])                # the mode really ran
        self.assertEqual(rec['vv10'][-2:], ['df64', 'df64'])
        self.assertNotIn('fp64', rec['vv10'])
        self.assertEqual(rec['vv10_n_fp32'] + rec['vv10_n_df64'], len(rec['vv10']))
        self.assertGreaterEqual(rec['vv10_n_fp32'], 1)
        self.assertEqual(len(rec['vv10_n_masked']), len(rec['vv10']))
        self.assertTrue(all(isinstance(n, int) and n > 0 for n in rec['vv10_n_masked']))
        self.assertTrue(all(p == 'fp64' for p in rec['xc'] + rec['k']))
        self.assertTrue(rec['vv10_switch_reason'].startswith('|dE_nlc|'))
        self.assertEqual(rec['vv10_tol'], 1e-5)
        self.assertEqual(rec['vv10_kernel']['fp32']['status'], 'built')
        self.assertEqual(rec['vv10_kernel']['df64']['status'], 'built')
        cert = rec['vv10_cert']
        self.assertIsNotNone(cert)
        self.assertIs(cert['ok'], True)
        self.assertEqual(cert['call'], len(rec['vv10']))
        for v in cert['rel'].values():
            self.assertLessEqual(v, vv10_mixed.VVDF_REL)
        self.assertLessEqual(cert['denlc'], vv10_mixed.VVDF_DENLC)
        self.assertTrue(math.isfinite(cert['wall_s']) and cert['wall_s'] > 0)
        self.assertIsNone(cert['error'])
        self.assertTrue(rec['fp64_tail'])
        self.assertEqual(rec['tail_precision'],
                         {'xc': 'fp64', 'k': 'fp64', 'vv10': 'df64'})
        return rec

    def test_vv10_water(self):
        self._check_scf(mol_w)

    def test_vv10_paracetamol(self):
        self._check_scf(mol_p)

    def test_vv10_water_without_density_fitting(self):
        # The non-DF rks.get_veff path (the DF path is df_jk._DFHF.get_veff).
        self._check_scf(mol_w, df=False)

    # -- the convergence contract ----------------------------------------------
    def test_convergence_waits_for_df64_tail(self):
        # A threshold the trajectory never meets: the switch can only come
        # from the convergence guard, which fires once the convergence tests
        # are met in FP32, so they are loosened above the FP32 noise floor.
        policy = MixedPrecision(vv10=True, vv10_switch_tol=1e-14, stall=1000,
                                call_cap=1000)
        _, e0 = stock(mol_w)
        mf1 = make_mf(mol_w, policy=policy)
        mf1.conv_tol = 1e-6
        e1 = mf1.kernel()
        self.assertTrue(mf1.converged)
        rec = mf1.mixed_precision_record
        self.assertTrue(rec['forced'])
        self.assertIn('fp32', rec['vv10'])
        self.assertEqual(rec['vv10'][-2:], ['df64', 'df64'])
        self.assertIs(rec['vv10_cert']['ok'], True)
        self.assertTrue(rec['fp64_tail'])
        self.assertAlmostEqual(e1, e0, delta=1e-6)

    def test_certificate_out_of_band_raises(self):
        orig = vv10_mixed._Bound.uwe_df64

        def perturbed(self, coords, rw, om, ka):
            U, W, E = orig(self, coords, rw, om, ka)
            return U, W, E * (1 + 1e-6)
        mf = make_mf(mol_w, policy=MixedPrecision(vv10=True))
        with mock.patch.object(vv10_mixed._Bound, 'uwe_df64', perturbed):
            with self.assertRaises(RuntimeError) as ctx:
                mf.kernel()
        self.assertIn('VV10 certificate out of band', str(ctx.exception))
        self.assertIs(mf.converged, False)
        self.assertIsNone(getattr(mf, '_mixed_precision_state', None))
        cert = mf.mixed_precision_record['vv10_cert']
        self.assertIs(cert['ok'], False)
        self.assertGreater(cert['rel']['E'], vv10_mixed.VVDF_REL)

    def test_scf_that_raises_skips_certificate_and_clears_state(self):
        orig = vv10_mixed._Bound.uwe_fp32
        calls = []

        def failing(self, coords, rw, om, ka):
            # call 2 is always FP32: the controller needs two energies
            calls.append(1)
            if len(calls) == 2:
                raise _Boom('injected')
            return orig(self, coords, rw, om, ka)
        mf = make_mf(mol_w, policy=MixedPrecision(vv10=True))
        with mock.patch.object(vv10_mixed._Bound, 'uwe_fp32', failing):
            with self.assertRaises(_Boom):
                mf.kernel()
        self.assertIsNone(getattr(mf, '_mixed_precision_state', None))
        rec = mf.mixed_precision_record
        self.assertEqual(rec['vv10'], ['fp32'])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(rec['vv10_cert'])

    def test_end_call_without_enlc_fails_closed(self):
        mf = make_mf(mol_w, policy=MixedPrecision(vv10=True))
        state = mp._SCFState(mf, mf.mixed_precision)
        state.begin_call()
        with self.assertRaises(RuntimeError):
            state.end_call(0., None)

    # -- refusals ----------------------------------------------------------------
    def _refused(self, mf):
        with self.assertRaises(NotImplementedError):
            mf.kernel()
        self.assertIsNone(getattr(mf, '_mixed_precision_state', None))

    def test_refuses_vv10_without_nlc(self):
        self._refused(make_mf(mol_w, 'b3lyp', MixedPrecision(vv10=True)))

    def test_refuses_xc_k_on_nlc(self):
        for policy in (MixedPrecision(xc=True), MixedPrecision(k=True),
                       MixedPrecision(xc=True, vv10=True)):
            self._refused(make_mf(mol_w, XC, policy))

    def test_refuses_vv10_uks(self):
        mf = uks.UKS(mol_o, xc=XC)
        mf.mixed_precision = MixedPrecision(vv10=True)
        self._refused(mf)

    def test_refuses_vv10_open_shell_rks(self):
        mf = rks.RKS(mol_o, xc=XC)
        mf.mixed_precision = MixedPrecision(vv10=True)
        self._refused(mf)

    def test_refuses_vv10_numint_without_vv10_kernel(self):
        # refused at begin(), before any kernel is built
        mf = make_mf(mol_w, policy=MixedPrecision(vv10=True))
        mf._numint = _NoKernelNumInt()
        with mock.patch.object(vv10_mixed, 'bind_kernels',
                               side_effect=AssertionError('kernels built')):
            self._refused(mf)
        with self.assertRaises(NotImplementedError):
            mp.check_supported(mf, mf.mixed_precision)
        mf._numint = numint.NumInt()
        mp.check_supported(mf, mf.mixed_precision)

    def test_refuses_xc_range_separated(self):
        self._refused(make_mf(mol_w, 'wb97x', MixedPrecision(xc=True)))

    # -- the default path is untouched ------------------------------------------
    def test_all_off_is_stock(self):
        _, e0 = stock(mol_w)
        mf1, e1 = run(mol_w, policy=MixedPrecision())
        self.assertAlmostEqual(e1, e0, delta=SAME)
        rec = mf1.mixed_precision_record
        self.assertTrue(len(rec['vv10']) > 0)
        self.assertTrue(all(p == 'fp64' for p in rec['vv10'] + rec['xc'] + rec['k']))
        self.assertTrue(all(n is None for n in rec['vv10_n_masked']))
        self.assertIsNone(rec['vv10_cert'])
        self.assertIsNone(rec['vv10_kernel'])
        self.assertEqual(rec['vv10_switch_call'], 0)
        self.assertEqual(rec['vv10_switch_reason'], 'VV10 mixed precision not requested')
        self.assertEqual(rec['tail_precision']['vv10'], 'fp64')
        self.assertIsNone(getattr(mf1, '_mixed_precision_state', None))


if __name__ == "__main__":
    print("Tests for opt-in mixed-precision VV10")
    unittest.main()
