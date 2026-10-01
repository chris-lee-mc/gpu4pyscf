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

import unittest
import numpy as np
import cupy
import pyscf
from pyscf import dft as cpu_dft
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.dft import mixed_precision as mp
from gpu4pyscf.dft.mixed_precision import MixedPrecision

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

ETOL = 1e-8       # |E_mixed - E_stock|, Hartree
# Two stock runs of the same SCF are not guaranteed to be bit-identical (some
# GPU reductions use atomicAdd), so "unchanged" is asserted to this bound.
SAME = 1e-11


def setUpModule():
    global mol_w, mol_p, mol_o
    mol_w = pyscf.M(atom=water, basis='def2-tzvpp', verbose=0, output='/dev/null')
    mol_p = pyscf.M(atom=paracetamol, basis='def2-mtzvpp', verbose=0,
                    output='/dev/null')
    mol_o = pyscf.M(atom='O 0 0 0; O 0 0 1.21', basis='def2-svp', spin=2,
                    verbose=0, output='/dev/null')


def tearDownModule():
    global mol_w, mol_p, mol_o
    for m in (mol_w, mol_p, mol_o):
        m.stdout.close()
    del mol_w, mol_p, mol_o


def make_mf(mol, xc, policy=None, df=True):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = 3
    mf.grids.prune = None
    mf.conv_tol = 1e-10
    if df:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.mixed_precision = policy
    return mf


def run(mol, xc, policy=None, df=True):
    mf = make_mf(mol, xc, policy, df)
    e = mf.kernel()
    assert mf.converged
    return mf, e


class KnownValues(unittest.TestCase):
    # -- the default path is untouched ------------------------------------------
    def test_default_is_none(self):
        self.assertIsNone(rks.RKS(mol_w, xc='b3lyp').mixed_precision)

    def test_both_components_off_is_stock(self):
        mf0, e0 = run(mol_w, 'b3lyp')
        mf1, e1 = run(mol_w, 'b3lyp', MixedPrecision(xc=False, k=False))
        self.assertAlmostEqual(e0, e1, delta=SAME)
        self.assertEqual(mf0.cycles, mf1.cycles)
        rec = mf1.mixed_precision_record
        self.assertTrue(all(p == 'fp64' for p in rec['xc'] + rec['k']))

    # -- enabled against stock -------------------------------------------------
    def _check_mixed(self, mol, xc, policy, expect_k32):
        mf0, e0 = run(mol, xc)
        mf1, e1 = run(mol, xc, policy)
        self.assertAlmostEqual(e1, e0, delta=ETOL)
        self.assertLessEqual(abs(mf1.cycles - mf0.cycles), 1)
        rec = mf1.mixed_precision_record
        self.assertIn('fp32', rec['xc'])               # the mode really ran
        self.assertEqual(rec['xc'][-2:], ['fp64', 'fp64'])
        self.assertTrue(rec['fp64_tail'])
        if expect_k32:
            self.assertIn('fp32', rec['k'])
            self.assertEqual(rec['k'][-2:], ['fp64', 'fp64'])
            self.assertIsNotNone(rec['k_full_rebuild_call'])
        dm0, dm1 = mf0.make_rdm1(), mf1.make_rdm1()
        self.assertLess(float(abs(dm1 - dm0).max()), 1e-5)
        return rec

    def test_r2scan_xc_water(self):
        self._check_mixed(mol_w, 'r2scan', MixedPrecision(xc=True), False)

    def test_r2scan_xc_water_without_density_fitting(self):
        # The non-DF rks.get_veff path (the DF path is df_jk._DFHF.get_veff).
        mf0, e0 = run(mol_w, 'r2scan', df=False)
        mf1, e1 = run(mol_w, 'r2scan', MixedPrecision(xc=True), df=False)
        self.assertAlmostEqual(e1, e0, delta=ETOL)
        rec = mf1.mixed_precision_record
        self.assertIn('fp32', rec['xc'])
        self.assertTrue(rec['fp64_tail'])

    def test_r2scan_xc_paracetamol(self):
        self._check_mixed(mol_p, 'r2scan', MixedPrecision(xc=True), False)

    def test_pbe_xc_paracetamol(self):
        self._check_mixed(mol_p, 'pbe', MixedPrecision(xc=True), False)

    def test_b3lyp_xc_k_paracetamol(self):
        self._check_mixed(mol_p, 'b3lyp',
                          MixedPrecision(xc=True, k=True, xc_switch_tol=3e-4,
                                         k_switch_tol=1e-3), True)

    def test_k_only_paracetamol(self):
        self._check_mixed_k_only()

    def _check_mixed_k_only(self):
        mf0, e0 = run(mol_p, 'b3lyp')
        mf1, e1 = run(mol_p, 'b3lyp', MixedPrecision(xc=False, k=True))
        self.assertAlmostEqual(e1, e0, delta=ETOL)
        rec = mf1.mixed_precision_record
        self.assertIn('fp32', rec['k'])
        self.assertTrue(all(p == 'fp64' for p in rec['xc']))
        self.assertTrue(rec['fp64_tail'])

    # -- enabled against CPU PySCF ---------------------------------------------
    def test_against_cpu_pyscf(self):
        mf_cpu = cpu_dft.RKS(mol_w, xc='b3lyp')
        mf_cpu.grids.level = 3
        mf_cpu.grids.prune = None
        mf_cpu.conv_tol = 1e-10
        mf_cpu = mf_cpu.density_fit(auxbasis='def2-universal-jkfit')
        e_cpu = mf_cpu.kernel()
        _, e_stock = run(mol_w, 'b3lyp')
        _, e_mixed = run(mol_w, 'b3lyp', MixedPrecision(xc=True, k=True))
        self.assertAlmostEqual(e_mixed, e_cpu, delta=1e-6)
        self.assertLessEqual(abs(e_mixed - e_cpu), abs(e_stock - e_cpu) + ETOL)

    # -- the convergence contract ----------------------------------------------
    def test_convergence_waits_for_fp64_tail(self):
        # Thresholds the trajectory never meets: the switch can only come
        # from the convergence guard.  The guard fires only once the SCF
        # convergence tests are met in FP32, so they are loosened here to lie
        # above the FP32 noise floor; at conv_tol=1e-10 an FP32 trajectory may
        # never meet them, and stall/call_cap are the backstops for that case.
        policy = MixedPrecision(xc=True, k=True, xc_switch_tol=1e-14,
                                k_switch_tol=1e-14, stall=1000, call_cap=1000)
        mf0, e0 = run(mol_p, 'b3lyp')
        mf1 = make_mf(mol_p, 'b3lyp', policy)
        mf1.conv_tol = 1e-6
        e1 = mf1.kernel()
        self.assertTrue(mf1.converged)
        rec = mf1.mixed_precision_record
        self.assertTrue(rec['forced'])
        self.assertEqual(rec['xc'][-2:], ['fp64', 'fp64'])
        self.assertEqual(rec['k'][-2:], ['fp64', 'fp64'])
        self.assertIsNotNone(rec['k_full_rebuild_call'])
        self.assertTrue(rec['fp64_tail'])
        # E is second order in the density error, so an FP64 tail at the
        # loosened tolerance still lands well within 1e-6 of the tight stock E.
        self.assertAlmostEqual(e1, e0, delta=1e-6)

    # -- refusals ----------------------------------------------------------------
    def test_refuses_uks(self):
        mf = uks.UKS(mol_o, xc='pbe')
        mf.mixed_precision = MixedPrecision(xc=True)
        with self.assertRaises(NotImplementedError):
            mf.kernel()

    def test_refuses_open_shell_rks(self):
        mf = rks.RKS(mol_o, xc='pbe')
        mf.mixed_precision = MixedPrecision(xc=True)
        with self.assertRaises(NotImplementedError):
            mf.kernel()

    def test_refuses_range_separated(self):
        mf = make_mf(mol_w, 'wb97x', MixedPrecision(xc=True))
        with self.assertRaises(NotImplementedError):
            mf.kernel()

    def test_refuses_nlc(self):
        mf = make_mf(mol_w, 'wb97m-v', MixedPrecision(xc=True))
        with self.assertRaises(NotImplementedError):
            mf.kernel()

    def test_refuses_k_without_density_fitting(self):
        mf = make_mf(mol_w, 'b3lyp', MixedPrecision(k=True), df=False)
        with self.assertRaises(NotImplementedError):
            mf.kernel()

    def test_refuses_non_policy(self):
        mf = make_mf(mol_w, 'pbe', {'xc': True})
        with self.assertRaises(TypeError):
            mf.kernel()

    # -- memory and lifecycle ---------------------------------------------------
    def test_ao_copy_that_does_not_fit_falls_back_to_stock(self):
        mf0, e0 = run(mol_w, 'pbe')
        mf1, e1 = run(mol_w, 'pbe', MixedPrecision(xc=True, ao_cache_mem_fraction=1e-12))
        rec = mf1.mixed_precision_record
        self.assertTrue(rec['ao_cache'].startswith('did not fit'))
        self.assertTrue(all(p == 'fp64' for p in rec['xc']))
        self.assertAlmostEqual(e1, e0, delta=SAME)

    def test_scanner_geometry_change(self):
        mf = make_mf(mol_w, 'pbe', MixedPrecision(xc=True))
        scanner = mf.as_scanner()
        mol2 = mol_w.set_geom_('''
            O  0.0  0.0  0.1274
            H -0.7670 0.0 -0.4796
            H  0.7670 0.0 -0.4796''', inplace=False)
        for mol in (mol_w, mol2, mol_w):
            e_mixed = scanner(mol)
            _, e_ref = run(mol, 'pbe')
            self.assertAlmostEqual(e_mixed, e_ref, delta=ETOL)
            self.assertTrue(scanner.mixed_precision_record['fp64_tail'])

    def test_state_is_cleared_after_kernel(self):
        mf, _ = run(mol_w, 'pbe', MixedPrecision(xc=True))
        self.assertIsNone(getattr(mf, '_mixed_precision_state', None))

    # -- the FP32 kernels against their FP64 counterparts -------------------------
    def test_nr_rks_fp32_matches_stock_nr_rks(self):
        for xc in ('lda', 'pbe', 'r2scan'):
            mf, _ = run(mol_p, xc)
            dm = mf.make_rdm1()
            ni = mf._numint
            n0, e0, v0 = ni.nr_rks(mol_p, mf.grids, xc, dm)
            mf.mixed_precision = MixedPrecision(xc=True)
            state = mp._SCFState(mf, mf.mixed_precision)
            # tagged density (the SCF iterations) and plain density (the guess)
            for d in (dm, cupy.asarray(dm.get())):     # .get(): drops the tags
                n1, e1, v1 = mp.nr_rks_fp32(state, ni, mol_p, mf.grids, xc, d)
                self.assertAlmostEqual(n1, n0, delta=1e-4)
                self.assertAlmostEqual(e1, e0, delta=1e-4)
                self.assertLess(float(abs(v1 - v0).max()), 1e-4)
                self.assertGreater(float(abs(v1 - v0).max()), 0)   # really FP32

    def test_k_block_fp32_chunked(self):
        from gpu4pyscf.df import df_jk
        from gpu4pyscf.lib.cupy_helper import contract
        rng = cupy.random.default_rng(3)
        nL, nao, nocc, n = 37, 23, 7, 2
        cderi = rng.standard_normal((nL, nao, nao))
        cderi = cderi + cderi.transpose(0, 2, 1)
        factor = rng.standard_normal((n, nao, nocc))
        rhok = contract('Lij,njk->nikL', cderi, factor)
        ref = contract('nikL,njkL->nij', rhok, rhok)
        for frac in (0.25, 1e-15):      # whole block, and forced 1-row chunks
            buf = cupy.empty((n, nL * nao * nocc))
            out = cupy.zeros((n, nao, nao))
            df_jk._k_block_fp32(cderi, factor, buf, out, chunk_mem_fraction=frac)
            self.assertEqual(out.dtype, np.float64)
            err = float(abs(out - ref).max()) / float(abs(ref).max())
            self.assertLess(err, 1e-5)
            self.assertGreater(err, 0)


class AOCache(unittest.TestCase):
    '''The FP64 AO cache: the cached FP64 XC path against stock numint.nr_rks,
    the cache tiers, and invalidation.'''
    def _stock_and_state(self, mol, xc, policy=None):
        mf, _ = run(mol, xc)
        dm = mf.make_rdm1()
        mf.mixed_precision = policy or MixedPrecision(xc=True)
        return mf, dm, mf._numint, mp._SCFState(mf, mf.mixed_precision)

    @staticmethod
    def _same_bits(a, b):
        a, b = cupy.asnumpy(a), cupy.asnumpy(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

    def test_fp64_cached_is_bitwise_stock_nr_rks(self):
        for xc in ('svwn', 'pbe', 'r2scan'):
            mf, dm, ni, state = self._stock_and_state(mol_p, xc)
            # tagged density (the SCF iterations) and plain density (the guess)
            for d in (dm, cupy.asarray(dm.get())):     # .get(): drops the tags
                n0, e0, v0 = ni.nr_rks(mol_p, mf.grids, xc, d)
                n0b, e0b, v0b = ni.nr_rks(mol_p, mf.grids, xc, d)
                # the premise: stock is bitwise reproducible
                self.assertTrue(n0 == n0b and e0 == e0b and self._same_bits(v0, v0b), xc)
                res = mp.nr_rks_fp64_cached(state, ni, mol_p, mf.grids, xc, d)
                self.assertIsNotNone(res, (xc, state.ao_cache_note))
                n1, e1, v1 = res
                self.assertEqual(state.ao_cache_tier, 'fp64', xc)
                self.assertEqual(type(n1), type(n0))
                self.assertEqual(n1, n0, xc)
                self.assertEqual(e1, e0, xc)
                self.assertTrue(self._same_bits(v1, v0), xc)
            # negative control: the path reads the cache, and the comparison
            # sees a one-ulp change in one cached AO value
            flat = state.ao_cache.blocks[0][0].ravel()     # a view of the cached block
            i = int(abs(flat).argmax())
            flat[i] = np.nextafter(float(flat[i]), np.inf)
            n2, e2, v2 = mp.nr_rks_fp64_cached(state, ni, mol_p, mf.grids, xc, dm)
            self.assertFalse(self._same_bits(v2, v0), xc)

    def _on_off(self, mol, xc, policy_kw):
        mf0, e0 = run(mol, xc, MixedPrecision(ao_cache_fp64=False, **policy_kw))
        mf1, e1 = run(mol, xc, MixedPrecision(ao_cache_fp64=True, **policy_kw))
        r0, r1 = mf0.mixed_precision_record, mf1.mixed_precision_record
        self.assertAlmostEqual(e1, e0, delta=1e-10)
        self.assertEqual(mf1.cycles, mf0.cycles)
        self.assertEqual(r1['xc'], r0['xc'])
        self.assertEqual(r1['k'], r0['k'])
        self.assertEqual(r0['xc_fp64_cached'], 0)
        return r0, r1

    def test_cache_on_off_r2scan_and_b3lyp(self):
        for xc, kw in (('r2scan', {'xc': True}),
                       ('b3lyp', {'xc': True, 'k': True, 'xc_switch_tol': 3e-4})):
            r0, r1 = self._on_off(mol_w, xc, kw)
            self.assertEqual(r0['ao_cache_tier'], 'fp32')
            self.assertEqual(r1['ao_cache_tier'], 'fp64+fp32')
            self.assertIn('fp32', r1['xc'])
            self.assertGreaterEqual(r1['xc_fp64_cached'], 2)
            self.assertEqual(r1['xc_fp64_stock'], 0)
            self.assertEqual(r1['xc_path'][-2:], ['fp64-cached', 'fp64-cached'])
            self.assertEqual(r1['ao_cache_mirror_released_call'], r1['xc_switch_call'])
            self.assertEqual(r1['ao_cache_bytes64'], 2 * r1['ao_cache_bytes32'])
            self.assertGreater(r1['ao_cache_bytes32'], 0)

    def test_cache_vv10_only_policy(self):
        r0, r1 = self._on_off(mol_w, 'wb97m-v', {'vv10': True})
        self.assertIsNone(r0['ao_cache_tier'])
        self.assertEqual(r1['ao_cache_tier'], 'fp64')
        self.assertEqual(r1['xc_path'], ['fp64-cached'] * len(r1['xc']))
        self.assertEqual(r1['ao_cache_bytes32'], 0)
        self.assertIsNotNone(r1['vv10_cert'])

    def _patched_free(self, free, xc='r2scan'):
        orig = mp._free_device_bytes
        mp._free_device_bytes = lambda: free
        try:
            return run(mol_w, xc, MixedPrecision(xc=True))
        finally:
            mp._free_device_bytes = orig

    def test_tier_fp32_when_fp64_does_not_fit(self):
        mf, _ = run(mol_w, 'r2scan', MixedPrecision(xc=True))
        nvals = mf.mixed_precision_record['ao_cache_bytes32'] // 4
        self.assertGreater(nvals, 0)
        # budget = 0.7 * free = 8 bytes per value: the FP32 copy fits, 12 B do not
        mf1, e1 = self._patched_free(8 * nvals / AO_FRACTION)
        mf0, e0 = run(mol_w, 'r2scan', MixedPrecision(xc=True, ao_cache_fp64=False))
        r1 = mf1.mixed_precision_record
        self.assertEqual(r1['ao_cache_tier'], 'fp32', r1['ao_cache'])
        self.assertIn('no FP64 copy', r1['ao_cache'])
        self.assertIn('fp32', r1['xc'])
        self.assertEqual(r1['xc_fp64_cached'], 0)
        self.assertEqual(r1['xc_fp64_stock'], r1['xc'].count('fp64'))
        self.assertEqual(r1['xc'], mf0.mixed_precision_record['xc'])
        self.assertAlmostEqual(e1, e0, delta=1e-10)

    def test_tier_none_when_nothing_fits(self):
        mf1, e1 = self._patched_free(1.)
        mf0, e0 = run(mol_w, 'r2scan')
        r1 = mf1.mixed_precision_record
        self.assertIsNone(r1['ao_cache_tier'])
        self.assertTrue(r1['ao_cache'].startswith('did not fit'), r1['ao_cache'])
        self.assertTrue(all(p == 'fp64' for p in r1['xc']))
        self.assertEqual(r1['xc_path'], ['fp64-stock'] * len(r1['xc']))
        self.assertAlmostEqual(e1, e0, delta=SAME)

    def test_cache_declines_other_grids(self):
        import copy
        mf, dm, ni, state = self._stock_and_state(mol_w, 'pbe')
        self.assertIsNotNone(mp.nr_rks_fp64_cached(state, ni, mol_w, mf.grids, 'pbe', dm))
        g2 = copy.copy(mf.grids)
        g2.coords = mf.grids.coords.copy()
        self.assertIsNone(mp.nr_rks_fp64_cached(state, ni, mol_w, g2, 'pbe', dm))
        self.assertIsNone(state.ao_cache)
        self.assertIn('dropped', state.ao_cache_note)
        self.assertIn('grids.coords', state.ao_cache_note)
        # dropped for the rest of the SCF, never rebuilt
        self.assertIsNone(mp.nr_rks_fp64_cached(state, ni, mol_w, mf.grids, 'pbe', dm))

    def test_cache_declines_empty_blocks(self):
        mf, dm, ni, state = self._stock_and_state(mol_w, 'pbe')
        self.assertIsNotNone(mp.nr_rks_fp64_cached(state, ni, mol_w, mf.grids, 'pbe', dm))
        state.ao_cache.n_empty = 1
        self.assertIsNone(mp.nr_rks_fp64_cached(state, ni, mol_w, mf.grids, 'pbe', dm))
        self.assertIsNone(state.ao_cache)
        self.assertIn('empty grid blocks', state.ao_cache_note)


AO_FRACTION = mp.AO_CACHE_MEM_FRACTION


if __name__ == "__main__":
    print("Tests for opt-in mixed-precision DF-RKS")
    unittest.main()
