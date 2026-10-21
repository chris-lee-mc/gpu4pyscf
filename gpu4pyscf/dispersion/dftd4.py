# Copyright 2024-2026 The PySCF Developers. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import ctypes

import pyscf.dispersion.dftd4 as _d4_mod
from pyscf import dispersion
from pyscf.dispersion.dftd4 import *

_version_major, _version_minor = dispersion.__version__.split('.')[:2]
if int(_version_major) == 1 and int(_version_minor) < 4:
    # Override DFTD4Dispersion class for compatibility with older versions
    class DFTD4Dispersion(DFTD4Dispersion):
        def __init__(self, mol, xc, version='d4', ga=None, gc=None, wf=None, atm=False):
            if version != 'd4':
                raise RuntimeError(
                    f'pyscf-dispersion {dispersion.__version__} does not support '
                    f'dftd4 code {version}. It is available in pyscf-dispersion>=1.5')
            super().__init__(mol, xc, ga, gc, wf, atm)

# Patch DFTD4Dispersion to fix upstream pyscf-dispersion bugs:
# 1. 'r2scan-3c' uses customized damping parameters set via set_param();
#    initialize underlying object with a supported functional ('pbe') if not recognized.
# 2. In set_param(), libdftd4.dftd4_new_rational_damping requires the error handle pointer
#    as its first C argument; omitting it returns a NULL pointer and causes
#    "RuntimeError: Damping parameters are missing".
_orig_init = _d4_mod.DFTD4Dispersion.__init__


def _init_with_r2scan_fix(self, mol, xc, *args, **kwargs):
    init_xc = 'pbe' if str(xc).lower() == 'r2scan-3c' else xc
    _orig_init(self, mol, init_xc, *args, **kwargs)


def _patched_set_param(self, s8, a1, a2, s6=1.0, s9=1.0, alp=16.0):
    err = _d4_mod.libdftd4.dftd4_new_error()
    self._param = _d4_mod.libdftd4.dftd4_new_rational_damping(
        err,
        ctypes.c_double(s6),
        ctypes.c_double(s8),
        ctypes.c_double(s9),
        ctypes.c_double(a1),
        ctypes.c_double(a2),
        ctypes.c_double(alp),
    )
    _d4_mod.error_check(err)
    _d4_mod.libdftd4.dftd4_delete_error(ctypes.byref(err))


_d4_mod.DFTD4Dispersion.__init__ = _init_with_r2scan_fix
_d4_mod.DFTD4Dispersion.set_param = _patched_set_param

