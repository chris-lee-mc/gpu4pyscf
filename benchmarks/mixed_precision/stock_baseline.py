"""Stock GPU4PySCF DF-RKS reference timings for the molecules in this bundle.

Standalone: needs only gpu4pyscf (+ its pyscf/cupy) and the XYZ files beside this script. It builds
exactly the stock calculation the banked measurements compared against, so a maintainer can check
the reference side on their own card before reading any of our ratios:

    gpu4pyscf.dft.RKS(mol, xc).density_fit(auxbasis="def2-tzvpp-jkfit")
    basis def2-mTZVPP, grids.level=3, grids.prune=None, conv_tol=1e-9, max_cycle=60,
    conv_tol_grad: library default (~3.162e-5 at conv_tol=1e-9) unless --conv-tol-grad is given.
    The r2SCAN ladder tables used --conv-tol-grad 1e-5; the B3LYP tables used the default.

Timing: one cold call (discarded; JIT and allocator warm-up), then one warm call that builds a new
Mole and a new object and runs the full `kernel()` including the density-fitting build, bracketed by
device synchronisation. This is the whole-SCF wall of one energy call. Our harness additionally
split that wall into phases, so expect agreement with the bundle's `stock` columns in magnitude on the
same card and pins, not to the millisecond.

Usage:
    python stock_baseline.py --xc r2scan paracetamol propranolol celecoxib
    python stock_baseline.py --xc b3lyp --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.join(HERE, "geometries")


def read_xyz(name):
    with open(os.path.join(GEOM, f"{name}.xyz")) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for ln in lines[2:2 + n]:
        sym, x, y, z = ln.split()[:4]
        atoms.append([sym, (float(x), float(y), float(z))])
    return atoms


def build(atoms, xc, conv_tol_grad):
    from pyscf import gto
    import gpu4pyscf.dft as gdft
    mol = gto.M(atom=atoms, basis="def2-mtzvpp", unit="Angstrom", verbose=0)
    mf = gdft.RKS(mol, xc=xc)
    mf.grids.prune = None
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.max_cycle = 60
    if conv_tol_grad is not None:
        mf.conv_tol_grad = conv_tol_grad
    mf = mf.density_fit(auxbasis="def2-tzvpp-jkfit")
    # density_fit() returns a new object: re-pin rather than trust the copy.
    mf.conv_tol = 1e-9
    mf.max_cycle = 60
    if conv_tol_grad is not None:
        mf.conv_tol_grad = conv_tol_grad
    return mf


def run_once(atoms, xc, conv_tol_grad):
    import cupy
    cycles = {"n": 0}

    def cb(envs):
        cycles["n"] = envs.get("cycle", -1) + 1
    cupy.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    mf = build(atoms, xc, conv_tol_grad)
    mf.callback = cb
    e = mf.kernel()
    cupy.cuda.runtime.deviceSynchronize()
    return {"wall_s": time.perf_counter() - t0, "e_tot": float(e),
            "converged": bool(mf.converged), "n_cycle": cycles["n"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("molecules", nargs="*")
    ap.add_argument("--all", action="store_true", help="every XYZ in geometries/")
    ap.add_argument("--xc", default="r2scan")
    ap.add_argument("--conv-tol-grad", type=float, default=None)
    a = ap.parse_args(argv)
    names = sorted(f[:-4] for f in os.listdir(GEOM) if f.endswith(".xyz")) if a.all else a.molecules
    if not names:
        ap.error("name molecules or pass --all")
    import cupy
    import gpu4pyscf
    import pyscf
    dev = cupy.cuda.runtime.getDeviceProperties(0)["name"]
    dev = dev.decode() if isinstance(dev, bytes) else dev
    print(json.dumps({"gpu": dev, "gpu4pyscf": gpu4pyscf.__version__, "pyscf": pyscf.__version__,
                      "cupy": cupy.__version__, "xc": a.xc, "conv_tol_grad": a.conv_tol_grad}))
    for name in names:
        atoms = read_xyz(name)
        run_once(atoms, a.xc, a.conv_tol_grad)                 # cold, discarded
        r = run_once(atoms, a.xc, a.conv_tol_grad)
        print(json.dumps({"molecule": name, **r}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
