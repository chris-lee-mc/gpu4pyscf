#!/usr/bin/env python3
"""Re-measure one speed cell of the native-port campaign on your own GPU.

    python reproduce_native.py paracetamol r2scan              # mixed then stock, R = 3
    python reproduce_native.py celecoxib b3lyp --arms mixed,cache,stock
    python reproduce_native.py sildenafil b3lyp --basis tzvp
    python reproduce_native.py propranolol wb97m-v --dry-run   # print the plan, import nothing

It builds each SCF exactly as the campaign's pods did (rfcbench_cells.build_and_run in the
gpu-conformer-engine repository), with:
  - basis def2-mTZVPP (or def2-SVP / def2-TZVP with --basis) and aux def2-tzvpp-jkfit, except that
    SVP and TZVP default to def2-universal-jkfit. The basis scan B1 (PREREG-rfcbench-3) used
    def2-universal-jkfit for all three bases: reproduce its mTZVPP column with
    --aux def2-universal-jkfit. Grid level 3 with pruning off;
  - conv_tol 1e-9 re-pinned after density_fit, max_cycle 60, the library-default conv_tol_grad;
  - arms run in the order mixed, cache, stock (stock last), a fresh molecule and SCF object per run,
    timed sync-to-sync around mf.kernel();
  - pair 0 is the cold pair and is reported but excluded, as in the campaign; pairs 1..R are warm;
  - the policies:
        mixed r2SCAN   MixedPrecision(xc=True, ao_cache_fp64=True)
        mixed B3LYP    MixedPrecision(xc=True, k=True, xc_switch_tol=3e-4, k_switch_tol=1e-3,
                                      ao_cache_fp64=True)
        mixed wB97M-V  MixedPrecision(vv10=True, ao_cache_fp64=True)
        cache (any)    MixedPrecision(ao_cache_fp64=True)
        stock          None

The campaign also pinned the Python stack (../requirements.lock) and cuTENSOR (cutensor-cu12==2.3.1
with its lib dir on LD_LIBRARY_PATH), and gated every pod on cuTENSOR being loaded and used by
gpu4pyscf. With cupy 14, libcutensor is loaded only through `cupy_backends.cuda.libs.__getattr__`,
which gpu4pyscf 1.8.1's import form never calls, so this script preloads it the way the pods did,
before gpu4pyscf is imported, and then refuses to time anything unless gpu4pyscf's contract engine
is cuTENSOR and passes a self-test against cupy.einsum (the pods' own gate). Without cuTENSOR,
gpu4pyscf falls back to another engine and the walls are not comparable with the campaign's.

--conv-tol-grad 1e-5 with --repeats 1 reproduces the PREREG-rfcbench-1 bridge leg.

It prints each run, then per-arm warm medians, spreads and the mixed/stock (and mixed/cache,
cache/stock) ratios, defined as in verify_native.py. Downstream (gradient / dipole) cells and the
MIG pods are not reproduced by this script; their procedure is in prereg/.

NOT GPU-EXECUTED in this form: it is a standalone transcription of the campaign's measurement path,
checked here only by --dry-run. The campaign's numbers come from the harness, not from this file.
"""
import argparse
import gc
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.join(HERE, os.pardir, "geometries")

BASES = {"mtzvpp": ("def2-mtzvpp", "def2-tzvpp-jkfit"), "svp": ("def2-svp", "def2-universal-jkfit"),
         "tzvp": ("def2-tzvp", "def2-universal-jkfit")}
AUXES = ("def2-tzvpp-jkfit", "def2-universal-jkfit")
GRID_LEVEL = 3
CONV_TOL = 1e-9
MAX_CYCLE = 60
XCS = ("r2scan", "b3lyp", "wb97m-v")
ARMS = ("mixed", "cache", "stock")


def policy_kwargs(arm, xc):
    if arm == "stock":
        return None
    if arm == "cache":
        return {"ao_cache_fp64": True}
    return {"r2scan": {"xc": True, "ao_cache_fp64": True},
            "b3lyp": {"xc": True, "k": True, "xc_switch_tol": 3e-4, "k_switch_tol": 1e-3,
                      "ao_cache_fp64": True},
            "wb97m-v": {"vv10": True, "ao_cache_fp64": True}}[xc]


def read_xyz(mol):
    path = os.path.join(GEOM, f"{mol}.xyz")
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0])
    atoms = []
    for ln in lines[2:2 + n]:
        s, x, y, z = ln.split()[:4]
        atoms.append((s, (float(x), float(y), float(z))))
    if len(atoms) != n:
        raise ValueError(f"{path}: header says {n} atoms, read {len(atoms)}")
    return atoms


def check_sha(mol):
    import hashlib
    sums = {}
    with open(os.path.join(GEOM, "SHA256SUMS")) as f:
        for ln in f:
            h, name = ln.split()
            sums[name] = h
    if f"{mol}.xyz" not in sums:
        raise SystemExit(f"no committed geometry for {mol!r}; one of: "
                         + ", ".join(sorted(n[:-4] for n in sums)))
    with open(os.path.join(GEOM, f"{mol}.xyz"), "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if sums.get(f"{mol}.xyz") != got:
        raise SystemExit(f"{mol}.xyz does not match geometries/SHA256SUMS")


def cutensor_preload():
    """As rfcbench_cells.cutensor_preload: run BEFORE anything imports gpu4pyscf."""
    import importlib
    try:
        importlib.import_module("cupy_backends.cuda.libs.cutensor")
    except Exception:
        pass
    try:
        getattr(importlib.import_module("cupy_backends.cuda.libs"), "cutensor")
    except Exception as e:
        print(f"cuTENSOR accessor failed: {type(e).__name__}: {e}")


def cutensor_gate():
    """As rfcbench_cells.contract_engine_info + cutensor_problems: cuTENSOR loaded, gpu4pyscf's
    contract engine is cuTENSOR, and its FP64/FP32 self-tests pass. Exits otherwise."""
    import cupy as cp
    from gpu4pyscf.lib import cutensor as ct
    probs = []
    if getattr(ct, "cutensor", None) is None:
        probs.append("cuTENSOR did not load")
    engine = getattr(ct, "contract_engine", None) or "cutensor"
    if engine != "cutensor":
        probs.append(f"contract engine is {engine!r}, not cutensor")
    rng = cp.random.default_rng(7)
    a = rng.standard_normal((6, 5, 4))
    b = rng.standard_normal((3, 5, 7))
    ref = cp.einsum("ijk,ljm->iklm", a, b)
    err64 = float(cp.abs(ct.contract("ijk,ljm->iklm", a, b) - ref).max())
    got32 = ct.contract("ijk,ljm->iklm", a.astype(cp.float32), b.astype(cp.float32))
    rel32 = float(cp.abs(got32 - ref).max() / cp.abs(ref).max())
    if not err64 <= 1e-10:
        probs.append(f"FP64 self-test {err64:.3e} > 1e-10")
    if not rel32 <= 1e-4:
        probs.append(f"FP32 self-test {rel32:.3e} > 1e-4")
    if probs:
        raise SystemExit("refusing to time without cuTENSOR (the campaign required it): "
                         + "; ".join(probs))
    print(f"cuTENSOR: loaded, engine {engine}, self-test fp64 {err64:.1e} fp32 {rel32:.1e}")


def run_once(atoms, xc, basis, aux, arm, conv_tol_grad=None):
    import cupy
    import pyscf
    from gpu4pyscf.dft import rks
    from gpu4pyscf.dft.mixed_precision import MixedPrecision
    bas = BASES[basis][0]
    kw = policy_kwargs(arm, xc)
    mol = pyscf.M(atom=atoms, basis=bas, verbose=0)
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = GRID_LEVEL
    mf.grids.prune = None
    mf = mf.density_fit(auxbasis=aux)
    mf.conv_tol = CONV_TOL                 # re-pinned AFTER density_fit
    mf.max_cycle = MAX_CYCLE
    if conv_tol_grad is not None:
        mf.conv_tol_grad = conv_tol_grad
    mf.mixed_precision = None if kw is None else MixedPrecision(**kw)
    cupy.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    e = mf.kernel()
    cupy.cuda.runtime.deviceSynchronize()
    wall = time.perf_counter() - t0
    rec = getattr(mf, "mixed_precision_record", None) or {}
    out = {"wall_s": wall, "e": float(e), "cycles": int(mf.cycles), "converged": bool(mf.converged),
           "tier": rec.get("ao_cache_tier"), "note": (rec.get("ao_cache") or "")[:120]}
    del mf
    gc.collect()
    return out


def summarise(runs, arms):
    warm = {a: [r["wall_s"] for p, a2, r in runs if a2 == a and p > 0] for a in arms}
    med = {a: statistics.median(w) for a, w in warm.items() if w}
    for a in arms:
        w = warm[a]
        sp = (max(w) - min(w)) / med[a]
        print(f"  {a:6s} warm median {med[a]:9.3f} s   spread {100 * sp:5.1f} %"
              f"{'   NOISY' if sp > 0.10 else ''}")
    for a, b in (("mixed", "stock"), ("mixed", "cache"), ("cache", "stock")):
        if a in med and b in med:
            print(f"  {a}/{b} = {med[b] / med[a]:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("molecule")
    ap.add_argument("xc", choices=XCS)
    ap.add_argument("--basis", choices=sorted(BASES), default="mtzvpp")
    ap.add_argument("--aux", choices=AUXES, default=None,
                    help="auxiliary basis (default: def2-tzvpp-jkfit for mTZVPP, universal otherwise)")
    ap.add_argument("--arms", default="mixed,stock")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--conv-tol-grad", type=float, default=None,
                    help="default: the library's; 1e-5 with --repeats 1 is the PREREG-1 bridge leg")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    asked = a.arms.split(",")
    if not asked or set(asked) - set(ARMS):
        raise SystemExit(f"--arms must be a subset of {ARMS}")
    arms = [x for x in ARMS if x in asked]          # always run in campaign order, stock last
    aux = a.aux or BASES[a.basis][1]
    if a.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    check_sha(a.molecule)
    atoms = read_xyz(a.molecule)
    print(f"{a.molecule} ({len(atoms)} atoms) {a.xc} {BASES[a.basis][0]} / {aux}, "
          f"grid {GRID_LEVEL} unpruned, conv_tol {CONV_TOL:g}, pairs 0..{a.repeats} (0 = cold)")
    for arm in arms:
        print(f"  {arm:6s} policy {policy_kwargs(arm, a.xc)}")
    if a.conv_tol_grad is not None:
        print(f"  conv_tol_grad {a.conv_tol_grad:g}")
    if a.dry_run:
        return 0
    cutensor_preload()
    cutensor_gate()
    runs = []
    for p in range(a.repeats + 1):
        for arm in arms:
            r = run_once(atoms, a.xc, a.basis, aux, arm, a.conv_tol_grad)
            runs.append((p, arm, r))
            print(f"pair {p} {arm:6s} {r['wall_s']:9.3f} s  E={r['e']:.10f}  cycles={r['cycles']}"
                  f"  conv={r['converged']}  tier={r['tier']}")
            if not r["converged"]:
                raise SystemExit(f"{arm} did not converge: not a valid timing")
    summarise(runs, arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
