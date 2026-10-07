#!/usr/bin/env python3
"""Re-derive every number CLAIMS.md quotes from the native-port CSVs in data/, and check it.

Stdlib only. Each check recomputes a value from the CSV rows and compares it with the number as
printed in the RESULT sections of prereg/PREREG-rfcbench-*.md, at that number's printed precision
(decimal places for fixed notation, significant figures for scientific notation). Exit status is
non-zero on any mismatch, on a missing CSV, and on a check whose selection is empty: an empty
selection is never a pass.

Definitions (prereg/PREREG-rfcbench-0-measurand.md, sections 3-4; the extractor that wrote the
RESULTs uses the same):
  - a cell's wall for an arm is the MEDIAN over the warm pairs 1..R (pair 0, the cold pair, is
    excluded; a bridge cell has the single pair 1);
  - ratio "A/B" of a cell = median(B) / median(A), so mixed/stock > 1 means mixed is faster;
  - an aggregate is the geometric mean of per-cell ratios;
  - spread of an arm = (max - min) / median over the warm pairs; a cell is NOISY when either arm's
    spread exceeds 10 %; a pod is DEGRADED when more than 20 % of its cells are NOISY;
  - MIG gain of a cell and arm = 4 x median on the whole card / median on one 1g.24gb instance.
"""
import csv
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R = 3            # warm pairs per cell in every pod of the campaign (a bridge cell has pair 1 only)

results = []  # (claim, check, recomputed, banked, ok)
_cache = {}

# Numbers printed in a RESULT that disagree with the data at their printed precision. They are NOT
# adjusted to pass: each is reported as DISCREPANCY with its reason, CLAIMS.md quotes the
# recomputed value, and the script fails if one of them ever starts matching (so this list cannot
# go stale silently).
# Each entry: (claim, check) -> (the value the data gives at the printed precision, reason). A check
# listed here passes as DISC only if it recomputes to exactly that value: any other value, and any
# error, is a FAIL, so a listed number is still guarded.
KNOWN_DISCREPANCIES = {
    ("N2", "L12 trio B3LYP (healthy PRO 6000 row)"):
        ("1.860", "PREREG-2 part 2 and PREREG-5 printed 1.859; the value is 1.8595"),
    ("N6", "MIG gain atorvastatin b3lyp stock"):
        ("1.10", "PREREG-6 RESULT printed 1.09 (twice); the value is 1.0950"),
    # PREREG-7's RESULT computed these from the pod's 2-decimal readings, not from the walls.
    ("N8", "CS1 P1_stock_first: wb97m-v cold extra (s)"):
        ("0.65", "PREREG-7 RESULT printed 0.64 (difference of rounded walls); the value is 0.646"),
    ("N8", "CS1 P2_mixed_first: wb97m-v cold extra (s)"):
        ("1.12", "PREREG-7 RESULT printed 1.13 (difference of rounded walls); the value is 1.124"),
    ("N8", "CS1 P4_mixed_wiped_cupy: r2scan cold extra (s)"):
        ("9.93", "PREREG-7 RESULT printed 9.92 (difference of rounded walls); the value is 9.929"),
    ("N8", "CS1 P4_mixed_wiped_cupy: b3lyp cold extra (s)"):
        ("1.60", "PREREG-7 RESULT printed 1.61 (difference of rounded walls); the value is 1.604"),
    ("N8", "X3r - X3, b3lyp mixed/stock"):
        ("0.115", "PREREG-7 RESULT printed 0.116 (difference of rounded geomeans); still > 0.10"),
    ("N8", "X3r - X3, r2scan mixed/cache"):
        ("-0.001", "PREREG-7 RESULT printed 0.000 (difference of rounded geomeans)"),
}


def rows(name):
    if name not in _cache:
        path = os.path.join(DATA, name + ".csv")
        with open(path, newline="") as fh:
            _cache[name] = list(csv.DictReader(fh))
    return _cache[name]


def walls(pods, key, arm):
    """Warm walls of one cell and arm. `pods` is a CSV name or a tuple of names; the cell must be
    found in exactly one of them."""
    pods = (pods,) if isinstance(pods, str) else tuple(pods)
    found = [p for p in pods if any(r["key"] == key for r in rows(p))]
    if len(found) != 1:
        raise LookupError(f"{key} found in {found} of {pods}")
    rs = [r for r in rows(found[0]) if r["key"] == key and r["arm"] == arm and r["pair"] not in ("", "0")]
    if not rs:
        raise LookupError(f"no warm {arm} rows for {key} in {found[0]}")
    want = {1} if key.startswith("bridge/") else set(range(1, R + 1))
    pairs = sorted(int(r["pair"]) for r in rs)
    if set(pairs) != want or len(pairs) != len(want):
        raise ValueError(f"{found[0]} {key} {arm}: warm pairs {pairs}, expected {sorted(want)}")
    for r in rs:
        if r["error"] or r["converged"] != "True" or r["wall_s"] in ("", None):
            raise ValueError(f"{found[0]} {key} {arm} pair {r['pair']}: error={r['error']!r} "
                             f"converged={r['converged']!r}")
    return [float(r["wall_s"]) for r in rs]


def med(pods, key, arm):
    return statistics.median(walls(pods, key, arm))


def ratio(pods, key, a, b):
    return med(pods, key, b) / med(pods, key, a)


def spread(pods, key, arm):
    w = walls(pods, key, arm)
    return (max(w) - min(w)) / statistics.median(w)


def geomean(xs):
    xs = list(xs)
    if not xs:
        raise ValueError("geomean of an empty selection")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def keys(pods, kind="speed", xc=None, basis="mtzvpp", mols=None):
    pods = (pods,) if isinstance(pods, str) else tuple(pods)
    out = []
    for p in pods:
        for r in rows(p):
            k = r["key"]
            parts = k.split("/")
            if parts[0] != kind or (xc and parts[2] != xc) or (basis and parts[3] != basis):
                continue
            if mols is not None and parts[1] not in mols:
                continue
            if k not in out:
                out.append(k)
    if not out:
        raise LookupError(f"no {kind} cells for xc={xc} basis={basis} in {pods}")
    return out


def cell(mol, xc, basis="mtzvpp", kind="speed"):
    return f"{kind}/{mol}/{xc}/{basis}"


def fmt_like(value, banked):
    b = banked.strip()
    if "e" in b.lower():
        mant = b.lower().split("e")[0]
        digits = len(mant.replace("-", "").replace(".", "").lstrip("0")) or 1
        return "%.*e" % (digits - 1, value)
    dec = len(b.split(".")[1]) if "." in b else 0
    return "%.*f" % (dec, value)


def same_sci(a, b):
    ma, ea = a.lower().split("e")
    mb, eb = b.lower().split("e")
    return float(ma) == float(mb) and int(ea) == int(eb)


def check(claim, what, fn, banked):
    try:
        value = fn()
        got = fmt_like(value, banked)
        ok = same_sci(got, banked) if "e" in banked.lower() else got == banked.strip()
    except Exception as e:  # fail closed
        got, ok = f"ERROR {type(e).__name__}: {e}", False
    results.append((claim, what, got, banked, ok))


def check_true(claim, what, fn):
    try:
        ok = bool(fn())
        got = str(ok)
    except Exception as e:
        got, ok = f"ERROR {type(e).__name__}: {e}", False
    results.append((claim, what, got, "True", ok))


def gm_ratio(pods, xc, a="mixed", b="stock", basis="mtzvpp", mols=None, kind="speed"):
    return lambda: geomean(ratio(pods, k, a, b) for k in keys(pods, kind, xc, basis, mols))


def tier_mols(tier):
    return [r["molecule"] for r in rows("tiers") if r["tier"] == tier]


def noisy(pods, key):
    return any(spread(pods, key, a) > 0.10 for a in ("mixed", "cache", "stock")
               if any(r["key"] == key and r["arm"] == a for r in rows(pods)))


def mig_gain(whole, inst, key, arm):
    return lambda: 4 * med(whole, key, arm) / med(inst, key, arm)


TRIO = ("paracetamol", "propranolol", "celecoxib")
XCS3 = ("r2scan", "b3lyp", "wb97m-v")
def pod(name):
    hit = [r for r in rows("pods") if r["csv"] == name]
    if len(hit) != 1:
        raise LookupError(f"pods.csv has {len(hit)} rows for {name}")
    return hit[0]


# The wB97M-V ladder pool is every W pod that pods.csv records as healthy (not CONTENDED), so a
# contention flag cannot be swapped without changing the reading.
W_HEALTHY = tuple(n for n in ("w1", "w2", "w3", "w4", "w4r", "w5") if pod(n)["contended"] == "False")

# --------------------------------------------------------------------------------------------- #
# N1. Native ladder on the RTX PRO 6000 Workstation (PREREG-1)
# --------------------------------------------------------------------------------------------- #
for xc, whole, tiers, lo, hi in (("r2scan", "1.657", ("1.576", "1.665", "1.804"), "1.523", "1.871"),
                                 ("b3lyp", "1.783", ("1.736", "1.779", "1.897"), "1.584", "1.994")):
    check("N1", f"L12 {xc} ladder geomean (24)", gm_ratio("l12", xc), whole)
    check_true("N1", f"L12 {xc} has 24 cells", lambda xc=xc: len(keys("l12", xc=xc)) == 24)
    for t, banked in zip(("S", "M", "L"), tiers):
        check("N1", f"L12 {xc} tier {t}", gm_ratio("l12", xc, mols=tier_mols(t)), banked)
    check("N1", f"L12 {xc} lowest cell", lambda xc=xc: min(ratio("l12", k, "mixed", "stock")
                                                         for k in keys("l12", xc=xc)), lo)
    check("N1", f"L12 {xc} highest cell", lambda xc=xc: max(ratio("l12", k, "mixed", "stock")
                                                          for k in keys("l12", xc=xc)), hi)
check("N1", "bridge leg geomean (24, conv_tol_grad 1e-5, R=1)",
      gm_ratio("l12", "r2scan", kind="bridge"), "1.586")
check("N1", "bridge leg lowest", lambda: min(ratio("l12", k, "mixed", "stock")
                                              for k in keys("l12", "bridge", "r2scan")), "1.19")
check("N1", "bridge leg highest", lambda: max(ratio("l12", k, "mixed", "stock")
                                               for k in keys("l12", "bridge", "r2scan")), "1.76")
check("N1", "wB97M-V ladder geomean (24, healthy pods)", gm_ratio(W_HEALTHY, "wb97m-v"), "2.914")
check_true("N1", "wB97M-V healthy pods cover 24 distinct cells",
           lambda: len(keys(W_HEALTHY, xc="wb97m-v")) == 24)
for t, banked in zip(("S", "M", "L"), ("2.848", "2.995", "2.806")):
    check("N1", f"wB97M-V tier {t}", gm_ratio(W_HEALTHY, "wb97m-v", mols=tier_mols(t)), banked)
check("N1", "wB97M-V lowest cell", lambda: min(ratio(W_HEALTHY, k, "mixed", "stock")
                                                for k in keys(W_HEALTHY, xc="wb97m-v")), "2.593")
check("N1", "wB97M-V highest cell", lambda: max(ratio(W_HEALTHY, k, "mixed", "stock")
                                                 for k in keys(W_HEALTHY, xc="wb97m-v")), "3.250")
for mol, contended, healthy in (("ketoprofen", "2.102", "3.047"), ("diphenhydramine", "1.916", "2.873"),
                                ("propranolol", "2.134", "2.986"), ("atenolol", "2.101", "3.028")):
    k = cell(mol, "wb97m-v")
    check("N1", f"W4 (CONTENDED) {mol}", lambda k=k: ratio("w4", k, "mixed", "stock"), contended)
    check("N1", f"W4r {mol}", lambda k=k: ratio("w4r", k, "mixed", "stock"), healthy)

# --------------------------------------------------------------------------------------------- #
# N2. Cross-GPU (PREREG-2)
# --------------------------------------------------------------------------------------------- #
def trio3(pods, xc, a, b):
    return gm_ratio(pods, xc, a, b, mols=TRIO)


for xc, ms, mc, cs in (("r2scan", "1.607", "1.543", "1.041"), ("b3lyp", "1.688", "1.633", "1.034"),
                       ("wb97m-v", "2.059", "2.051", "1.004")):
    check("N2", f"C0 {xc} mixed/stock (CONTENTION-UNKNOWN)", trio3("c0", xc, "mixed", "stock"), ms)
    check("N2", f"C0 {xc} mixed/cache", trio3("c0", xc, "mixed", "cache"), mc)
    check("N2", f"C0 {xc} cache/stock", trio3("c0", xc, "cache", "stock"), cs)
check("N2", "C0w wB97M-V mixed/stock", trio3("c0w", "wb97m-v", "mixed", "stock"), "2.963")
check("N2", "C0w wB97M-V mixed/cache", trio3("c0w", "wb97m-v", "mixed", "cache"), "2.941")
check("N2", "C0w wB97M-V cache/stock", trio3("c0w", "wb97m-v", "cache", "stock"), "1.008")
check("N2", "L12 trio r2SCAN (healthy PRO 6000 row)", gm_ratio("l12", "r2scan", mols=TRIO), "1.638")
check("N2", "L12 trio B3LYP (healthy PRO 6000 row)", gm_ratio("l12", "b3lyp", mols=TRIO), "1.859")
for card, pods, vals in (
        ("L40S", ("x2a", "x2b"), (("r2scan", "1.729", "1.655", "1.045"), ("b3lyp", "1.524", "1.463", "1.041"),
                                  ("wb97m-v", "2.994", "2.981", "1.004"))),
        ("H100", "x3", (("r2scan", "1.320", "1.084", "1.217"), ("b3lyp", "0.905", "0.745", "1.215"),
                        ("wb97m-v", "0.577", "0.539", "1.070")))):
    for xc, ms, mc, cs in vals:
        check("N2", f"{card} {xc} mixed/stock", trio3(pods, xc, "mixed", "stock"), ms)
        check("N2", f"{card} {xc} mixed/cache (precision effect)", trio3(pods, xc, "mixed", "cache"), mc)
        check("N2", f"{card} {xc} cache/stock", trio3(pods, xc, "cache", "stock"), cs)
check_true("N2", "L40S X2a has exactly 1 NOISY cell of 6 (not DEGRADED)",
           lambda: sum(noisy("x2a", k) for k in keys("x2a", xc=None)) == 1)
check_true("N2", "H100 X3 has no NOISY cell",
           lambda: sum(noisy("x3", k) for k in keys("x3", xc=None)) == 0)
for pn, n in (("x4", 2), ("x4r", 4)):
    check_true("N2", f"A100 {pn.upper()} has {n} NOISY cells of 9, so DEGRADED (> 20 %)",
               lambda pn=pn, n=n: sum(noisy(pn, k) for k in keys(pn, xc=None)) == n
               and n / len(keys(pn, xc=None)) > 0.20)
A100_CLEAN = {  # (pod, mol, xc): (mixed/cache, mixed/stock) as printed in PREREG-2 part 3
    ("x4", "propranolol", "r2scan"): ("1.025", "1.276"), ("x4", "celecoxib", "r2scan"): ("0.991", "1.143"),
    ("x4", "paracetamol", "b3lyp"): ("0.901", "1.091"), ("x4", "propranolol", "b3lyp"): ("0.689", "0.817"),
    ("x4", "celecoxib", "b3lyp"): ("0.611", "0.711"), ("x4", "paracetamol", "wb97m-v"): ("0.531", "0.562"),
    ("x4", "propranolol", "wb97m-v"): ("0.472", "0.490"),
    ("x4r", "paracetamol", "r2scan"): ("1.184", "1.390"), ("x4r", "propranolol", "r2scan"): ("0.987", "1.179"),
    ("x4r", "paracetamol", "wb97m-v"): ("0.566", "0.596"), ("x4r", "propranolol", "wb97m-v"): ("0.492", "0.509"),
    ("x4r", "celecoxib", "wb97m-v"): ("0.470", "0.488"),
}
for (pn, mol, xc), (mc, ms) in A100_CLEAN.items():
    k = cell(mol, xc)
    check_true("N2", f"A100 {pn} {mol} {xc} is a clean (non-NOISY) cell",
               lambda pn=pn, k=k: not noisy(pn, k))
    check("N2", f"A100 {pn} {mol} {xc} mixed/cache", lambda pn=pn, k=k: ratio(pn, k, "mixed", "cache"), mc)
    check("N2", f"A100 {pn} {mol} {xc} mixed/stock", lambda pn=pn, k=k: ratio(pn, k, "mixed", "stock"), ms)
check_true("N2", "A100: exactly 12 clean cells across X4 and X4r",
           lambda: sum(not noisy(p, k) for p in ("x4", "x4r") for k in keys(p, xc=None)) == 12)

# --------------------------------------------------------------------------------------------- #
# N3. Large molecules and bases (PREREG-3)
# --------------------------------------------------------------------------------------------- #
LARGE6 = ("sildenafil", "imatinib", "atorvastatin", "montelukast", "lopinavir", "ritonavir")
check("N3", "XL r2SCAN geomean (6)", gm_ratio("xl1", "r2scan", mols=LARGE6), "1.931")
for m, v in zip(LARGE6, ("1.982", "1.777", "1.903", "1.673", "1.695", "2.729")):
    check("N3", f"XL r2SCAN {m}", lambda m=m: ratio("xl1", cell(m, "r2scan"), "mixed", "stock"), v)
check("N3", "XL B3LYP geomean (6)", gm_ratio("xl2", "b3lyp", mols=LARGE6), "1.338")
for m, v in zip(LARGE6, ("1.297", "1.335", "1.385", "1.360", "1.298", "1.355")):
    check("N3", f"XL B3LYP {m}", lambda m=m: ratio("xl2", cell(m, "b3lyp"), "mixed", "stock"), v)
check("N3", "XL wB97M-V sildenafil", lambda: ratio("xl3a", cell("sildenafil", "wb97m-v"), "mixed", "stock"),
      "2.813")
check("N3", "XL wB97M-V atorvastatin",
      lambda: ratio("xl3b", cell("atorvastatin", "wb97m-v"), "mixed", "stock"), "2.709")
for xc, vals in (("r2scan", ("1.693", "1.762", "2.004")), ("b3lyp", ("1.957", "1.744", "1.590"))):
    for basis, v in zip(("svp", "mtzvpp", "tzvp"), vals):
        check("N3", f"B1 {xc} {basis} geomean (4)", gm_ratio("b1", xc, basis=basis), v)
        check_true("N3", f"B1 {xc} {basis} has 4 cells",
                   lambda xc=xc, basis=basis: len(keys("b1", xc=xc, basis=basis)) == 4)
check("N3", "B1 sildenafil B3LYP mTZVPP",
      lambda: ratio("b1", cell("sildenafil", "b3lyp", "mtzvpp"), "mixed", "stock"), "1.299")
check("N3", "B1 sildenafil B3LYP TZVP",
      lambda: ratio("b1", cell("sildenafil", "b3lyp", "tzvp"), "mixed", "stock"), "1.311")
check_true("N3", "B1 has 2 NOISY cells of 24",
           lambda: sum(noisy("b1", k) for k in keys("b1", xc=None, basis=None)) == 2)

# --------------------------------------------------------------------------------------------- #
# N4. Downstream accuracy (PREREG-4): read from the sentinel readouts in data/downstream.csv
# --------------------------------------------------------------------------------------------- #
def ds(pod, mol, xc, field):
    hit = [r for r in rows("downstream") if r["csv"] == pod and r["key"] == f"downstream/{mol}/{xc}/mtzvpp"]
    if len(hit) != 1:
        raise LookupError(f"{pod} {mol} {xc}: {len(hit)} rows")
    return float(hit[0][field])


DS = {  # pod, mol, xc: (max_dg, max_dmu, de, rho_g, rho_mu) as printed
    ("d1", "paracetamol", "r2scan"): ("1.37e-6", "5.00e-5", "1.3e-10", "4.09", "8.56"),
    ("d1", "propranolol", "r2scan"): ("7.04e-7", "2.10e-5", "3.1e-11", "0.76", "0.31"),
    ("d1", "celecoxib", "r2scan"): ("1.16e-6", "7.00e-5", "3.7e-11", "0.81", "2.70"),
    ("d1", "fluconazole", "r2scan"): ("1.96e-6", "1.004e-4", "1.4e-10", "2.20", "1.39"),
    ("d1", "warfarin", "r2scan"): ("6.55e-7", "6.43e-5", "3.9e-11", "0.24", "1.17"),
    ("d1", "omeprazole", "r2scan"): ("1.49e-6", "6.46e-5", "9.3e-11", "0.82", "0.78"),
    ("d1", "paracetamol", "b3lyp"): ("1.34e-6", "4.79e-5", "2.3e-11", "1.64", "2.22"),
    ("d1", "propranolol", "b3lyp"): ("7.36e-7", "1.91e-5", "2.0e-11", "0.68", "0.72"),
    ("d1", "celecoxib", "b3lyp"): ("1.84e-6", "8.49e-5", "9.0e-11", "0.70", "0.80"),
    ("d1", "fluconazole", "b3lyp"): ("8.29e-7", "5.16e-5", "1.6e-11", "0.53", "1.06"),
    ("d1", "warfarin", "b3lyp"): ("1.67e-6", "2.01e-5", "2.1e-11", "1.24", "0.38"),
    ("d1", "omeprazole", "b3lyp"): ("1.00e-6", "3.48e-5", "4.5e-11", "0.84", "0.48"),
    ("d2a", "paracetamol", "wb97m-v"): ("1.20e-6", "4.07e-5", "2.9e-11", "0.76", "1.14"),
    ("d2a", "propranolol", "wb97m-v"): ("3.10e-6", "7.66e-5", "1.2e-10", "0.96", "4.89"),
    ("d2a", "celecoxib", "wb97m-v"): ("1.87e-6", "4.08e-5", "6.3e-11", "0.62", "0.49"),
    ("d2b", "fluconazole", "wb97m-v"): ("2.37e-6", "3.76e-5", "1.5e-10", "2.41", "3.97"),
    ("d2b", "warfarin", "wb97m-v"): ("2.20e-6", "1.03e-5", "6.4e-11", "0.84", "0.66"),
    ("d2c", "omeprazole", "wb97m-v"): ("2.17e-6", "1.44e-5", "1.0e-10", "1.57", "0.68"),
}
for (pn, mol, xc), banked in DS.items():
    for field, b in zip(("max_dg", "max_dmu", "de", "rho_g", "rho_mu"), banked):
        check("N4", f"{pn} {mol} {xc} {field}", lambda pn=pn, mol=mol, xc=xc, field=field:
              ds(pn, mol, xc, field), b)
CEIL = {"max_dg": 1e-5, "max_dmu": 1e-4, "de": 1e-8}
check_true("N4", "17 of 18 downstream cells within every ceiling; the one outside is fluconazole r2SCAN's dipole",
           lambda: [k for k in DS if any(ds(*k, f) > c for f, c in CEIL.items())]
           == [("d1", "fluconazole", "r2scan")]
           and ds("d1", "fluconazole", "r2scan", "max_dg") <= CEIL["max_dg"])
check_true("N4", "every gradient within the ceiling with >= 3x margin",
           lambda: max(ds(*k, "max_dg") for k in DS) * 3 <= CEIL["max_dg"])
check("N4", "largest gradient deviation", lambda: max(ds(*k, "max_dg") for k in DS), "3.10e-6")
check_true("N4", "every |dE| <= 1.6e-10 Ha", lambda: max(ds(*k, "de") for k in DS) <= 1.6e-10)
check_true("N4", "17 of 18 read rho_g <= 3", lambda: sum(ds(*k, "rho_g") <= 3 for k in DS) == 17)
check_true("N4", "15 of 18 read rho_mu <= 3", lambda: sum(ds(*k, "rho_mu") <= 3 for k in DS) == 15)
check("N4", "S' (stock, atom guess) dipole deviation on celecoxib B3LYP",
      lambda: ds("d1", "celecoxib", "b3lyp", "max_dmu_sp"), "1.058e-4")
check_true("N4", "D1's FAIL is recorded in pods.csv",
           lambda: [r["status"] for r in rows("pods") if r["csv"] == "d1"] == ["FAIL"])

# --------------------------------------------------------------------------------------------- #
# N5. MIG 1g.24gb instances against a whole Server Edition card (PREREG-5)
# --------------------------------------------------------------------------------------------- #
M1 = ("m1a", "m1b")
M2 = ("m2ar", "m2br")
M2C = ("m2a", "m2b")
MIG_CELLS = [cell(m, x) for x in ("r2scan", "b3lyp") for m in TRIO] + \
            [cell(m, "wb97m-v") for m in ("paracetamol", "propranolol")]
GAIN5 = {  # (mol, xc, arm): (M1, M2) as printed in the PREREG-5 RESULT table
    ("paracetamol", "r2scan", "stock"): ("1.51", "1.55"), ("propranolol", "r2scan", "stock"): ("1.42", "1.45"),
    ("celecoxib", "r2scan", "stock"): ("1.35", "1.38"), ("paracetamol", "r2scan", "mixed"): ("2.00", "1.99"),
    ("propranolol", "r2scan", "mixed"): ("1.69", "1.75"), ("celecoxib", "r2scan", "mixed"): ("1.59", "1.62"),
    ("paracetamol", "b3lyp", "stock"): ("1.72", "1.73"), ("propranolol", "b3lyp", "stock"): ("1.35", "1.38"),
    ("celecoxib", "b3lyp", "stock"): ("1.16", "1.19"), ("paracetamol", "b3lyp", "mixed"): ("1.98", "1.95"),
    ("propranolol", "b3lyp", "mixed"): ("1.08", "1.09"), ("celecoxib", "b3lyp", "mixed"): ("0.92", "0.93"),
    ("paracetamol", "wb97m-v", "stock"): ("1.00", "1.02"), ("propranolol", "wb97m-v", "stock"): ("1.07", "1.05"),
    ("paracetamol", "wb97m-v", "mixed"): ("1.19", "1.20"), ("propranolol", "wb97m-v", "mixed"): ("1.18", "1.16"),
}
for (mol, xc, arm), (g1, g2) in GAIN5.items():
    k = cell(mol, xc)
    check("N5", f"MIG gain M1 {mol} {xc} {arm}", mig_gain("m0", M1, k, arm), g1)
    check("N5", f"MIG gain M2 {mol} {xc} {arm}", mig_gain("m0", M2, k, arm), g2)
for xc, mols, vals in (("r2scan", TRIO, ("1.43", "1.46", "1.75", "1.78")),
                       ("b3lyp", TRIO, ("1.39", "1.42", "1.25", "1.26")),
                       ("wb97m-v", ("paracetamol", "propranolol"), ("1.03", "1.03", "1.19", "1.18"))):
    for (arm, inst), v in zip((("stock", M1), ("stock", M2), ("mixed", M1), ("mixed", M2)), vals):
        check("N5", f"MIG gain geomean {xc} {arm} {'M1' if inst == M1 else 'M2'}",
              lambda xc=xc, mols=mols, arm=arm, inst=inst:
              geomean(4 * med("m0", cell(m, xc), arm) / med(inst, cell(m, xc), arm) for m in mols), v)
INST5 = {  # mixed/stock inside the instance, M0 / M1 / M2, as printed
    ("paracetamol", "r2scan"): ("1.670", "2.211", "2.152"), ("propranolol", "r2scan"): ("1.654", "1.968", "1.986"),
    ("celecoxib", "r2scan"): ("1.749", "2.059", "2.053"), ("paracetamol", "b3lyp"): ("2.038", "2.336", "2.293"),
    ("propranolol", "b3lyp"): ("1.996", "1.597", "1.575"), ("celecoxib", "b3lyp"): ("2.025", "1.598", "1.587"),
    ("paracetamol", "wb97m-v"): ("3.081", "3.664", "3.630"), ("propranolol", "wb97m-v"): ("3.200", "3.536", "3.525"),
}
for (mol, xc), (v0, v1, v2) in INST5.items():
    k = cell(mol, xc)
    check("N5", f"M0 mixed/stock {mol} {xc}", lambda k=k: ratio("m0", k, "mixed", "stock"), v0)
    check("N5", f"M1 mixed/stock {mol} {xc}", lambda k=k: ratio(M1, k, "mixed", "stock"), v1)
    check("N5", f"M2 mixed/stock {mol} {xc}", lambda k=k: ratio(M2, k, "mixed", "stock"), v2)
check_true("N5", "pod-to-pod: every M2/M1 instance wall within 0.971..1.023",
           lambda: all(0.9705 <= med(M2, k, a) / med(M1, k, a) <= 1.0235
                       for k in MIG_CELLS for a in ("mixed", "stock")))
check_true("N5", "CONTENDED M2a/M2b were 1.3-5.1 % faster than their healthy re-runs",
           lambda: all(1.0125 <= med(M2, k, a) / med(M2C, k, a) <= 1.0515
                       for k in MIG_CELLS for a in ("mixed", "stock")))
check_true("N5", "M2a and M2b CONTENDED in pods.csv, re-runs healthy",
           lambda: {r["csv"]: r["contended"] for r in rows("pods") if r["csv"].startswith("m2")}
           == {"m2a": "True", "m2b": "True", "m2ar": "False", "m2br": "False"})

# --------------------------------------------------------------------------------------------- #
# N6. Where one instance stops holding the treated path (PREREG-6)
# --------------------------------------------------------------------------------------------- #
F1 = ("f1a", "f1b")
XL3 = ("sildenafil", "imatinib", "atorvastatin")
GAIN6 = {("r2scan", "stock"): ("1.28", "1.28", "1.19"), ("r2scan", "mixed"): ("1.42", "1.44", "1.29"),
         ("b3lyp", "stock"): ("1.16", "1.09", "1.09"), ("b3lyp", "mixed"): ("1.11", "1.08", "0.98")}
for (xc, arm), vals in GAIN6.items():
    for m, v in zip(XL3, vals):
        check("N6", f"MIG gain {m} {xc} {arm}", mig_gain("f0", F1, cell(m, xc), arm), v)


def one(s):
    if len(s) != 1:
        raise ValueError(f"expected exactly one value, got {sorted(s)}")
    return next(iter(s))


def tiers_of(pods, arm="mixed"):
    pods = (pods,) if isinstance(pods, str) else pods
    return {(r["key"], r["ao_cache_tier"]) for p in pods for r in rows(p) if r["arm"] == arm}


check_true("N6", "all 6 F0 mixed cells at tier fp64+fp32",
           lambda: {t for _, t in tiers_of("f0")} == {"fp64+fp32"} and len({k for k, _ in tiers_of("f0")}) == 6)
check_true("N6", "all 6 instance mixed cells at tier fp32 (CACHE_FALLBACK)",
           lambda: {t for _, t in tiers_of(F1)} == {"fp32"} and len({k for k, _ in tiers_of(F1)}) == 6)
for m, gb in zip(XL3, ("5.55", "5.23", "8.62")):
    check("N6", f"B3LYP {m} DF tensor (GB), instance", lambda m=m: one({
        float(r["cderi_bytes"]) for r in rows("f1b") if r["key"] == cell(m, "b3lyp")}) / 1e9, gb)
check_true("N6", "every B3LYP DF tensor on the device, identical size on both cards",
           lambda: all(r["cderi_types"] == "cupy.ndarray" for p in ("f0", "f1b") for r in rows(p)
                       if "/b3lyp/" in r["key"])
           and all({r["cderi_bytes"] for r in rows("f0") if r["key"] == cell(m, "b3lyp")}
                   == {r["cderi_bytes"] for r in rows("f1b") if r["key"] == cell(m, "b3lyp")} for m in XL3))
check_true("N6", "no r2SCAN run builds a DF tensor",
           lambda: all(r["cderi_types"] == "" and r["cderi_bytes"] == "0"
                       for p in ("f0", "f1a") for r in rows(p) if "/r2scan/" in r["key"]))

# --------------------------------------------------------------------------------------------- #
# N7. Flags, fit tiers, counts and disclosures that the claims rest on
# --------------------------------------------------------------------------------------------- #
# Derived from the CSVs themselves (pods with speed rows), never from the pods.csv field it checks,
# and pinned: a pod cannot drop out of these checks by editing its own flags.
SPEED_PODS = sorted(r["csv"] for r in rows("pods")
                    if any(x.get("key", "").startswith("speed/") for x in rows(r["csv"])))
check_true("N7", "the 32 speed pods are exactly the expected set",
           lambda: SPEED_PODS == sorted(["b1", "c0", "c0w", "f0", "f1a", "f1b", "l12", "l12r", "l12r2", "m0", "m1a",
                                         "m1b", "m2a", "m2ar", "m2b", "m2br", "w1", "w2", "w3", "w4", "w4r", "w5",
                                         "x2a", "x2b", "x3", "x3r", "x4", "x4r", "xl1", "xl2", "xl3a", "xl3b"]))


def speed_keys(p):
    return keys(p, "speed", None, None)


for p in SPEED_PODS:
    check_true("N7", f"{p}: pods.csv speed_cells / noisy_cells / degraded match the data",
               lambda p=p: (len(speed_keys(p)) == int(pod(p)["speed_cells"])
                            and sum(noisy(p, k) for k in speed_keys(p)) == int(pod(p)["noisy_cells"])
                            and (sum(noisy(p, k) for k in speed_keys(p)) / len(speed_keys(p)) > 0.20)
                            == (pod(p)["degraded"] == "True")))
check_true("N7", "every row of every pod CSV converged, with no error",
           lambda: all(r["converged"] == "True" and not r["error"]
                       for p in rows("pods") for r in rows(p["csv"])))
check_true("N7", "the only NOISY cell of M0 is paracetamol wB97M-V (both MIG gains for it rest on it)",
           lambda: [k for k in speed_keys("m0") if noisy("m0", k)] == [cell("paracetamol", "wb97m-v")])


def under_1s(p, k):
    return any(w < 1.0 for a in ("mixed", "cache", "stock")
               if any(r["key"] == k and r["arm"] == a for r in rows(p)) for w in walls(p, k, a))


check_true("N7", "exactly eleven cells have a warm wall under 1 s, all B3LYP (metformin, paracetamol)",
           lambda: sorted((p["csv"], k) for p in rows("pods") if int(p["speed_cells"]) > 0
                          for k in speed_keys(p["csv"]) if under_1s(p["csv"], k)) == sorted([
               ("l12", cell("metformin", "b3lyp")), ("b1", cell("paracetamol", "b3lyp", "svp")),
               ("b1", cell("paracetamol", "b3lyp", "mtzvpp")), ("x3", cell("paracetamol", "b3lyp")),
               ("x4", cell("paracetamol", "b3lyp")), ("x4r", cell("paracetamol", "b3lyp")),
               ("l12r", cell("metformin", "b3lyp")), ("l12r", cell("paracetamol", "b3lyp")),
               ("l12r2", cell("metformin", "b3lyp")), ("l12r2", cell("paracetamol", "b3lyp")),
               ("x3r", cell("paracetamol", "b3lyp"))]))
check_true("N7", "XL: r2SCAN/B3LYP mixed at fp64+fp32, wB97M-V at fp64",
           lambda: {t for _, t in tiers_of(("xl1", "xl2"))} == {"fp64+fp32"}
           and {t for _, t in tiers_of(("xl3a", "xl3b"))} == {"fp64"})
check_true("N7", "MIG trio pods: r2SCAN/B3LYP mixed at fp64+fp32 and wB97M-V at fp64, on M0, M1 and M2",
           lambda: all({t for k, t in tiers_of(p) if "/wb97m-v/" not in k} <= {"fp64+fp32"}
                       and {t for k, t in tiers_of(p) if "/wb97m-v/" in k} <= {"fp64"}
                       for p in ("m0", "m1a", "m1b", "m2ar", "m2br"))
           and len({k for p in M1 for k, _ in tiers_of(p)}) == 8)
check_true("N7", "A100: 11 of the 12 clean readings are NEUTRAL or HARMS on the precision effect",
           lambda: sum(ratio(p, cell(m, x), "mixed", "cache") < 1.15
                       for (p, m, x) in A100_CLEAN) == 11 and len(A100_CLEAN) == 12)
check("N7", "A100 clean r2SCAN/B3LYP cells: lowest cache/stock",
      lambda: min(ratio(p, cell(m, x), "cache", "stock") for (p, m, x) in A100_CLEAN if x != "wb97m-v"),
      "1.153")
check("N7", "A100 clean r2SCAN/B3LYP cells: highest cache/stock",
      lambda: max(ratio(p, cell(m, x), "cache", "stock") for (p, m, x) in A100_CLEAN if x != "wb97m-v"),
      "1.245")
check_true("N7", "gradient prediction (<= 1e-6 Ha/Bohr) missed in 8 of 12 D1 cells and 14 of 18 cells",
           lambda: sum(ds(*k, "max_dg") > 1e-6 for k in DS if k[0] == "d1") == 8
           and sum(ds(*k, "max_dg") > 1e-6 for k in DS) == 14 and len(DS) == 18)
check_true("N7", "rho_g <= 3 in 11 of 12 D1 cells",
           lambda: sum(ds(*k, "rho_g") <= 3 for k in DS if k[0] == "d1") == 11)
check_true("N7", "downstream.csv: only fluconazole r2SCAN is GATE_FAIL; every row carries its pod's RUN_ID",
           lambda: [(r["csv"], r["key"]) for r in rows("downstream") if r["status"] != "OK"]
           == [("d1", "downstream/fluconazole/r2scan/mtzvpp")]
           and all(r["run_id"] == pod(r["csv"])["run_id"] for r in rows("downstream")))
COLD = {"c0": ("paracetamol", "r2scan", "83.8"), "l12": ("metformin", "r2scan", "96.1"),
        "m0": ("paracetamol", "r2scan", "82.3"), "m1a": ("paracetamol", "r2scan", "96.0"),
        "xl1": ("sildenafil", "r2scan", "87.9"), "c0w": ("paracetamol", "wb97m-v", "64.0")}
for p, (m, x, v) in COLD.items():
    check("N7", f"cold first mixed run (pair 0) in {p}: {m} {x}, seconds",
          lambda p=p, m=m, x=x: one({float(r["wall_s"]) for r in rows(p) if r["key"] == cell(m, x)
                                     and r["arm"] == "mixed" and r["pair"] == "0"}), v)

check("N7", "M0 (Server Edition) B3LYP trio mixed/stock", gm_ratio("m0", "b3lyp", mols=TRIO), "2.019")
check("N7", "XL r2SCAN geomean without ritonavir (5)", gm_ratio("xl1", "r2scan", mols=LARGE6[:5]), "1.802")
check("N7", "ritonavir r2SCAN stock warm wall (s)", lambda: med("xl1", cell("ritonavir", "r2scan"), "stock"), "72.2")
check("N7", "lopinavir r2SCAN stock warm wall (s)", lambda: med("xl1", cell("lopinavir", "r2scan"), "stock"), "50.9")
check("N7", "A100 wB97M-V mixed/stock, lowest clean cell (both pods)",
      lambda: min(ratio(p, k, "mixed", "stock") for p in ("x4", "x4r") for k in keys(p, xc="wb97m-v")
                  if not noisy(p, k)), "0.488")
check("N7", "A100 wB97M-V mixed/stock, highest clean cell (both pods)",
      lambda: max(ratio(p, k, "mixed", "stock") for p in ("x4", "x4r") for k in keys(p, xc="wb97m-v")
                  if not noisy(p, k)), "0.596")
check("N7", "H100 wB97M-V trio mixed/stock as a slowdown factor (stock/mixed inverted)",
      lambda: 1 / gm_ratio("x3", "wb97m-v", mols=TRIO)(), "1.73")
check("N7", "H100 celecoxib wB97M-V mixed/stock", lambda: ratio("x3", cell("celecoxib", "wb97m-v"), "mixed", "stock"),
      "0.521")
check("N7", "worst B3LYP mixed/stock on H100/A100",
      lambda: min(ratio(p, k, "mixed", "stock") for p in ("x3", "x4", "x4r") for k in keys(p, xc="b3lyp")), "0.711")
check("N7", "C0 vs D1: r2SCAN paracetamol gradient residual on C0",
      lambda: ds("c0", "paracetamol", "r2scan", "max_dg"), "1.368e-6")
check("N7", "C0 vs D1: r2SCAN paracetamol gradient residual on D1",
      lambda: ds("d1", "paracetamol", "r2scan", "max_dg"), "1.369e-6")
for p, (m, x, v) in {"x2a": ("paracetamol", "r2scan", "8.6"), "x3": ("paracetamol", "r2scan", "8.4")}.items():
    check("N7", f"cold first mixed run (pair 0) in {p}: {m} {x}, seconds",
          lambda p=p, m=m, x=x: one({float(r["wall_s"]) for r in rows(p) if r["key"] == cell(m, x)
                                     and r["arm"] == "mixed" and r["pair"] == "0"}), v)

# Pod status, and CONTENDED exactly when the idle draw before the first group exceeds 200 W
# (PREREG-0 Amendment 3); C0 predates the telemetry and has neither.
check_true("N7", "every pod PASSes except D1",
           lambda: {r["csv"]: r["status"] for r in rows("pods") if r["status"] != "PASS"} == {"d1": "FAIL"})
check_true("N7", "CONTENDED <=> idle power > 200 W, on every pod but C0; C0 has no reading",
           lambda: all((r["contended"] == "True") == (float(r["idle_power_w"]) > 200.0)
                       for r in rows("pods") if r["csv"] != "c0")
           and (pod("c0")["contended"], pod("c0")["idle_power_w"]) == ("", "")
           and sorted(r["csv"] for r in rows("pods") if r["contended"] == "True") == ["m2a", "m2b", "st2", "w4"])

# Treatment evidence and the accuracy gates, re-derived on every speed and bridge row of every pod.
POLICY = {
    ("r2scan", "mixed"): "MixedPrecision(xc=True, k=False, vv10=False, xc_switch_tol=0.001, k_switch_tol=0.001, "
                         "vv10_switch_tol=1e-05, ao_cache_fp64=True)",
    ("b3lyp", "mixed"): "MixedPrecision(xc=True, k=True, vv10=False, xc_switch_tol=0.0003, k_switch_tol=0.001, "
                        "vv10_switch_tol=1e-05, ao_cache_fp64=True)",
    ("wb97m-v", "mixed"): "MixedPrecision(xc=False, k=False, vv10=True, xc_switch_tol=0.001, k_switch_tol=0.001, "
                          "vv10_switch_tol=1e-05, ao_cache_fp64=True)",
}
for x in XCS3:
    POLICY[(x, "cache")] = ("MixedPrecision(xc=False, k=False, vv10=False, xc_switch_tol=0.001, k_switch_tol=0.001, "
                            "vv10_switch_tol=1e-05, ao_cache_fp64=True)")
    POLICY[(x, "stock")] = ""


def tier_expected(p, x, arm):
    if arm == "stock":
        return ""
    if arm == "cache" or x == "wb97m-v":
        return "fp64"
    return "fp32" if p in ("f1a", "f1b") else "fp64+fp32"


def timed_rows(p):
    return [r for r in rows(p) if r["key"].split("/")[0] in ("speed", "bridge")]


check_true("N7", "every timed row carries the expected policy and cache tier for its functional and arm",
           lambda: all(r["policy"] == POLICY[(r["key"].split("/")[2], r["arm"])]
                       and r["ao_cache_tier"] == tier_expected(p, r["key"].split("/")[2], r["arm"])
                       for p in SPEED_PODS for r in timed_rows(p)))


def gate_worst():
    de, dc, n = 0.0, 0, 0
    for p in SPEED_PODS:
        by = {}
        for r in timed_rows(p):
            by.setdefault((r["key"], r["pair"]), {})[r["arm"]] = r
        for arms in by.values():
            s = arms["stock"]
            for a, r in arms.items():
                if a != "stock":
                    de = max(de, abs(float(r["e"]) - float(s["e"])))
                    dc = max(dc, abs(int(r["cycles"]) - int(s["cycles"])))
                    n += 1
    if not n:
        raise ValueError("no arm/stock pairs")
    return de, dc


check_true("N7", "accuracy gates on every timed pair: |E_arm - E_stock| <= 1e-8 Ha and |cycles| within 1",
           lambda: gate_worst()[0] <= 1e-8 and gate_worst()[1] <= 1)
check("N7", "worst |E_arm - E_stock| over every timed pair (Ha)", lambda: gate_worst()[0], "1.75e-10")


# Cold start: the first group of a pod pays it, later groups (fresh processes, same pod) do not.
def first_cold(p, x):
    ks = [r["key"] for r in timed_rows(p) if r["key"].split("/")[2] == x]
    k = ks[0]
    return one({float(r["wall_s"]) for r in rows(p) if r["key"] == k and r["arm"] == "mixed" and r["pair"] == "0"})


check("N7", "cold first mixed SCF, first group, Blackwell whole cards and instances: lowest (s)",
      lambda: min(first_cold(p, timed_rows(p)[0]["key"].split("/")[2]) for p in SPEED_PODS
                  if pod(p)["card"].startswith("NVIDIA RTX PRO 6000")), "64.0")
check("N7", "cold first mixed SCF, first group, Blackwell whole cards and instances: highest (s)",
      lambda: max(first_cold(p, timed_rows(p)[0]["key"].split("/")[2]) for p in SPEED_PODS
                  if pod(p)["card"].startswith("NVIDIA RTX PRO 6000")), "167.6")
for p, x, v in (("c0", "b3lyp", "1.7"), ("c0", "wb97m-v", "15.3"), ("m0", "b3lyp", "1.6"), ("l12", "b3lyp", "1.6")):
    check("N7", f"cold first mixed SCF in a LATER group of {p} ({x}), seconds",
          lambda p=p, x=x: first_cold(p, x), v)

# --------------------------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------------------------- #
check_true("P", "every RUN_ID is <run>-1-<sha8> and its sentinel path follows the ci-bus pattern",
           lambda: all(p["run_id"] == f"{p['gh_run']}-1-{p['repo_sha'][:8]}"
                       and p["sentinel"] == f"ci-results/runs/rfcbench-{p['repo_sha'][:12]}-{p['run_id']}.txt"
                       for p in rows("pods")))
check_true("P", "pods.csv lists every CSV in data/ exactly once",
           lambda: sorted(r["csv"] for r in rows("pods")) == sorted(
               f[:-4] for f in os.listdir(DATA) if f.endswith(".csv")
               and f[:-4] not in ("pods", "downstream", "tiers")))
check_true("P", "every speed CSV row carries its pod's RUN_ID",
           lambda: all({r["run_id"] for r in rows(p["csv"])} == {p["run_id"]} for p in rows("pods")))


# --------------------------------------------------------------------------------------------- #
# P2. Provenance: every CSV re-extracted from its shipped sentinel by the shipped extractor
# --------------------------------------------------------------------------------------------- #
PROV = os.path.join(HERE, "provenance")
# PREREG-7's extra-mode pods, pinned (their CSVs have no `key` column).
EXTRA_PODS = ("cc1", "cs1", "st1", "st2", "st2r")


def _extractor():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rfcbench_extract_shipped",
                                                  os.path.join(PROV, "rfcbench_extract.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sentinel(name):
    with open(os.path.join(PROV, "sentinels", name + ".txt"), "rb") as fh:
        return fh.read()


def _sha(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _reextracted_matches(name):
    """The CSV in data/ equals, column for column, what the shipped extractor writes from the shipped
    sentinel (columns the CSV carries; older CSVs predate the cderi_* columns). PREREG-7's extra
    modes (one row per SCF, with stage timers) go through the extractor's extra-mode writer."""
    import io
    X = _extractor()
    p = pod(name)
    text = _sentinel(name).decode()
    buf = io.StringIO()
    if name in EXTRA_PODS:
        w = csv.DictWriter(buf, fieldnames=X.EXTRA_CSV_FIELDS)
        w.writeheader()
        w.writerows(X.extra_rows(text, p["run_id"]))
    else:
        w = csv.DictWriter(buf, fieldnames=X.CSV_FIELDS)
        w.writeheader()
        w.writerows(X.csv_rows(text, p["run_id"]))
    fresh = list(csv.DictReader(io.StringIO(buf.getvalue())))
    have = rows(name)
    if len(fresh) != len(have) or not have:
        return False
    cols = [c for c in have[0] if c in fresh[0]]
    return all(a[c] == b[c] for a, b in zip(have, fresh) for c in cols) and len(cols) == len(have[0])


def _flags_match(name):
    X = _extractor()
    p = pod(name)
    out = X.extract(_sentinel(name).decode(), p["run_id"], unquotable=True)
    c = out.get("contended")
    common = (out["status"] == p["status"]
              and ("" if c is None else str(c)) == p["contended"]
              and ("" if out.get("idle_power_w") is None else str(out["idle_power_w"])) == p["idle_power_w"])
    if name in EXTRA_PODS:   # no speed cells: the SCF count is the CSV's row count instead
        return (common and out.get("mode") in ("coldstart", "stages", "concur")
                and (p["speed_cells"], p["noisy_cells"], p["degraded"]) == ("0", "", "")
                and out["scfs"] == len(rows(name)) > 0)
    return (common and str(out["noisy"]) == p["noisy_cells"] and str(out["degraded"]) == p["degraded"]
            and str(len(out["cells"])) == p["speed_cells"])


def _downstream_matches():
    X = _extractor()
    got = []
    for p in rows("pods"):
        out = X.extract(_sentinel(p["csv"]).decode(), p["run_id"], unquotable=True)
        for d in out.get("downstream") or []:
            got.append((p["csv"], d["key"], d["status"], d["readings"]))
    have = rows("downstream")
    if len(got) != len(have) or not have:
        return False
    for (pn, key, status, rd), r in zip(got, have):
        if (pn, key, status) != (r["csv"], r["key"], r["status"]):
            return False
        for f in ("max_dg", "max_dmu", "de", "rho_g", "rho_mu", "max_dg_sp", "max_dmu_sp"):
            if float(rd[f]) != float(r[f]):
                return False
    return True


for _p in rows("pods"):
    _n = _p["csv"]
    check_true("P2", f"{_n}: shipped sentinel matches pods.csv sentinel_sha256 and carries its RUN_ID",
               lambda n=_n: _sha(_sentinel(n)) == pod(n)["sentinel_sha256"]
               and _sentinel(n).decode().splitlines()[1].strip() == f"RUN_ID={pod(n)['run_id']}")
    check_true("P2", f"{_n}: CSV re-extracted from the sentinel by the shipped extractor is identical",
               lambda n=_n: _reextracted_matches(n))
    check_true("P2", f"{_n}: status / contention / idle W / NOISY / DEGRADED / cells re-derived from the sentinel",
               lambda n=_n: _flags_match(n))
check_true("P2", "downstream.csv re-derived from the D1/D2/C0 sentinels is identical",
           _downstream_matches)

# --------------------------------------------------------------------------------------------- #
# N8. Cold start, replication, B3LYP stage timing and concurrency (PREREG-7)
# --------------------------------------------------------------------------------------------- #
# The extra-mode CSVs (cs1, st1, st2, st2r, cc1) carry one row per SCF: phase, proc, pair, t0/t1 and
# exclusive stage timers. Pair 0 is the cold (or first-pass) run, pair 1 the warm (second-pass) one.
def xrows(p, **kw):
    out = [r for r in rows(p) if all(r[k] == v for k, v in kw.items())]
    if not out:
        raise LookupError(f"no rows in {p} for {kw}")
    return out


def xone(p, field, **kw):
    return float(one({r[field] for r in xrows(p, **kw)}))


def cold_extra(phase, x):
    return lambda: (xone("cs1", "wall_s", phase=phase, xc=x, mol="paracetamol", pair="0")
                    - xone("cs1", "wall_s", phase=phase, xc=x, mol="paracetamol", pair="1"))


CS1 = {"P1_stock_first": ("134.0", "1.45", "0.64"), "P2_mixed_first": ("1.86", "0.19", "1.13"),
       "P3_mixed_again": ("0.81", "0.01", "0.06"), "P4_mixed_wiped_cupy": ("9.92", "1.61", "1.62"),
       "P5_mixed_wiped_cuda": ("125.7", "0.01", "0.18")}
for ph, vals in CS1.items():
    for x, v in zip(XCS3, vals):
        check("N8", f"CS1 {ph}: {x} cold extra (s)", cold_extra(ph, x), v)
check_true("N8", "CS1 thresholds: P1 and P5 r2SCAN >= 30 s; P2 and P4 all <= 15 s; P3 all <= 5 s",
           lambda: cold_extra("P1_stock_first", "r2scan")() >= 30 and cold_extra("P5_mixed_wiped_cuda", "r2scan")() >= 30
           and all(cold_extra(ph, x)() <= lim for ph, lim in (("P2_mixed_first", 15), ("P4_mixed_wiped_cupy", 15),
                                                              ("P3_mixed_again", 5)) for x in XCS3))
check_true("N8", "CS1: r2SCAN runs first in every phase, so the cold cost is charged to it",
           lambda: all(min(xrows("cs1", phase=ph), key=lambda r: int(r["seq"]))["xc"] == "r2scan" for ph in CS1))


def _cs1_cache(phase, which):
    X = _extractor()
    ph = [q for q in X.extract(_sentinel("cs1").decode(), pod("cs1")["run_id"], unquotable=True)["phases"]
          if q["phase"] == phase]
    return one({(ph[0][which]["files"], ph[0][which]["bytes"])}) if len(ph) == 1 else None


check_true("N8", "CS1: P1 leaves the driver JIT cache at 60 files and 127,910,485 bytes (128 MB)",
           lambda: _cs1_cache("P1_stock_first", "cuda_cache") == (60, 127910485))

# L12r / L12r2: the L12 replication. Both DEGRADED, so clean cells are reported (PREREG-0 sec. 6).
L12R = ("l12r", "l12r2")
for p, (r2, b3) in {"l12r": ("1.721", "1.844"), "l12r2": ("1.718", "1.868")}.items():
    check("N8", f"{p} r2SCAN ladder geomean, all 24", gm_ratio(p, "r2scan"), r2)
    check("N8", f"{p} B3LYP ladder geomean, all 24", gm_ratio(p, "b3lyp"), b3)


def gm_clean(pods, p, x):
    ks = [k for k in keys(p, xc=x) if not any(noisy(q, k) for q in pods)]
    return geomean(ratio(p, k, "mixed", "stock") for k in ks), len(ks)


for p, vals in {"l12r": (("1.717", 16), ("1.872", 13)), "l12r2": (("1.734", 16), ("1.873", 11))}.items():
    for x, (v, n) in zip(("r2scan", "b3lyp"), vals):
        check("N8", f"{p} {x} geomean over its own clean cells (n = {n})",
              lambda p=p, x=x, n=n: gm_clean((p,), p, x)[0] if gm_clean((p,), p, x)[1] == n else -1, v)
for p, vals in {"l12r": ("1.709", "1.830"), "l12r2": ("1.720", "1.818")}.items():
    for x, v, n in zip(("r2scan", "b3lyp"), vals, (11, 5)):
        check("N8", f"{p} {x} geomean over cells clean on both replicates (n = {n})",
              lambda p=p, x=x, n=n: gm_clean(L12R, p, x)[0] if gm_clean(L12R, p, x)[1] == n else -1, v)
check("N8", "L12 replication: largest |replicate - L12| over all, own-clean and both-clean geomeans",
      lambda: max(abs(v - gm_ratio("l12", x)()) for x in ("r2scan", "b3lyp") for p in L12R
                  for v in (gm_ratio(p, x)(), gm_clean((p,), p, x)[0], gm_clean(L12R, p, x)[0])), "0.090")
check_true("N8", "L12r, L12r2 and ST1 ran on the same physical card (one GPU UUID in all three sentinels)",
           lambda: len({_sentinel(p).decode().split('"listing":["GPU 0: ')[1].split("UUID: ")[1][:12]
                        for p in ("l12r", "l12r2", "st1")}) == 1)
check("N8", "L12r/L12r2 r2SCAN per-cell mixed/stock, lowest", lambda: min(ratio(p, k, "mixed", "stock")
      for p in L12R for k in keys(p, xc="r2scan")), "1.55")
check("N8", "L12r/L12r2 r2SCAN per-cell mixed/stock, highest", lambda: max(ratio(p, k, "mixed", "stock")
      for p in L12R for k in keys(p, xc="r2scan")), "2.02")
check("N8", "L12r/L12r2 B3LYP per-cell mixed/stock, lowest", lambda: min(ratio(p, k, "mixed", "stock")
      for p in L12R for k in keys(p, xc="b3lyp")), "1.26")
check("N8", "L12r/L12r2 B3LYP per-cell mixed/stock, highest", lambda: max(ratio(p, k, "mixed", "stock")
      for p in L12R for k in keys(p, xc="b3lyp")), "2.13")

# X3r: the H100 replication.
X3R = {("r2scan", "mixed", "stock"): "1.398", ("b3lyp", "mixed", "stock"): "1.021",
       ("wb97m-v", "mixed", "stock"): "0.615", ("r2scan", "mixed", "cache"): "1.084",
       ("b3lyp", "mixed", "cache"): "0.764", ("wb97m-v", "mixed", "cache"): "0.563"}
for (x, a, b), v in X3R.items():
    check("N8", f"X3r H100 trio {x} {a}/{b}", gm_ratio("x3r", x, a, b, mols=TRIO), v)
for (x, a, b), v in {("b3lyp", "mixed", "stock"): "0.116", ("r2scan", "mixed", "stock"): "0.078",
                     ("wb97m-v", "mixed", "stock"): "0.038", ("r2scan", "mixed", "cache"): "0.000",
                     ("b3lyp", "mixed", "cache"): "0.019", ("wb97m-v", "mixed", "cache"): "0.024"}.items():
    check("N8", f"X3r - X3, {x} {a}/{b}",
          lambda x=x, a=a, b=b: gm_ratio("x3r", x, a, b, mols=TRIO)() - gm_ratio("x3", x, a, b, mols=TRIO)(), v)
check_true("N8", "X3r: only B3LYP mixed/stock moves by more than 0.10 (the replication falsifier fires once)",
           lambda: [(x, a, b) for (x, a, b) in X3R if abs(gm_ratio("x3r", x, a, b, mols=TRIO)()
                                                         - gm_ratio("x3", x, a, b, mols=TRIO)()) > 0.10]
           == [("b3lyp", "mixed", "stock")])


# ST1 / ST2 / ST2r: exclusive stage times, pair 1 only.
def stage(p, mol, x, arm, st):
    return xone(p, st, mol=mol, xc=x, arm=arm, pair="1")


def st_ratio(p, mol, x, st, a="stock", b="mixed"):
    return lambda: stage(p, mol, x, a, st) / stage(p, mol, x, b, st)


def share(p, mol, x, st, arm="stock"):
    return lambda: stage(p, mol, x, arm, st) / xone(p, "wall_s", mol=mol, xc=x, arm=arm, pair="1")


ST = {("st1", "paracetamol", "b3lyp"): ("0.09", "2.56", "0.95", "0.91", "0.992"),
      ("st1", "celecoxib", "b3lyp"): ("0.34", "3.12", "1.50", "0.94", "0.987"),
      ("st1", "sildenafil", "b3lyp"): ("0.43", "1.99", "1.05", "1.00", "1.008"),
      ("st1", "atorvastatin", "b3lyp"): ("0.50", "3.08", "1.07", "0.95", "1.000"),
      ("st1", "celecoxib", "r2scan"): ("0.11", "2.06", "0.97", "0.92", "1.009"),
      ("st1", "sildenafil", "r2scan"): ("0.11", "2.31", "1.04", "0.97", "0.974"),
      ("st2r", "celecoxib", "b3lyp"): ("0.40", "3.80", "1.09", "0.95", "0.999"),
      ("st2r", "sildenafil", "b3lyp"): ("0.48", "2.01", "1.17", "0.95", "1.001")}
for (p, m, x), (jks, xcr, jkr, cxc, cjk) in ST.items():
    check("N8", f"{p} {m} {x}: J/K share of the stock SCF", share(p, m, x, "jk_s"), jks)
    check("N8", f"{p} {m} {x}: XC stock/mixed", st_ratio(p, m, x, "xc_s"), xcr)
    check("N8", f"{p} {m} {x}: J/K stock/mixed", st_ratio(p, m, x, "jk_s"), jkr)
    check("N8", f"{p} {m} {x}: cache XC / stock XC", st_ratio(p, m, x, "xc_s", "cache", "stock"), cxc)
    check("N8", f"{p} {m} {x}: cache J/K / stock J/K", st_ratio(p, m, x, "jk_s", "cache", "stock"), cjk)
check("N8", "ST1 r2SCAN celecoxib: XC share of the stock SCF", share("st1", "celecoxib", "r2scan", "xc_s"), "0.85")
check("N8", "ST1 r2SCAN sildenafil: XC share of the stock SCF", share("st1", "sildenafil", "r2scan", "xc_s"), "0.86")
check("N8", "ST1 sildenafil B3LYP: cache XC (s)", lambda: stage("st1", "sildenafil", "b3lyp", "cache", "xc_s"), "10.324")
check("N8", "ST1 sildenafil B3LYP: stock XC (s)", lambda: stage("st1", "sildenafil", "b3lyp", "stock", "xc_s"), "10.319")
for m, v in (("celecoxib", "1.08"), ("sildenafil", "1.16")):
    check("N8", f"ST2 (CONTENDED) {m} B3LYP: J/K stock/mixed", st_ratio("st2", m, "b3lyp", "jk_s"), v)
check_true("N8", "ST1, ST2, ST2r: DF build and eigensolver within 5 % (or 10 ms) of stock in every pair-1 cell",
           lambda: all(abs(stage(p, m, x, a, st) / stage(p, m, x, "stock", st) - 1) <= 0.05
                       or abs(stage(p, m, x, a, st) - stage(p, m, x, "stock", st)) <= 0.010
                       for (p, m, x) in list(ST) + [("st2", "celecoxib", "b3lyp"), ("st2", "sildenafil", "b3lyp")]
                       for a in ("mixed", "cache") for st in ("dfbuild_s", "eig_s")))
check_true("N8", "ST1, ST2r: 'other' (wall minus timed stages) is <= 6 % of every stock SCF",
           lambda: all(1 - sum(share(p, m, x, st)() for st in ("xc_s", "jk_s", "dfbuild_s", "eig_s")) <= 0.06
                       for (p, m, x) in ST))
check_true("N8", "on the slice, sildenafil B3LYP mixed is at tier fp32 with 22 XC entries (15 for stock and "
           "cache); celecoxib is fp64+fp32",
           lambda: all(xone(p, "xc_n", mol="sildenafil", xc="b3lyp", arm=a, pair="1") == n
                       for p in ("st2", "st2r") for a, n in (("mixed", 22), ("stock", 15), ("cache", 15)))
           and {r["tier"] for p in ("st2", "st2r") for r in xrows(p, mol="sildenafil", arm="mixed")} == {"fp32"}
           and {r["tier"] for p in ("st2", "st2r") for r in xrows(p, mol="celecoxib", arm="mixed")} == {"fp64+fp32"})


# CC1: G4 = 4 x T_S / T_C4. T_S = the serial phase's second pass, summed spans; T_C4 = makespan.
def span(r):
    return float(r["t1"]) - float(r["t0"])


def t_s(arm):
    return sum(span(r) for r in xrows("cc1", phase=f"S_{arm}", pair="1"))


def t_c4(arm, mps=False):
    rs = xrows("cc1", phase=f"C4_{arm}" + ("_mps" if mps else ""))
    return max(float(r["t1"]) for r in rs) - min(float(r["t0"]) for r in rs)


for arm, (ts, tc, g, tm, gm) in {"stock": ("51.20", "212.92", "0.96", "134.91", "1.52"),
                                 "mixed": ("29.19", "114.63", "1.02", "66.70", "1.75")}.items():
    check("N8", f"CC1 {arm}: T_S (s)", lambda arm=arm: t_s(arm), ts)
    check("N8", f"CC1 {arm}: T_C4 makespan (s)", lambda arm=arm: t_c4(arm), tc)
    check("N8", f"CC1 {arm}: G4", lambda arm=arm: 4 * t_s(arm) / t_c4(arm), g)
    check("N8", f"CC1 {arm}: T_C4-MPS makespan (s)", lambda arm=arm: t_c4(arm, True), tm)
    check("N8", f"CC1 {arm}: G4-MPS", lambda arm=arm: 4 * t_s(arm) / t_c4(arm, True), gm)
check_true("N8", "CC1 decision rule: max(G4, G4-MPS) >= 1.25 for both arms, so C5 is dropped",
           lambda: all(max(4 * t_s(a) / t_c4(a), 4 * t_s(a) / t_c4(a, True)) >= 1.25 for a in ("stock", "mixed")))
check_true("N8", "CC1: six cells per process; four processes x six cells in every C4 phase; two passes in S",
           lambda: all(len(xrows("cc1", phase=f"C4_{a}" + s)) == 24
                       and {r["proc"] for r in xrows("cc1", phase=f"C4_{a}" + s)} == {"0", "1", "2", "3"}
                       for a in ("stock", "mixed") for s in ("", "_mps"))
           and all(len(xrows("cc1", phase=f"S_{a}", pair=q)) == 6 for a in ("stock", "mixed") for q in ("0", "1")))
check_true("N8", "CC1: every mixed SCF in every phase keeps fp64+fp32 (four processes do not degrade the tier)",
           lambda: {r["tier"] for r in rows("cc1") if r["arm"] == "mixed"} == {"fp64+fp32"})


def _cc1_mps():
    X = _extractor()
    phs = X.extract(_sentinel("cc1").decode(), pod("cc1")["run_id"], unquotable=True)["phases"]
    return {q["phase"]: q.get("mps_server_seen") for q in phs}


check_true("N8", "CC1: MPS engaged (server process seen) in both MPS phases, and only there",
           lambda: _cc1_mps() == {"S_mixed": None, "C4_mixed": None, "C4_mixed_mps": True,
                                  "S_stock": None, "C4_stock": None, "C4_stock_mps": True})

for arm, v in (("stock", "0.56"), ("mixed", "0.73")):
    check("N8", f"CC1 {arm}: MPS uplift, G4-MPS - G4",
          lambda arm=arm: 4 * t_s(arm) / t_c4(arm, True) - 4 * t_s(arm) / t_c4(arm), v)
for x, v in zip(XCS3, ("1.290", "1.336", "1.092")):
    check("N8", f"X3r H100 trio {x} cache/stock", gm_ratio("x3r", x, "cache", "stock", mols=TRIO), v)
check("N8", "X3r H100 wB97M-V trio mixed/stock as a slowdown factor", lambda: 1 / gm_ratio("x3r", "wb97m-v", mols=TRIO)(),
      "1.63")
check("N8", "H100 paracetamol B3LYP median warm wall, any arm, X3 and X3r: lowest (s)",
      lambda: min(med(p, cell("paracetamol", "b3lyp"), a) for p in ("x3", "x3r")
                  for a in ("mixed", "cache", "stock")), "0.60")
check("N8", "H100 paracetamol B3LYP median warm wall, any arm, X3 and X3r: highest (s)",
      lambda: max(med(p, cell("paracetamol", "b3lyp"), a) for p in ("x3", "x3r")
                  for a in ("mixed", "cache", "stock")), "0.99")
check("N8", "CS1: largest cold extra of a functional that did not run first (s)",
      lambda: max(cold_extra(ph, x)() for ph in CS1 for x in ("b3lyp", "wb97m-v")), "1.62")


# PREREG-7 gates, re-derived: every SCF converged, and |E_arm - E_stock| <= 1e-8 Ha against the
# pod's own stock energy for the same molecule and functional.
def x_gate_worst():
    worst, n = 0.0, 0
    for p in EXTRA_PODS:
        ref = {}
        for r in rows(p):
            if r["arm"] == "stock":
                ref.setdefault((r["mol"], r["xc"]), float(r["e"]))
        for r in rows(p):
            if r["arm"] != "stock":
                worst = max(worst, abs(float(r["e"]) - ref[(r["mol"], r["xc"])]))
                n += 1
    if not n:
        raise ValueError("no extra-mode arm rows")
    return worst


check_true("N8", "PREREG-7 extra pods: every SCF converged with no error; every arm within 1e-8 Ha of stock",
           lambda: all(r["converged"] == "True" and not r["error"] for p in EXTRA_PODS for r in rows(p))
           and x_gate_worst() <= 1e-8)
check_true("N8", "CS1 and ST1 (one process, 96 GB card): every mixed and cache SCF is at its full tier",
           lambda: all(r["tier"] == ("fp64" if r["arm"] == "cache" or r["xc"] == "wb97m-v" else "fp64+fp32")
                       for p in ("cs1", "st1") for r in rows(p) if r["arm"] != "stock"))
check_true("N8", "PREREG-7 pods: ST2 is the only CONTENDED one; L12r and L12r2 are DEGRADED, X3r is not",
           lambda: [p for p in ("cs1", "l12r", "l12r2", "x3r", "st1", "st2", "st2r", "cc1")
                    if pod(p)["contended"] == "True"] == ["st2"]
           and [pod(p)["degraded"] for p in ("l12r", "l12r2", "x3r")] == ["True", "True", "False"])


def data_integrity():
    """Every file in data/ must be listed in data/SHA256SUMS with a matching hash, and every listed
    file must exist: a missing, extra or edited data file is a FAIL before any number is read."""
    import hashlib
    path = os.path.join(DATA, "SHA256SUMS")
    if not os.path.isfile(path):
        return [f"missing {path}"]
    want = {}
    with open(path) as fh:
        for ln in fh:
            h, name = ln.split()
            want[name] = h
    have = sorted(f for f in os.listdir(DATA) if f != "SHA256SUMS")
    probs = [f"not in SHA256SUMS: {f}" for f in have if f not in want]
    probs += [f"listed but missing: {f}" for f in want if f not in have]
    for f in have:
        if f in want:
            with open(os.path.join(DATA, f), "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() != want[f]:
                    probs.append(f"hash mismatch: {f}")
    return probs


EXPECTED_CHECKS = 649   # pinned: a check that silently disappears (or appears) is a FAIL


def main():
    integrity = data_integrity()
    for p in integrity:
        print(f"FAIL data  {p}")
    if integrity:
        print("\ndata/ does not match data/SHA256SUMS: no number was checked")
        return 1
    bad, known = [], []
    width = max(len(r[1]) for r in results)
    for claim, what, got, banked, ok in results:
        entry = KNOWN_DISCREPANCIES.get((claim, what))
        if entry is not None:
            want, why = entry
            if ok:
                tag = "STALE"
            elif got == want:
                tag = "DISC"
            else:
                tag = "FAIL"
            (known if tag == "DISC" else bad).append(what)
            print(f"{tag:5s} {claim:3s} {what:<{width}}  {got:>12}  (printed {banked}, data {want}): {why}")
            continue
        if not ok:
            bad.append(what)
        print(f"{'ok  ' if ok else 'FAIL'} {claim:3s} {what:<{width}}  {got:>12}  (banked {banked})")
    missing = [k for k in KNOWN_DISCREPANCIES if k not in {(r[0], r[1]) for r in results}]
    bad += [f"known discrepancy never checked: {k}" for k in missing]
    n = len(results)
    if n != EXPECTED_CHECKS:
        bad.append(f"ran {n} checks, expected {EXPECTED_CHECKS}")
    print(f"\n{n - len(bad) - len(known)}/{n} checks reproduced; {len(known)} known discrepancies "
          f"(listed above, not adjusted); {len(bad)} failures")
    return 1 if bad or not results else 0


if __name__ == "__main__":
    sys.exit(main())
