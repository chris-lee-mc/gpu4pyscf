#!/usr/bin/env python3
"""rfcbench cell runner and cell judge (docs/upstream/PREREG-rfcbench-0-measurand.md).

Two halves in one file:

* `judge_cell` and the gate functions are PURE: they read only the streamed run records, import
  nothing beyond the stdlib, and are what the pod (`rfcbench_pod.py`) calls to decide every cell. The
  host tests drive them with fake records, so every gate is shown able to FAIL.
* `main` (`--group <json>`) runs one xc group of cells on the GPU, in its own process, and streams one
  `RFCBENCH_CELL {json}` line per pair (per run for a downstream cell) as it lands, then one
  `RFCBENCH_CELLEND {json}` per cell. A cap kill therefore loses only unfinished cells. It never
  judges: the verdict is the pod's, computed from the same lines that are banked.

It uses only `MixedPrecision` and `mf.mixed_precision_record` from the fork, and imports nothing from
`scf_specialized` or `scf_headtohead.bench`. Geometries are read with numpy only.

Measurand (PREREG-0 section 1): every run builds a fresh `pyscf.M` and a fresh
`rks.RKS(...).density_fit(auxbasis=...)` and times `kernel()` between two deviceSynchronize calls.
Pair 0 is the cold pair (disclosed); pairs 1..R are the warm pairs. Within a pair the arms run in
`rfcbench_sets.ARM_ORDER` order, stock last.
"""
from __future__ import annotations

import gc
import json
import math
import numbers
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rfcbench_sets as S  # noqa: E402

DE_TOL = 1e-8
CYCLE_TOL = 1
VV10_CERT_REL = 1e-10
VV10_CERT_DENLC = 1e-11
# PREREG-4 gates, fixed now.
DS_GRAD_TOL = 1e-5        # Ha/Bohr
DS_DIP_TOL = 1e-4         # Debye
DS_E_TOL = 1e-8           # Ha
DS_RUNS = ("R", "S", "Sp", "M")      # Sp = S' (init_guess='atom')

# Cell statuses (PREREG-0 section 5) and fit outcomes (section 4).
OK, GATE_FAIL, CACHE_FALLBACK, NOT_TREATED, ERROR, CAP_KILLED = (
    "OK", "GATE_FAIL", "CACHE_FALLBACK", "NOT_TREATED", "ERROR", "CAP_KILLED")
STATUSES = (OK, GATE_FAIL, CACHE_FALLBACK, NOT_TREATED, ERROR, CAP_KILLED)
TREATED = "TREATED"
FIT_RANK = {TREATED: 0, CACHE_FALLBACK: 1, NOT_TREATED: 2}
# At most this many problems per cell are printed (the rest are counted); sizes the sentinel tail.
MAX_PRINTED_PROBLEMS = 12


def policy_kwargs(arm: str, xc: str):
    """The MixedPrecision kwargs of an arm, or None for stock. PREREG-0 section 1, verbatim."""
    if arm == "stock":
        return None
    if arm == "cache":
        return {"ao_cache_fp64": True}
    return {"r2scan": {"xc": True, "ao_cache_fp64": True},
            "b3lyp": {"xc": True, "k": True, "xc_switch_tol": 3e-4, "k_switch_tol": 1e-3,
                      "ao_cache_fp64": True},
            "wb97m-v": {"vv10": True, "ao_cache_fp64": True}}[xc]


# PREREG-rfcbench-8: the campaign mixed policy plus ONE opt-in field of the 09271907 tree.
POLICY_VARIANTS = ("mixed_dreset", "mixed_ws0", "mixed_wsw")


def variant_kwargs(variant: str, xc: str):
    """The MixedPrecision kwargs of a PREREG-8 policy variant: `mixed` is the campaign policy
    (`policy_kwargs("mixed", xc)`); `mixed_dreset` adds diis_reset_at_switch=True (Md);
    `mixed_ws0` adds warm_start_gorb=1e-300 (M0, record only); `mixed_wsw` adds warm_start_gorb=1e-2
    (MW). None (stock) stays None."""
    if variant is None:
        return None
    base = policy_kwargs("mixed", xc)
    extra = {"mixed": {}, "mixed_dreset": {"diis_reset_at_switch": True},
             "mixed_ws0": {"warm_start_gorb": S.WARM_GORB_RECORD_ONLY},
             "mixed_wsw": {"warm_start_gorb": S.WARM_GORB_MW}}[variant]
    return dict(base, **extra)


# --------------------------------------------------------------------------------------------- #
# compact call lists: run-length "fp32*4,fp64*6" (the sentinel must stay small)
# --------------------------------------------------------------------------------------------- #

def rle(calls) -> str:
    out, prev, n = [], None, 0
    for c in list(calls):
        if c == prev:
            n += 1
            continue
        if prev is not None:
            out.append(f"{prev}*{n}")
        prev, n = c, 1
    if prev is not None:
        out.append(f"{prev}*{n}")
    return ",".join(out)


def unrle(s) -> list:
    """Inverse of `rle`. Raises on anything malformed (the caller turns that into a problem)."""
    if not isinstance(s, str):
        raise TypeError(f"call list {s!r} is not a string")
    out = []
    for part in s.split(",") if s else []:
        tok, _, n = part.partition("*")
        if not tok or not n.isdigit() or int(n) < 1:
            raise ValueError(f"call list part {part!r} is malformed")
        out += [tok] * int(n)
    return out


# --------------------------------------------------------------------------------------------- #
# gates -- fail closed: a missing key, a wrong type or a NaN is a problem, never a pass
# --------------------------------------------------------------------------------------------- #

def _count(x):
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _finite_nonneg(x):
    return (isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)
            and x >= 0)


def _finite_pos(x):
    return (isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)
            and x > 0)


def settings_problems(tag, run, cell, kind_settings):
    """PREREG-0 section 1: the settings READ BACK off the built objects."""
    probs = []
    try:
        st = run["settings"]
        if st["grid_level"] != S.GRID_LEVEL:
            probs.append(f"{tag}: grids.level {st['grid_level']!r} != {S.GRID_LEVEL}")
        if st["prune_none"] is not True:
            probs.append(f"{tag}: grids.prune is not None")
        if not (_count(st["ngrids"]) and st["ngrids"] > 0):
            probs.append(f"{tag}: ngrids {st['ngrids']!r} not a positive count")
        if st["conv_tol"] != kind_settings["conv_tol"]:
            probs.append(f"{tag}: conv_tol {st['conv_tol']!r} != {kind_settings['conv_tol']}")
        if st["max_cycle"] != kind_settings["max_cycle"]:
            probs.append(f"{tag}: max_cycle {st['max_cycle']!r} != {kind_settings['max_cycle']}")
        if st["conv_tol_grad"] != kind_settings["conv_tol_grad"]:
            probs.append(f"{tag}: conv_tol_grad {st['conv_tol_grad']!r} != "
                         f"{kind_settings['conv_tol_grad']!r}")
        if str(st["auxbasis"]).lower() != cell["aux"]:
            probs.append(f"{tag}: auxbasis {st['auxbasis']!r} != {cell['aux']}")
        if str(st["basis"]).lower() != S.BASES[cell["basis"]]:
            probs.append(f"{tag}: basis {st['basis']!r} != {S.BASES[cell['basis']]}")
        if not (_count(st["naux"]) and st["naux"] > 0 and _count(st["nao"]) and st["nao"] > 0):
            probs.append(f"{tag}: naux/nao {st['naux']!r}/{st['nao']!r} not positive counts")
    except Exception as e:
        probs.append(f"{tag}: settings not recorded: {type(e).__name__}: {e}"[:300])
    return probs


def kind_settings(kind):
    """What `settings_problems` requires per run kind. None = the library default (not set)."""
    base = {"conv_tol": S.CONV_TOL, "max_cycle": S.MAX_CYCLE, "conv_tol_grad": None}
    if kind == "bridge":
        return dict(base, conv_tol_grad=S.BRIDGE_CONV_TOL_GRAD)
    if kind == "R":
        return {"conv_tol": S.REF_CONV_TOL, "max_cycle": S.REF_MAX_CYCLE,
                "conv_tol_grad": S.REF_CONV_TOL_GRAD}
    return base


# The AO cache notes the fork writes (gpu4pyscf/dft/mixed_precision.py `_build_ao_cache` and
# `_SCFState.drop_ao_cache`, VALIDATED_FORK_SHA 63af0568). A fallback is classified ONLY from one of
# these fit-based reasons; any other note (a prediction that disagreed with block_loop, a cache
# dropped mid-SCF, an overridden nr_rks, an empty or truncated note) is not a fit outcome and fails.
_GIB = r"[0-9]+\.[0-9]{2} GiB"
NOTE_BUILT = re.compile(rf"^built: {_GIB}, tier (fp64\+fp32|fp64|fp32)$")
NOTE_BUILT_NO64 = re.compile(rf"^built: {_GIB}, tier fp32 \(no FP64 copy: "
                             r"(FP64 copy and FP32 mirror do not fit|[0-9]+ empty grid blocks)\)$")
NOTE_NOT_BUILT = re.compile(rf"^not built: (FP64 copy does not fit|[0-9]+ empty grid blocks) "
                            rf"\(needs {_GIB}, budget {_GIB}\); FP64 XC as stock$")
NOTE_DID_NOT_FIT = re.compile(rf"^did not fit: needs > {_GIB}, budget {_GIB}; XC stays FP64$")


def classify_fit(arm, xc, r):
    """PREREG-0 section 2 fit outcome of one treated run: TREATED, CACHE_FALLBACK or NOT_TREATED,
    read from the tier AND the fork's own note. Raises ValueError on anything else (the caller makes
    it a gate problem), so a fallback is never inferred from an absence.

    mixed r2SCAN/B3LYP: tier 'fp64+fp32' with a plain 'built' note is TREATED; tier 'fp32' with
    'no FP64 copy: <fit reason>' is CACHE_FALLBACK; tier None with 'did not fit: ...; XC stays FP64'
    is NOT_TREATED. mixed wB97M-V and every cache arm: tier 'fp64' with a plain 'built' note is
    TREATED; tier None with 'not built: <fit reason> ...; FP64 XC as stock' is CACHE_FALLBACK."""
    tier, note = r["ao_cache_tier"], r["ao_cache"]
    if not isinstance(note, str):
        raise ValueError(f"AO cache note {note!r} is not a string")
    m = NOTE_BUILT.match(note)
    xck = arm == "mixed" and xc != S.VV10_XC
    want = "fp64+fp32" if xck else "fp64"
    if tier == want and m and m.group(1) == want:
        return TREATED
    if xck and tier == "fp32" and NOTE_BUILT_NO64.match(note):
        return CACHE_FALLBACK
    if xck and tier is None and NOTE_DID_NOT_FIT.match(note):
        return NOT_TREATED
    if not xck and tier is None and NOTE_NOT_BUILT.match(note):
        return CACHE_FALLBACK
    raise ValueError(f"AO cache tier {tier!r} with note {note!r} is neither the expected tier nor "
                     f"a fit-based fallback")


def accuracy_problems(tag, run, stock):
    """Arm vs the stock run of the SAME pair: both converged, |dE| <= 1e-8 Ha, |dcycles| <= 1."""
    probs = []
    try:
        if run["converged"] is not True or stock["converged"] is not True:
            probs.append(f"{tag}: not converged (arm {run['converged']!r}, "
                         f"stock {stock['converged']!r})")
        de = abs(float(run["e"]) - float(stock["e"]))
        if not de <= DE_TOL:
            probs.append(f"{tag}: |dE| {de:.3e} > {DE_TOL:g}")
        if not (_count(run["cycles"]) and _count(stock["cycles"])
                and abs(run["cycles"] - stock["cycles"]) <= CYCLE_TOL):
            probs.append(f"{tag}: cycles {stock['cycles']!r} (stock) vs {run['cycles']!r}")
    except Exception as e:
        probs.append(f"{tag}: malformed run record: {type(e).__name__}: {e}"[:300])
    return probs


def tail_problems(tag, r):
    try:
        if r["fp64_tail"] is not True:
            return [f"{tag}: no FP64 tail ({r['fp64_tail']!r})"]
    except Exception as e:
        return [f"{tag}: malformed record: {type(e).__name__}: {e}"[:300]]
    return []


def stock_arith_problems(tag, r):
    """The cache arm ran stock arithmetic only: no fp32/df64 in xc, k or vv10."""
    probs = []
    try:
        for comp in ("xc", "k", "vv10"):
            calls = unrle(r[comp])
            bad = sorted({c for c in calls if c in ("fp32", "df64")})
            if bad or not calls:
                probs.append(f"{tag}: cache arm {comp} calls {r[comp]!r} are not all stock FP64")
    except Exception as e:
        probs.append(f"{tag}: malformed record: {type(e).__name__}: {e}"[:300])
    return probs


def xc_k_problems(tag, xc, r):
    """Treatment really ran (mixed r2SCAN/B3LYP): >= 1 FP32 XC call, the last two XC calls FP64;
    B3LYP also >= 1 FP32 K call, the last two K calls FP64, and k_full_rebuild_call set."""
    probs = []
    try:
        xcs, ks = unrle(r["xc"]), unrle(r["k"])
        if xcs.count("fp32") < 1:
            probs.append(f"{tag}: no FP32 XC call ran ({r['xc']}; {r.get('ao_cache')})")
        if len(xcs) < 2 or xcs[-2:] != ["fp64", "fp64"]:
            probs.append(f"{tag}: XC did not end on two FP64 calls ({r['xc']})")
        if xc == "b3lyp":
            if ks.count("fp32") < 1:
                probs.append(f"{tag}: no FP32 K call ran ({r['k']})")
            if len(ks) < 2 or ks[-2:] != ["fp64", "fp64"]:
                probs.append(f"{tag}: K did not end on two FP64 calls ({r['k']})")
            if not _count(r["k_full_rebuild_call"]):
                probs.append(f"{tag}: k_full_rebuild_call {r['k_full_rebuild_call']!r} not set")
    except Exception as e:
        probs.append(f"{tag}: malformed record: {type(e).__name__}: {e}"[:300])
    return probs


def vv10_problems(tag, r):
    """The wB97M-V gates of g4pport_trio.vv10_problems, verbatim in substance (the convergence /
    |dE| / cycle part is `accuracy_problems`)."""
    probs = []
    try:
        calls = unrle(r["vv10"])
        if not (r["vv10_n_fp32"] >= 1 and calls.count("fp32") == r["vv10_n_fp32"]):
            probs.append(f"{tag}: FP32 VV10 not exercised (n_fp32={r['vv10_n_fp32']}, {r['vv10']})")
        if not (r["vv10_n_df64"] >= 1 and calls.count("df64") == r["vv10_n_df64"]):
            probs.append(f"{tag}: df64 VV10 not exercised (n_df64={r['vv10_n_df64']}, {r['vv10']})")
        if calls[-2:] != ["df64", "df64"]:
            probs.append(f"{tag}: VV10 did not end on two df64 calls ({calls[-2:]})")
        if "fp64" in calls:
            probs.append(f"{tag}: an FP64 VV10 call ran with vv10 on ({r['vv10']})")
        cert = r["vv10_cert"]
        if not isinstance(cert, dict):
            probs.append(f"{tag}: no VV10 certificate ({cert!r})")
        else:
            rel = cert["rel"]
            if cert["ok"] is not True:
                probs.append(f"{tag}: VV10 certificate not ok ({cert.get('error')})")
            if not (_count(cert["call"]) and cert["call"] == len(calls)):
                probs.append(f"{tag}: VV10 certificate on call {cert['call']!r}, not the last "
                             f"({len(calls)})")
            if not (isinstance(rel, dict) and sorted(rel) == ["E", "U", "W"]
                    and all(_finite_nonneg(rel[k]) and rel[k] <= VV10_CERT_REL for k in rel)):
                probs.append(f"{tag}: VV10 certificate rel {rel} > {VV10_CERT_REL:g}")
            if not (_finite_nonneg(cert["denlc"]) and cert["denlc"] <= VV10_CERT_DENLC):
                probs.append(f"{tag}: VV10 certificate denlc {cert['denlc']} > {VV10_CERT_DENLC:g}")
            if not _finite_pos(cert["wall_s"]):
                probs.append(f"{tag}: VV10 certificate wall {cert['wall_s']!r} not finite > 0")
    except Exception as e:
        probs.append(f"{tag}: malformed VV10 record: {type(e).__name__}: {e}"[:300])
    return probs


def cache_problems(tag, expect_tier, r):
    """The AO cache really ran (PORT-PLAN-ao-cache gates (g)-(h), the logic of
    g4pport_trio.cache_problems): xc_fp64_stock == 0; the expected tier; for 'fp64+fp32' at least two
    cached FP64 XC calls and the FP32 mirror released on the XC switch call; for 'fp64' every XC call
    from the cache."""
    probs = []
    try:
        tier, cached, stock = r["ao_cache_tier"], r["xc_fp64_cached"], r["xc_fp64_stock"]
        n_calls = len(unrle(r["xc"]))
        if not (_count(cached) and _count(stock)):
            return [f"{tag}: AO cache counts not integers ({cached!r}, {stock!r})"]
        if stock != 0:
            probs.append(f"{tag}: {stock} FP64 XC calls ran stock, not from the AO cache")
        if tier != expect_tier:
            probs.append(f"{tag}: AO cache tier {tier!r}, not {expect_tier!r} ({r.get('ao_cache')})")
        if expect_tier == "fp64":
            if not (n_calls >= 1 and cached == n_calls):
                probs.append(f"{tag}: {cached} of {n_calls} XC calls from the AO cache")
        else:
            if not cached >= 2:
                probs.append(f"{tag}: {cached} FP64 XC calls from the AO cache, not >= 2")
            rel, sw = r["ao_cache_mirror_released_call"], r["xc_switch_call"]
            if not (_count(rel) and _count(sw) and rel == sw):
                probs.append(f"{tag}: FP32 mirror released on call {rel!r}, XC switched on {sw!r}")
    except Exception as e:
        probs.append(f"{tag}: malformed AO cache record: {type(e).__name__}: {e}"[:300])
    return probs


def expected_tier(arm, xc):
    return "fp64+fp32" if (arm == "mixed" and xc != S.VV10_XC) else "fp64"


def treated_run_problems(tag, arm, xc, rec, fit):
    """Every gate on one mixed or cache run's record, given its fit outcome. Accuracy against stock is
    separate (`accuracy_problems`) and always applies.
      - fp64_tail: always;
      - cache arm: stock arithmetic only, always;
      - mixed arm, treatment really ran: unless NOT_TREATED (then there was no treatment to gate);
      - the cache really ran: only when TREATED (a fallback is classified, not gated twice)."""
    probs = tail_problems(tag, rec)
    if arm == "cache":
        probs += stock_arith_problems(tag, rec)
    elif fit != NOT_TREATED:
        probs += vv10_problems(tag, rec) if xc == S.VV10_XC else xc_k_problems(tag, xc, rec)
    if fit == TREATED:
        probs += cache_problems(tag, expected_tier(arm, xc), rec)
    return probs


def _worst_fit(fits):
    return max(fits, key=lambda f: FIT_RANK[f]) if fits else TREATED


def _status(mode, structural, gate, fit):
    if structural:
        return ERROR
    if gate:
        return GATE_FAIL
    if fit != TREATED:
        return fit
    return OK


def policy_repr_table(mixed_precision_cls, variants=False):
    """`{"arm|xc": repr(MixedPrecision(**policy_kwargs(arm, xc)))}` for every treated arm, built from
    the REAL class (on the pod, in the probe process). The judge compares every run's recorded
    `policy` against it, so a run whose record was not produced by the arm's policy cannot pass.
    `variants=True` returns the PREREG-8 variants (`mixed_dreset|xc`, ...) instead; on a tree without
    those fields the class raises, and the caller records that."""
    if variants:
        return {f"{v}|{xc}": repr(mixed_precision_cls(**variant_kwargs(v, xc)))
                for v in POLICY_VARIANTS for xc in S.XCS}
    return {f"{arm}|{xc}": repr(mixed_precision_cls(**policy_kwargs(arm, xc)))
            for arm in ("mixed", "cache") for xc in S.XCS}


def observation_problems(tag, run, arm, xc, expect):
    """R6: what ran is read off the built objects, never inferred from the arm label.
    stock: `mf.mixed_precision is None` and no `mixed_precision_record`; treated: the record's
    `policy` is the repr of the arm's own MixedPrecision."""
    try:
        if arm == "stock":
            if run["mp_none"] is not True or run["rec_none"] is not True or "rec" in run:
                return [f"{tag}: the stock run carried a policy or a record "
                        f"(mp_none={run.get('mp_none')!r}, rec_none={run.get('rec_none')!r})"]
            return []
        if not isinstance(expect, dict) or f"{arm}|{xc}" not in expect:
            return [f"{tag}: no expected policy repr to judge the record against"]
        got = (run.get("rec") or {}).get("policy")
        if got != expect[f"{arm}|{xc}"]:
            return [f"{tag}: recorded policy {got!r} is not {expect[f'{arm}|{xc}']!r}"]
    except Exception as e:
        return [f"{tag}: malformed run record: {type(e).__name__}: {e}"[:300]]
    return []


def consistency_problems(key, runs, atom_run=None):
    """ngrids / naux / nao identical across every run of a cell; init_guess identical across every
    run except `atom_run` (downstream S'), which must read back 'atom' while no other run does."""
    probs = []
    try:
        for f in ("ngrids", "naux", "nao"):
            vals = {repr(r["settings"][f]) for r in runs}
            if len(vals) != 1:
                probs.append(f"{key}: {f} differs across runs ({sorted(vals)})")
        others = {repr(r["settings"]["init_guess"]) for r in runs
                  if atom_run is None or r.get("run") != atom_run}
        if len(others) != 1 or "'atom'" in others:
            probs.append(f"{key}: init_guess {sorted(others)} not one non-atom value")
        if atom_run is not None:
            sp = [r for r in runs if r.get("run") == atom_run]
            if len(sp) != 1 or sp[0]["settings"]["init_guess"] != "atom":
                probs.append(f"{key}: {atom_run} did not read back init_guess='atom'")
    except Exception as e:
        probs.append(f"{key}: settings unreadable for the consistency check: "
                     f"{type(e).__name__}: {e}"[:300])
    return probs


def expected_pairs(cell, repeats):
    return [1] if cell["kind"] == "bridge" else list(range(0, int(repeats) + 1))


def judge_speed(mode, cell, runs, repeats, expect=None):
    """Structural, accuracy, treatment and cache gates over every pair of a speed or bridge cell."""
    structural, gate, fits = [], [], []
    key = S.cell_key(cell)
    arms = list(cell["arms"])
    by_pair = {}
    for r in runs:
        by_pair.setdefault(r.get("pair"), []).append(r)
    want = expected_pairs(cell, repeats)
    if sorted(p for p in by_pair if p is not None) != want or None in by_pair:
        structural.append(f"{key}: pairs {sorted(map(str, by_pair))}, expected {want}")
    ks = kind_settings(cell["kind"])
    for p in want:
        got = by_pair.get(p, [])
        order = [r.get("arm") for r in got]
        if order != arms:
            structural.append(f"{key}: pair {p} ran arms {order}, expected {arms} (stock last)")
            continue
        stock = got[-1]
        for r in got:
            tag = f"{key} pair {p} {r['arm']}"
            gate += settings_problems(tag, r, cell, ks)
            if r.get("error"):
                structural.append(f"{tag}: {r['error']}")
                continue
            gate += observation_problems(tag, r, r["arm"], cell["xc"], expect)
            if r["arm"] == "stock":
                continue
            if stock.get("error"):
                continue
            gate += accuracy_problems(tag, r, stock)
            rec = r.get("rec")
            if not isinstance(rec, dict):
                gate.append(f"{tag}: no mixed_precision_record")
                continue
            try:
                fit = classify_fit(r["arm"], cell["xc"], rec)
            except Exception as e:
                gate.append(f"{tag}: fit unclassifiable: {type(e).__name__}: {e}"[:300])
                continue
            fits.append(fit)
            gate += treated_run_problems(tag, r["arm"], cell["xc"], rec, fit)
    if not structural:
        gate += consistency_problems(key, runs)
    return structural, gate, _worst_fit(fits)


def _maxabs_diff(a, b):
    """max |a - b| over two equal-shape nested float lists; raises on a shape mismatch or a
    non-finite entry (NaN is never 0.0)."""
    fa, fb = _flat(a), _flat(b)
    if len(fa) != len(fb) or not fa:
        raise ValueError(f"shape mismatch or empty ({len(fa)} vs {len(fb)})")
    out = 0.0
    for x, y in zip(fa, fb):
        d = abs(float(x) - float(y))
        if not math.isfinite(d):
            raise ValueError("non-finite entry")
        out = max(out, d)
    return out


def _flat(x):
    if isinstance(x, (list, tuple)):
        return [v for e in x for v in _flat(e)]
    return [x]


def judge_downstream(mode, cell, runs, expect=None):
    """PREREG-4: R, S, S' and M, in that order; gates on M against the reference (R, or S when R did
    not converge, disclosed as reference=S); plus PREREG-0's accuracy gates of M against S and the
    mixed-arm treatment and cache gates on M's record."""
    structural, gate, fits = [], [], []
    key = S.cell_key(cell)
    order = [r.get("run") for r in runs]
    readings = {"reference": None, "rho_g": None, "rho_mu": None}
    if order != list(DS_RUNS):
        structural.append(f"{key}: runs {order}, expected {list(DS_RUNS)}")
        return structural, gate, TREATED, readings
    R, Sr, Sp, M = runs
    for r in runs:
        tag = f"{key} {r['run']}"
        gate += settings_problems(tag, r, cell, kind_settings("R" if r["run"] == "R" else "speed"))
        if r.get("error"):
            structural.append(f"{tag}: {r['error']}")
    if structural:
        return structural, gate, TREATED, readings
    for r in runs:
        gate += observation_problems(f"{key} {r['run']}", r, "mixed" if r["run"] == "M" else "stock",
                                     cell["xc"], expect)
    gate += consistency_problems(key, runs, atom_run="Sp")
    for r in runs:
        if r.get("state_none") is not True:
            gate.append(f"{key} {r['run']}: _mixed_precision_state was not None before the gradient")
    ref = R if R.get("converged") is True else Sr
    readings["reference"] = "R" if ref is R else "S"
    if ref is Sr and Sr.get("converged") is not True:
        gate.append(f"{key}: neither R nor S converged; no reference")
    gate += accuracy_problems(f"{key} M vs S", M, Sr)
    try:
        dg = _maxabs_diff(M["grad"], ref["grad"])
        dmu = _maxabs_diff(M["dip"], ref["dip"])
        de = abs(float(M["e"]) - float(ref["e"]))
        readings.update(max_dg=dg, max_dmu=dmu, de=de)
        if not dg <= DS_GRAD_TOL:
            gate.append(f"{key}: max|g_M - g_{readings['reference']}| {dg:.3e} > {DS_GRAD_TOL:g}")
        if not dmu <= DS_DIP_TOL:
            gate.append(f"{key}: max|mu_M - mu_{readings['reference']}| {dmu:.3e} > {DS_DIP_TOL:g}")
        if not de <= DS_E_TOL:
            gate.append(f"{key}: |E_M - E_{readings['reference']}| {de:.3e} > {DS_E_TOL:g}")
    except Exception as e:
        gate.append(f"{key}: gradient/dipole/energy unreadable: {type(e).__name__}: {e}"[:300])
    # Disclosed readings: rho against S' (the alternative-trajectory noise comparator).
    try:
        if Sp.get("converged") is True:
            dgs = _maxabs_diff(Sp["grad"], ref["grad"])
            dmus = _maxabs_diff(Sp["dip"], ref["dip"])
            readings.update(max_dg_sp=dgs, max_dmu_sp=dmus)
            readings["rho_g"] = readings["max_dg"] / dgs if dgs > 0 else None
            readings["rho_mu"] = readings["max_dmu"] / dmus if dmus > 0 else None
    except Exception:
        pass
    rec = M.get("rec")
    if not isinstance(rec, dict):
        gate.append(f"{key} M: no mixed_precision_record")
    else:
        try:
            fit = classify_fit("mixed", cell["xc"], rec)
            fits.append(fit)
            gate += treated_run_problems(f"{key} M", "mixed", cell["xc"], rec, fit)
        except Exception as e:
            gate.append(f"{key} M: fit unclassifiable: {type(e).__name__}: {e}"[:300])
    return structural, gate, _worst_fit(fits), readings


# --------------------------------------------------------------------------------------------- #
# PREREG-rfcbench-8 A1/A2: eight-arm attribution
# --------------------------------------------------------------------------------------------- #
ATTRIB_E_TOL = 1e-8                    # |E_arm - E_R'| (Ha)
ATTRIB_MIXED = ("M", "Mt", "Md")
# The pairwise max-abs gradient / dipole deltas the predictions P-A .. P-E read (disclosed only).
ATTRIB_PAIRS = (("S2", "S"), ("St", "Rp"), ("Mt", "Rp"), ("Mt", "St"), ("Md", "S"), ("M", "S"),
                ("S", "Rp"))


def attrib_settings(arm):
    """What `settings_problems` requires of an attrib arm (S.ATTRIB_PLAN)."""
    return dict(S.ATTRIB_PLAN[arm][1])


def _vector_problems(tag, x, rows=None, width=None):
    """A gradient (N x 3) or dipole (3) present, rectangular and finite. Fail closed."""
    try:
        if rows is None:
            ok = (isinstance(x, list) and len(x) == width
                  and all(_finite_real(v) for v in x))
        else:
            ok = (isinstance(x, list) and len(x) >= 1
                  and all(isinstance(r, list) and len(r) == 3 and all(_finite_real(v) for v in r)
                          for r in x))
    except Exception:
        ok = False
    return [] if ok else [f"{tag}: {'gradient' if rows else 'dipole'} missing or malformed"]


def _finite_real(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def judge_attrib(mode, cell, runs, expect=None):
    """PREREG-8 A1/A2 gates, plus PREREG-0's observation discipline (settings read back per arm, the
    recorded policy is the arm's own, a stock arm carries no policy, ngrids/naux/nao and a non-atom
    init_guess shared by every arm, no mixed state alive before the stock gradient) and the treated-run
    gates of `treated_run_problems` on M, Mt and Md. Returns (structural, gate, fit, readings)."""
    structural, gate, fits = [], [], []
    key = S.cell_key(cell)
    xc = cell["xc"]
    readings = {"cycles": {}, "wall_s": {}, "grad_wall_s": {}, "dg": {}, "dmu": {}, "de_vs_Rp": {}}
    order = [r.get("run") for r in runs]
    if order != list(S.ATTRIB_ARMS):
        structural.append(f"{key}: runs {order}, expected {list(S.ATTRIB_ARMS)}")
        return structural, gate, TREATED, readings
    by = {r["run"]: r for r in runs}
    for r in runs:
        tag = f"{key} {r['run']}"
        if r.get("error"):
            structural.append(f"{tag}: {r['error']}")
            continue
        gate += settings_problems(tag, r, cell, attrib_settings(r["run"]))
        variant = S.ATTRIB_PLAN[r["run"]][0]
        gate += observation_problems(tag, r, variant or "stock", xc, expect)
        if r.get("converged") is not True:
            gate.append(f"{tag}: not converged ({r.get('converged')!r})")
        if r.get("state_none") is not True:
            gate.append(f"{tag}: _mixed_precision_state was not None before the gradient")
        gate += _vector_problems(tag, r.get("grad"), rows=True)
        gate += _vector_problems(tag, r.get("dip"), width=3)
        readings["cycles"][r["run"]] = r.get("cycles")
        readings["wall_s"][r["run"]] = r.get("wall_s")
        readings["grad_wall_s"][r["run"]] = r.get("grad_wall_s")
    if structural:
        return structural, gate, TREATED, readings
    gate += consistency_problems(key, runs)
    try:
        e_ref = float(by["Rp"]["e"])
        for arm in S.ATTRIB_ARMS:
            de = abs(float(by[arm]["e"]) - e_ref)
            readings["de_vs_Rp"][arm] = de
            if not de <= ATTRIB_E_TOL:
                gate.append(f"{key} {arm}: |E - E_R'| {de:.3e} > {ATTRIB_E_TOL:g}")
    except Exception as e:
        gate.append(f"{key}: energies unreadable: {type(e).__name__}: {e}"[:300])
    for arm in ATTRIB_MIXED:
        tag = f"{key} {arm}"
        rec = by[arm].get("rec")
        if not isinstance(rec, dict):
            gate.append(f"{tag}: no mixed_precision_record")
            continue
        try:
            fit = classify_fit("mixed", xc, rec)
        except Exception as e:
            gate.append(f"{tag}: fit unclassifiable: {type(e).__name__}: {e}"[:300])
            continue
        fits.append(fit)
        if fit != TREATED:
            gate.append(f"{tag}: fit {fit}, not TREATED (PREREG-8 gate)")
        gate += treated_run_problems(tag, "mixed", xc, rec, fit)
        fields = rec.get("geo_fields")
        if not (isinstance(fields, list) and "diis_reset_call" in fields):
            gate.append(f"{tag}: the record carries no diis_reset_call field ({fields!r}); a reset "
                        f"(or its absence) must be observed, not inferred")
            continue
        reset = rec.get("diis_reset_call")
        if arm == "Md" and reset is None:
            gate.append(f"{tag}: diis_reset_at_switch=True but no diis_reset_call recorded")
        if arm != "Md" and reset is not None:
            gate.append(f"{tag}: a DIIS reset was recorded ({reset!r}) on a policy without one")
    for a, b in ATTRIB_PAIRS:
        try:
            readings["dg"][f"{a}-{b}"] = _maxabs_diff(by[a]["grad"], by[b]["grad"])
            readings["dmu"][f"{a}-{b}"] = _maxabs_diff(by[a]["dip"], by[b]["dip"])
        except Exception:
            readings["dg"][f"{a}-{b}"] = readings["dmu"][f"{a}-{b}"] = None
    return structural, gate, _worst_fit(fits), readings


# --------------------------------------------------------------------------------------------- #
# PREREG-rfcbench-8 G1: geometry optimisations through the gradient scanner
# --------------------------------------------------------------------------------------------- #
GEO_E_TOL = 1e-6                       # |E_stock(endpoint) - E_S(final)| (Ha)
GEO_MIXED = ("M0", "MW")
BOHR_TO_A = 0.52917721092


def grad_norms(g):
    """geomeTRIC's Cartesian convention: RMS and max over atoms of the per-atom norm |g_a|. Pure.
    Raises on a malformed or non-finite gradient (the caller makes it a problem)."""
    rows = [[float(v) for v in r] for r in g]
    if not rows or any(len(r) != 3 for r in rows):
        raise ValueError("gradient is not N x 3")
    n = [math.sqrt(sum(v * v for v in r)) for r in rows]
    if not all(math.isfinite(x) for x in n):
        raise ValueError("non-finite gradient")
    return {"grms": math.sqrt(sum(x * x for x in n) / len(n)), "gmax": max(n)}


def kabsch_rmsd(a, b):
    """Kabsch-aligned RMSD between two N x 3 coordinate sets (centroids removed, optimal proper
    rotation, reflection excluded). numpy, imported lazily (the judge's module level is stdlib)."""
    import numpy as np
    p = np.asarray(a, dtype=float)
    q = np.asarray(b, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3 or not p.shape[0]:
        raise ValueError(f"shape mismatch {p.shape} vs {q.shape}")
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    diff = (rot @ p.T).T - q
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# The warm-start record of fork 09271907 (`hf._mixed_precision_warm_start`) is exactly
# {'tol': float, 'supplied': bool, 'gorb': float | None, 'fp64': bool}. Disclosure only, never a gate.
WS_NORM_KEYS = ("gorb",)
WS_TRIGGER_KEYS = ("fp64",)


def warm_start_fields(ws):
    """(initial orbital-gradient norm, triggered) out of a warm-start record, each None when the
    record does not carry it in a recognised form."""
    if not isinstance(ws, dict):
        return None, None
    norm = next((float(ws[k]) for k in WS_NORM_KEYS if _finite_real(ws.get(k))), None)
    trig = next((ws[k] for k in WS_TRIGGER_KEYS if isinstance(ws.get(k), bool)), None)
    return norm, trig


def _steps_problems(tag, run, mixed):
    """Every SCF of every step converged; a mixed SCF ended on an FP64 tail; there is one SCF record
    per optimisation step at least (a step whose SCF was never observed is not a converged one)."""
    probs = []
    steps, recs = run.get("steps"), run.get("step")
    if not (_count(steps) and steps >= 1 and isinstance(recs, list) and len(recs) >= steps):
        return [f"{tag}: {len(recs) if isinstance(recs, list) else recs!r} SCF records for "
                f"{steps!r} optimisation steps"]
    for i, st in enumerate(recs):
        if not isinstance(st, dict):
            probs.append(f"{tag} step {i}: malformed SCF record")
            continue
        if st.get("ok") is not True:
            probs.append(f"{tag} step {i}: SCF not converged ({st.get('ok')!r})")
        if mixed and st.get("tail") is not True:
            probs.append(f"{tag} step {i}: mixed SCF did not end on an FP64 tail "
                         f"({st.get('tail')!r})")
        if not mixed and st.get("rec_none") is not True:
            probs.append(f"{tag} step {i}: the stock SCF carried a mixed-precision record")
    return probs


def _warm(recs, field):
    return [st.get(field) if isinstance(st, dict) else None for st in recs[1:]]


def _ratio(a, b):
    try:
        return a / b if (a is not None and b) else None
    except Exception:
        return None


def judge_geoopt(mode, cell, runs, expect=None):
    """PREREG-8 G1 gates: the three optimisations converged within GEO_MAXSTEPS steps; every SCF of
    every step converged and every mixed one ended on an FP64 tail; at the M0 and MW endpoints the
    fresh stock gradient meets grms <= 3e-4 and gmax <= 4.5e-4 and |E_stock(endpoint) - E_S(final)|
    <= 1e-6 Ha. Plus PREREG-0's observation discipline: settings read back, each arm's recorded
    policy is its own (every step), stock carries none. The fit outcome is DISCLOSED here (PREREG-8
    gates no fit on G1). Returns (structural, gate, fit, readings)."""
    structural, gate, fits = [], [], []
    key = S.cell_key(cell)
    xc = cell["xc"]
    readings = {"steps": {}, "fit": {}}
    order = [r.get("run") for r in runs]
    if order != list(S.GEOOPT_ARMS):
        structural.append(f"{key}: runs {order}, expected {list(S.GEOOPT_ARMS)}")
        return structural, gate, TREATED, readings
    by = {r["run"]: r for r in runs}
    for r in runs:
        arm = r["run"]
        tag = f"{key} {arm}"
        if r.get("error"):
            structural.append(f"{tag}: {r['error']}")
            continue
        variant = S.GEOOPT_VARIANT[arm]
        gate += settings_problems(tag, r, cell, dict(S.GEOOPT_SETTINGS))
        readings["steps"][arm] = r.get("steps")
        if r.get("converged") is not True:
            gate.append(f"{tag}: optimisation did not converge ({r.get('converged')!r})")
        if not (_count(r.get("steps")) and 1 <= r["steps"] <= S.GEO_MAXSTEPS):
            gate.append(f"{tag}: {r.get('steps')!r} steps, not within 1..{S.GEO_MAXSTEPS}")
        gate += _steps_problems(tag, r, variant is not None)
        if variant is None:
            if r.get("mp_none") is not True:
                gate.append(f"{tag}: the stock optimisation carried a policy")
            continue
        pols = r.get("policies")
        want = (expect or {}).get(f"{variant}|{xc}") if isinstance(expect, dict) else None
        if want is None:
            gate.append(f"{tag}: no expected policy repr to judge the record against")
        elif pols != [want]:
            gate.append(f"{tag}: recorded policies {pols!r} are not [{want!r}]")
        arm_fits = set()
        for st in r.get("step") or []:
            try:
                arm_fits.add(classify_fit("mixed", xc, {"ao_cache_tier": st.get("tier"),
                                                        "ao_cache": st.get("note")}))
            except Exception as e:
                arm_fits.add(f"unclassifiable ({type(e).__name__})")
        readings["fit"][arm] = sorted(arm_fits)
        fits += [f for f in arm_fits if f in FIT_RANK]
    if structural:
        return structural, gate, _worst_fit(fits), readings
    # Endpoint certificates against S's final energy.
    try:
        e_s = float(by["S"]["e_final"])
        if not math.isfinite(e_s):
            raise ValueError("non-finite")
    except Exception as e:
        e_s = None
        gate.append(f"{key} S: final energy unreadable ({type(e).__name__})")
    for arm in GEO_MIXED:
        tag = f"{key} {arm} endpoint"
        ep = by[arm].get("endpoint")
        if not isinstance(ep, dict) or ep.get("error"):
            gate.append(f"{tag}: no stock certificate "
                        f"({ep.get('error') if isinstance(ep, dict) else ep!r})")
            continue
        gate += settings_problems(tag, ep, cell, dict(S.GEOOPT_SETTINGS))
        if ep.get("converged") is not True or ep.get("mp_none") is not True:
            gate.append(f"{tag}: the certificate SCF is not a converged stock SCF "
                        f"(converged={ep.get('converged')!r}, mp_none={ep.get('mp_none')!r})")
        try:
            nrm = grad_norms(ep["grad"])
            readings.setdefault("endpoint", {})[arm] = dict(nrm)
            if not nrm["grms"] <= S.GEO_CONV["convergence_grms"]:
                gate.append(f"{tag}: stock grms {nrm['grms']:.3e} > "
                            f"{S.GEO_CONV['convergence_grms']:g}")
            if not nrm["gmax"] <= S.GEO_CONV["convergence_gmax"]:
                gate.append(f"{tag}: stock gmax {nrm['gmax']:.3e} > "
                            f"{S.GEO_CONV['convergence_gmax']:g}")
        except Exception as e:
            gate.append(f"{tag}: gradient unreadable: {type(e).__name__}: {e}"[:300])
        if e_s is not None:
            try:
                de = abs(float(ep["e"]) - e_s)
                readings.setdefault("endpoint", {}).setdefault(arm, {})["de_vs_S"] = de
                if not de <= GEO_E_TOL:
                    gate.append(f"{tag}: |E_stock(endpoint) - E_S(final)| {de:.3e} > {GEO_E_TOL:g}")
            except Exception as e:
                gate.append(f"{tag}: energy unreadable: {type(e).__name__}"[:300])
    readings.update(geoopt_readings(by))
    return structural, gate, _worst_fit(fits), readings


def geoopt_readings(by):
    """DISCLOSED only (P-F .. P-H): per-arm warm-step cycle deltas against S matched by step index,
    summed warm-step SCF walls and the stock/mixed ratio, M0's initial orbital-gradient norms, MW's
    trigger fraction, Kabsch RMSD of the M0 / MW endpoints against S's. Never raises."""
    out = {}
    try:
        s_rec = by["S"].get("step") or []
        s_cyc = _warm(s_rec, "c")
        s_wall = [w for w in _warm(s_rec, "t") if _finite_real(w)]
        out["warm_scf_s"] = {"S": sum(s_wall) if s_wall else None}
        for arm in GEO_MIXED:
            rec = by[arm].get("step") or []
            cyc = _warm(rec, "c")
            n = min(len(cyc), len(s_cyc))
            d = [(cyc[i] - s_cyc[i]) if (_count(cyc[i]) and _count(s_cyc[i])) else None
                 for i in range(n)]
            dd = [x for x in d if x is not None]
            fast = [d[i] for i in range(n) if d[i] is not None and s_cyc[i] <= 5]
            walls = [w for w in _warm(rec, "t") if _finite_real(w)]
            out["warm_scf_s"][arm] = sum(walls) if walls else None
            out[f"{arm}_cycle_delta"] = d
            out[f"{arm}_frac_ge1"] = (sum(x >= 1 for x in dd) / len(dd)) if dd else None
            out[f"{arm}_frac_ge1_where_S_le5"] = (sum(x >= 1 for x in fast) / len(fast)
                                                 if fast else None)
            out[f"{arm}_frac_le0"] = (sum(x <= 0 for x in dd) / len(dd)) if dd else None
            out[f"{arm}_wall_ratio_S_over"] = _ratio(out["warm_scf_s"]["S"],
                                                     out["warm_scf_s"][arm])
            ws = [warm_start_fields(x) for x in _warm(rec, "ws")]
            if arm == "M0":
                out["M0_warm_gorb"] = [w[0] for w in ws]
            else:
                trig = [w[1] for w in ws]
                out["MW_trigger_fraction"] = (sum(trig) / len(trig)
                                              if trig and all(isinstance(t, bool) for t in trig)
                                              else None)
            try:
                out[f"{arm}_kabsch_rmsd_A"] = kabsch_rmsd(
                    [r[1:] for r in by[arm]["final"]], [r[1:] for r in by["S"]["final"]])
            except Exception:
                out[f"{arm}_kabsch_rmsd_A"] = None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def judge_cell(mode, cell, runs, end, repeats, expect=None):
    """The cell's verdict: {key, status, fit, problems, fails_run, readings}. `end` is the cell's
    RFCBENCH_CELLEND record, or None if the cell never finished (CAP_KILLED). Never raises."""
    out = {"key": None, "status": None, "fit": None, "problems": [], "readings": {}}
    try:
        key = out["key"] = S.cell_key(cell)
        if end is None:
            out.update(status=CAP_KILLED, problems=[f"{key}: never finished (cap or crash)"])
        elif end.get("error"):
            out.update(status=ERROR, problems=[f"{key}: {end['error']}"])
        else:
            if cell["kind"] in ("attrib", "geoopt"):
                judge = judge_attrib if cell["kind"] == "attrib" else judge_geoopt
                structural, gate, fit, readings = judge(mode, cell, runs, expect)
                out.update(readings=readings, fit=fit, problems=structural + gate,
                           status=ERROR if structural else (GATE_FAIL if gate else OK))
                out["fails_run"] = fails_run(mode, out["status"])
                return out
            if cell["kind"] == "downstream":
                structural, gate, fit, readings = judge_downstream(mode, cell, runs, expect)
                out["readings"] = readings
            else:
                structural, gate, fit = judge_speed(mode, cell, runs, repeats, expect)
            out.update(status=_status(mode, structural, gate, fit), fit=fit,
                       problems=structural + gate)
            if fit != TREATED and mode in S.FALLBACK_FAIL_MODES:
                out["problems"].append(f"{key}: fit {fit} is a FAIL in mode {mode} (PREREG-0 s2)")
    except Exception as e:
        out.update(status=ERROR,
                   problems=[f"{out['key']}: judge exception {type(e).__name__}: {e}"[:300]])
    out["fails_run"] = fails_run(mode, out["status"])
    return out


def fails_run(mode, status):
    if status == OK:
        return False
    if status in (CACHE_FALLBACK, NOT_TREATED):
        return mode not in S.FALLBACK_OUTCOME_MODES
    return True


# --------------------------------------------------------------------------------------------- #
# the GPU half -- never imported by the pod's judge path or the host tests' gate tests
# --------------------------------------------------------------------------------------------- #

def emit(tag, obj):
    print(f"{tag} " + json.dumps(obj, default=str, separators=(",", ":")), flush=True)


def cutensor_preload():
    """A copy of bench.cutensor_preload (never imported): cupy 14 loads libcutensor only through
    `cupy_backends.cuda.libs.__getattr__`, which gpu4pyscf 1.8.1's import form never calls. Run
    BEFORE anything imports gpu4pyscf. Never raises."""
    import importlib
    out = {"direct_import_error": None, "accessor_ok": False, "accessor_error": None}
    try:
        importlib.import_module("cupy_backends.cuda.libs.cutensor")
    except Exception as e:
        out["direct_import_error"] = f"{type(e).__name__}: {e}"[:300]
    try:
        libs = importlib.import_module("cupy_backends.cuda.libs")
        getattr(libs, "cutensor")
        out["accessor_ok"] = True
    except Exception as e:
        out["accessor_error"] = f"{type(e).__name__}: {e}"[:300]
    return out


CUTENSOR_SELFTEST_TOL64 = 1e-10
CUTENSOR_SELFTEST_REL32 = 1e-4


def contract_engine_info():
    """A copy of bench.contract_engine_info: which engine gpu4pyscf's `contract` runs on, read off
    the module, plus a self-test against cupy.einsum. Never raises."""
    out = {"cutensor_loaded": False, "engine": None, "selftest_err64": None,
           "selftest_rel32": None, "error": ""}
    try:
        from gpu4pyscf.lib import cutensor as ct
        import cupy as cp
        out["cutensor_loaded"] = getattr(ct, "cutensor", None) is not None
        out["engine"] = getattr(ct, "contract_engine", None) or "cutensor"
        rng = cp.random.default_rng(7)
        a = rng.standard_normal((6, 5, 4))
        b = rng.standard_normal((3, 5, 7))
        ref = cp.einsum("ijk,ljm->iklm", a, b)
        got = ct.contract("ijk,ljm->iklm", a, b)
        out["selftest_err64"] = float(cp.abs(got - ref).max())
        got32 = ct.contract("ijk,ljm->iklm", a.astype(cp.float32), b.astype(cp.float32))
        out["selftest_rel32"] = float(cp.abs(got32 - ref).max() / cp.abs(ref).max())
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def cutensor_problems(info):
    """PREREG-0 section 1: cuTENSOR on, and OBSERVED. Pure."""
    probs = []
    try:
        if info["cutensor_loaded"] is not True:
            probs.append(f"cuTENSOR did not load ({info.get('error')!r})")
        if info["engine"] != "cutensor":
            probs.append(f"gpu4pyscf contract engine is {info['engine']!r}, not cutensor")
        e64, r32 = info["selftest_err64"], info["selftest_rel32"]
        if not (isinstance(e64, float) and e64 <= CUTENSOR_SELFTEST_TOL64):
            probs.append(f"cuTENSOR fp64 self-test {e64!r} > {CUTENSOR_SELFTEST_TOL64}")
        if not (isinstance(r32, float) and r32 <= CUTENSOR_SELFTEST_REL32):
            probs.append(f"cuTENSOR fp32 self-test rel {r32!r} > {CUTENSOR_SELFTEST_REL32}")
    except Exception as e:
        probs.append(f"cuTENSOR record malformed: {type(e).__name__}: {e}"[:300])
    return probs


def read_atoms(mol):
    """`[(symbol, (x, y, z)), ...]` with numpy only: the LARGE reader for the LARGE set (its SHA256
    is checked by the pod before any SCF), freeze_geometries' for the ladder."""
    if mol in S.LARGE_NAMES:
        from scf_headtohead.freeze_rfc_large import load_geometry
    else:
        from scf_headtohead.freeze_geometries import load_geometry
    symbols, coords = load_geometry(mol)
    return [(s, tuple(float(v) for v in xyz)) for s, xyz in zip(symbols, coords)]


def _to_host(x):
    try:
        import cupy
        if isinstance(x, cupy.ndarray):
            return cupy.asnumpy(x)
    except Exception:
        pass
    return x


def build_mf(atoms, cell, policy, kind="speed", init_guess=None, settings=None):
    """A fresh molecule + configured SCF object, not run. `settings` ({conv_tol, conv_tol_grad,
    max_cycle}, conv_tol_grad None = library default) overrides `kind_settings(kind)`: PREREG-8's
    per-arm settings. Returns (mol, mf)."""
    import pyscf
    from gpu4pyscf.dft import rks
    ks = kind_settings(kind) if settings is None else settings
    mol = pyscf.M(atom=atoms, basis=S.BASES[cell["basis"]], verbose=0)
    mf = rks.RKS(mol, xc=cell["xc"])
    mf.grids.level = S.GRID_LEVEL
    mf.grids.prune = None
    mf = mf.density_fit(auxbasis=cell["aux"])
    mf.conv_tol = ks["conv_tol"]                 # re-pinned AFTER density_fit
    mf.max_cycle = ks["max_cycle"]
    if ks["conv_tol_grad"] is not None:
        mf.conv_tol_grad = ks["conv_tol_grad"]
    if init_guess is not None:
        mf.init_guess = init_guess
    mf.mixed_precision = policy
    return mol, mf


def settings_of(mf, mol):
    """The settings READ BACK off a built-and-run SCF object (PREREG-0 section 1)."""
    wdf = mf.with_df
    return {"grid_level": mf.grids.level, "prune_none": mf.grids.prune is None,
            "ngrids": int(mf.grids.coords.shape[0]), "conv_tol": mf.conv_tol,
            "max_cycle": mf.max_cycle, "conv_tol_grad": mf.conv_tol_grad,
            "auxbasis": wdf.auxbasis, "basis": mol.basis,
            "naux": int(getattr(wdf, "naux", None) or wdf.auxmol.nao),
            "nao": int(mol.nao), "init_guess": mf.init_guess,
            "cderi": cderi_storage(wdf)}


def record_of(r):
    """The compact mixed_precision_record of one treated SCF. The PREREG-8 fields (`diis_reset_call`,
    `warm_start`; 09271907 only) are read with `.get`, so a 63af0568 record still serialises, and
    `geo_fields` lists which of them the record really carries (a None must be OBSERVED)."""
    return {
        "policy": r["policy"], "xc": rle(r["xc"]), "k": rle(r["k"]), "vv10": rle(r["vv10"]),
        "fp64_tail": r["fp64_tail"], "forced": r["forced"], "ao_cache": r["ao_cache"][:400],
        "ao_cache_tier": r.get("ao_cache_tier"), "xc_fp64_cached": r.get("xc_fp64_cached"),
        "xc_fp64_stock": r.get("xc_fp64_stock"), "ao_cache_bytes64": r.get("ao_cache_bytes64"),
        "ao_cache_bytes32": r.get("ao_cache_bytes32"),
        "ao_cache_mirror_released_call": r.get("ao_cache_mirror_released_call"),
        "xc_switch_call": r.get("xc_switch_call"), "k_switch_call": r.get("k_switch_call"),
        "k_full_rebuild_call": r.get("k_full_rebuild_call"),
        "cderi_prebuilt": r.get("cderi_prebuilt"), "vv10_n_fp32": r.get("vv10_n_fp32"),
        "vv10_n_df64": r.get("vv10_n_df64"), "vv10_switch_call": r.get("vv10_switch_call"),
        "vv10_cert": r.get("vv10_cert"),
        "diis_reset_call": r.get("diis_reset_call"), "warm_start": r.get("warm_start"),
        "geo_fields": [f for f in ("diis_reset_call", "warm_start") if f in r]}


def build_and_run(atoms, cell, policy, kind="speed", init_guess=None, hook=None, settings=None):
    """One fresh molecule + SCF object, timed sync-to-sync. Returns (mf, run_record). `hook(mf)`, if
    given, is called on the fully configured object just before the timed kernel (PREREG-rfcbench-7
    stage timing); every campaign mode before it passes none. `settings` overrides the kind's SCF
    settings (PREREG-8 attrib arms); every mode before PREREG-8 passes none."""
    import cupy
    mol, mf = build_mf(atoms, cell, policy, kind=kind, init_guess=init_guess, settings=settings)
    if hook is not None:
        hook(mf)
    cupy.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    e = mf.kernel()
    cupy.cuda.runtime.deviceSynchronize()
    wall = time.perf_counter() - t0
    run = {"wall_s": round(wall, 4), "e": float(e), "cycles": int(mf.cycles),
           "converged": bool(mf.converged), "settings": settings_of(mf, mol),
           "mp_none": mf.mixed_precision is None,
           "rec_none": getattr(mf, "mixed_precision_record", None) is None}
    if policy is not None:
        run["rec"] = record_of(mf.mixed_precision_record)
    return mf, run


def cderi_storage(wdf):
    """Where the DF tensor lives after the SCF (PREREG-rfcbench-6): the type of every stored block and
    their total bytes. gpu4pyscf keeps it on the device only when it fits its own memory rule, and in
    host memory otherwise. Never raises: a failure is recorded as such."""
    try:
        c = getattr(wdf, "_cderi", None)
        if isinstance(c, dict):
            arrs = list(c.values())
        elif isinstance(c, (list, tuple)):
            arrs = list(c)
        else:
            arrs = [] if c is None else [c]
        kinds = sorted({f"{type(a).__module__}.{type(a).__name__}" for a in arrs})
        return {"types": kinds, "nbytes": int(sum(int(getattr(a, "nbytes", 0) or 0) for a in arrs))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def release_previous():
    """G2: no SCF object of an earlier run may be alive when the next one is built (it would hold its
    DF tensor and grids, and the arms would not start from the same state). Called with every
    reference already dropped. The cupy memory POOL is deliberately NOT freed: a warm pool is the
    advantage stock-last gives the baseline (PREREG-0 section 1)."""
    gc.collect()


def run_speed_cell(cell, repeats, atoms):
    from gpu4pyscf.dft.mixed_precision import MixedPrecision
    for p in expected_pairs(cell, repeats):
        runs = []
        for arm in cell["arms"]:                       # ARM_ORDER sub-sequence: stock LAST
            kw = policy_kwargs(arm, cell["xc"])
            run = {"arm": arm}
            release_previous()
            try:
                mf, rec = build_and_run(atoms, cell, None if kw is None else MixedPrecision(**kw),
                                        kind=cell["kind"])
                del mf                                 # the record holds no reference to it
                run.update(rec)
            except Exception as e:
                run["error"] = f"{type(e).__name__}: {e}"[:300]
            runs.append(run)
        emit("RFCBENCH_CELL", {"key": S.cell_key(cell), "pair": p, "cold": p == 0, "runs": runs})


def run_downstream_cell(cell, atoms):
    from gpu4pyscf.dft.mixed_precision import MixedPrecision
    plan = (("R", None, "R", None), ("S", None, "speed", None), ("Sp", None, "speed", "atom"),
            ("M", "mixed", "speed", None))
    done = {}
    mf = g = None
    for name, arm, kind, guess in plan:
        run = {"run": name}
        mf = g = None
        release_previous()
        try:
            kw = policy_kwargs(arm, cell["xc"]) if arm else None
            mf, rec = build_and_run(atoms, cell, None if kw is None else MixedPrecision(**kw),
                                    kind=kind, init_guess=guess)
            run.update(rec)
            run["dip"] = [float(v) for v in _to_host(mf.dip_moment(unit="Debye", verbose=0))]
            run["state_none"] = getattr(mf, "_mixed_precision_state", None) is None
            g = mf.nuc_grad_method()
            run["grad_module"] = type(g).__module__
            run["grad"] = [[float(v) for v in row] for row in _to_host(g.kernel())]
        except Exception as e:
            run["error"] = f"{type(e).__name__}: {e}"[:300]
        mf = g = None
        done[name] = run
        emit("RFCBENCH_CELL", {"key": S.cell_key(cell), "run": name, "runs": [run]})
        if name == "Sp":
            # PREREG-4: R, S and S' and their deltas are printed BEFORE M runs.
            ctl = {}
            for a, b in (("S", "R"), ("Sp", "R"), ("Sp", "S")):
                try:
                    ctl[f"{a}-{b}"] = {"de": abs(done[a]["e"] - done[b]["e"]),
                                       "dg": _maxabs_diff(done[a]["grad"], done[b]["grad"]),
                                       "dmu": _maxabs_diff(done[a]["dip"], done[b]["dip"])}
                except Exception as e:
                    ctl[f"{a}-{b}"] = f"n/a ({type(e).__name__})"
            emit("RFCBENCH_CONTROLS", {"key": S.cell_key(cell), "deltas": ctl})


def _sync():
    import cupy
    cupy.cuda.runtime.deviceSynchronize()


def stock_gradient(mf, run):
    """Dipole, the no-state check and the stock analytic gradient, timed sync-to-sync, into `run`."""
    run["dip"] = [float(v) for v in _to_host(mf.dip_moment(unit="Debye", verbose=0))]
    run["state_none"] = getattr(mf, "_mixed_precision_state", None) is None
    g = mf.nuc_grad_method()
    run["grad_module"] = type(g).__module__
    _sync()
    t0 = time.perf_counter()
    grad = _to_host(g.kernel())
    _sync()
    run["grad_wall_s"] = round(time.perf_counter() - t0, 4)
    run["grad"] = [[float(v) for v in row] for row in grad]


def run_attrib_cell(cell, atoms):
    """PREREG-8 A1/A2: R', R, S, S2, St, M, Mt, Md, each a fresh SCF from the default guess at its own
    settings (S.ATTRIB_PLAN), then the dipole and the stock analytic gradient. One RFCBENCH_CELL line
    per arm, as it lands."""
    from gpu4pyscf.dft.mixed_precision import MixedPrecision
    for arm in S.ATTRIB_ARMS:
        variant, settings = S.ATTRIB_PLAN[arm]
        run = {"run": arm}
        mf = None
        release_previous()
        try:
            kw = variant_kwargs(variant, cell["xc"])
            mf, rec = build_and_run(atoms, cell, None if kw is None else MixedPrecision(**kw),
                                    settings=dict(settings))
            run.update(rec)
            stock_gradient(mf, run)
        except Exception as e:
            run["error"] = f"{type(e).__name__}: {e}"[:300]
        mf = None
        emit("RFCBENCH_CELL", {"key": S.cell_key(cell), "run": arm, "runs": [run]})


def geometric_kernel(scanner, callback, maxsteps):
    """pyscf's geomeTRIC driver at PREREG-8's convergence set (l3bcomm.CommOps.geometric_kernel)."""
    from pyscf.geomopt import geometric_solver
    return geometric_solver.kernel(scanner, maxsteps=int(maxsteps), callback=callback,
                                   **S.GEO_CONV)


def geoopt_arm(atoms, cell, policy, geo_kernel=None):
    """One optimisation through the gradient scanner. TIMING METHOD: the scanner's SCF
    (`scanner.base.kernel`) and its gradient (`scanner.kernel`) are each wrapped on the instance by a
    deviceSynchronize-to-deviceSynchronize perf_counter timer, so per step the SCF wall (DF and grid
    rebuild included, as the campaign measurand) and the gradient wall are separate and synchronised.
    The SCF wrapper also reads the step's cycles, convergence, energy and (mixed) record right after
    the SCF returns, before the gradient runs. geomeTRIC's callback counts the steps."""
    mol, mf = build_mf(atoms, cell, policy, settings=dict(S.GEOOPT_SETTINGS))
    out = {"mp_none": mf.mixed_precision is None, "step": [], "policies": []}
    scanner = mf.nuc_grad_method().as_scanner()
    base = scanner.base
    scf_kernel, grad_kernel = base.kernel, scanner.kernel
    steps = out["step"]

    def timed_scf(*a, **k):
        _sync()
        t0 = time.perf_counter()
        e = scf_kernel(*a, **k)
        _sync()
        st = {"c": int(base.cycles), "ok": bool(base.converged), "e": float(e),
              "t": round(time.perf_counter() - t0, 4), "g": None}
        if "settings" not in out:
            out["settings"] = settings_of(base, base.mol)
        rec = getattr(base, "mixed_precision_record", None)
        if policy is None:
            st["rec_none"] = rec is None
        elif not isinstance(rec, dict):         # recorded, so the gate fails on it; never raised
            st.update(tail=None, rec_missing=True)
        else:
            st.update(xc=rle(rec["xc"]), k=rle(rec["k"]), vv10=rle(rec["vv10"]),
                      tail=rec["fp64_tail"], ws=rec.get("warm_start"),
                      tier=rec.get("ao_cache_tier"), note=str(rec["ao_cache"])[:160])
            if rec["policy"] not in out["policies"]:
                out["policies"].append(rec["policy"])
        steps.append(st)
        return e

    def timed_grad(*a, **k):
        _sync()
        t0 = time.perf_counter()
        g = grad_kernel(*a, **k)
        _sync()
        if steps:
            steps[-1]["g"] = round(time.perf_counter() - t0, 4)
        return g

    base.kernel = timed_scf
    scanner.kernel = timed_grad
    n_cb = [0]

    def callback(envs):
        n_cb[0] += 1

    t0 = time.perf_counter()
    conv, mol_eq = (geo_kernel or geometric_kernel)(scanner, callback, S.GEO_MAXSTEPS)
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    coords = _to_host(mol_eq.atom_coords(unit="Angstrom"))
    out.update(converged=bool(conv), steps=n_cb[0], scf_calls=len(steps),
               e_final=steps[-1]["e"] if steps else None,
               final=[[mol_eq.atom_symbol(i)] + [float(v) for v in coords[i]]
                      for i in range(mol_eq.natm)])
    return out


def endpoint_eval(final, cell):
    """PREREG-8: a fresh stock SCF (default guess, campaign settings) and the stock gradient at an
    optimisation's final coordinates (Angstrom)."""
    atoms = [(r[0], tuple(r[1:])) for r in final]
    mf, run = build_and_run(atoms, cell, None, settings=dict(S.GEOOPT_SETTINGS))
    stock_gradient(mf, run)
    return run


def run_geoopt_cell(cell, atoms, geo_kernel=None):
    """PREREG-8 G1: M0, MW, S in that order; after each mixed optimisation its stock certificate.
    One RFCBENCH_CELL line per arm."""
    from gpu4pyscf.dft.mixed_precision import MixedPrecision
    for arm in S.GEOOPT_ARMS:
        run = {"run": arm}
        release_previous()
        try:
            kw = variant_kwargs(S.GEOOPT_VARIANT[arm], cell["xc"])
            run.update(geoopt_arm(atoms, cell, None if kw is None else MixedPrecision(**kw),
                                  geo_kernel=geo_kernel))
        except Exception as e:
            run["error"] = f"{type(e).__name__}: {e}"[:300]
        if S.GEOOPT_VARIANT[arm] is not None and "final" in run:
            release_previous()
            try:
                run["endpoint"] = endpoint_eval(run["final"], cell)
            except Exception as e:
                run["endpoint"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        emit("RFCBENCH_CELL", {"key": S.cell_key(cell), "run": arm, "runs": [run]})


def main(argv):
    """`--group <json>`: {"cells": [...], "repeats": R}. Streams lines; never judges."""
    spec = json.loads(argv[argv.index("--group") + 1])
    pre = cutensor_preload()
    info = contract_engine_info()
    emit("RFCBENCH_GROUP", {"cutensor_preload": pre, "contract_engine": info,
                            "cutensor_problems": cutensor_problems(info)})
    for cell in spec["cells"]:
        end = {"key": S.cell_key(cell), "error": None}
        t0 = time.time()
        try:
            atoms = read_atoms(cell["mol"])
            if cell["kind"] == "downstream":
                run_downstream_cell(cell, atoms)
            elif cell["kind"] == "attrib":
                run_attrib_cell(cell, atoms)
            elif cell["kind"] == "geoopt":
                run_geoopt_cell(cell, atoms)
            else:
                run_speed_cell(cell, spec["repeats"], atoms)
        except Exception as e:
            end["error"] = f"{type(e).__name__}: {e}"[:300]
        end["wall_s"] = round(time.time() - t0, 1)
        emit("RFCBENCH_CELLEND", end)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
