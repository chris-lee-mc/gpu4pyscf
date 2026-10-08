#!/usr/bin/env python3
"""rfcbench extractor: a banked sentinel -> per-run CSV rows and the disclosed readings.

PREREG-rfcbench-0 section 3: every speed reading is computed HERE, from a banked sentinel, never on
the pod. Stdlib only.

    python3 .github/scripts/rfcbench_extract.py <sentinel.txt> --run-id <RUN_ID> [--csv out.csv]

Fails closed:
  - a sentinel that is not STATUS=PASS with WORK_RC=0 and an empty pod `problems` list is refused
    (a pod whose cuTENSOR was not observed, say, would otherwise yield PAYS geomeans). `--unquotable`
    reads it anyway and labels EVERY reading and aggregate UNQUOTABLE;
  - a cell enters the readings only with EXACTLY R warm walls per arm (R from the REPEATS header,
    else the pod's cfg; 1 for a bridge cell); otherwise it is `missing`;
  - the sentinel's line 2 must be `RUN_ID=<the requested run id>` (a file at the right path whose
    content is another dispatch's is refused);
  - it must carry the RFCBENCH_JSON line (the pod's verdicts), and every requested cell must have its
    streamed records, or the cell is listed as `missing` and enters no aggregate;
  - an aggregate over an empty selection is None, never 1.0.

Readings (PREREG-0 sections 3-4): per cell and arm the median warm wall over pairs 1..R (pair 0, the
cold pair, is disclosed only); ratio A/B = median(B) / median(A); spread (max - min) / median; NOISY
when either arm's spread exceeds 10 %; under_1s when any warm wall is below 1 s; PAYS / NEUTRAL /
HARMS at 1.15 / 0.95; a pod is DEGRADED when more than 20 % of its speed cells are NOISY. Only cells
whose pod status is OK enter the treated aggregates; fallback cells go to the fit table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys

PAYS, HARMS = 1.15, 0.95
NOISY_SPREAD = 0.10
DEGRADED_NOISY_FRACTION = 0.20
UNDER_S = 1.0
CANARY = ("speed/paracetamol/r2scan/mtzvpp", "stock")
CANARY_BAND_S = (3.8, 5.5)          # PRO 6000 only
CANARY_DEGRADED_S = 6.5
RATIOS = (("mixed", "stock"), ("mixed", "cache"), ("cache", "stock"))


class RefusedSentinel(Exception):
    pass


def reading(r):
    if r is None or not math.isfinite(r):
        return None
    return "PAYS" if r >= PAYS else ("HARMS" if r <= HARMS else "NEUTRAL")


def geomean(xs):
    xs = [x for x in xs if x is not None]
    if not xs or any(not (x > 0 and math.isfinite(x)) for x in xs):
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def parse(text, run_id):
    lines = text.splitlines()
    if len(lines) < 2 or lines[1].strip() != f"RUN_ID={run_id}":
        raise RefusedSentinel(f"line 2 is {lines[1].strip() if len(lines) > 1 else None!r}, "
                              f"not RUN_ID={run_id}")
    header = {}
    for ln in lines:
        if ln == "---":
            break
        k, _, v = ln.partition("=")
        header[k] = v
    if header.get("MODE") != "rfcbench":
        raise RefusedSentinel(f"MODE={header.get('MODE')!r} is not rfcbench")
    runs, pod = {}, None
    for ln in lines:
        if ln.startswith("RFCBENCH_CELL "):
            obj = json.loads(ln[len("RFCBENCH_CELL "):])
            for r in obj.get("runs") or []:
                r = dict(r, pair=obj.get("pair"), run=obj.get("run"))
                runs.setdefault(obj["key"], []).append(r)
        elif ln.startswith("RFCBENCH_JSON "):
            pod = json.loads(ln[len("RFCBENCH_JSON "):])
    if pod is None:
        raise RefusedSentinel("no RFCBENCH_JSON line (truncated or the pod never finished)")
    return header, runs, pod


def quotability(header, pod):
    """Why this sentinel's numbers may not be quoted; empty when they may."""
    why = []
    if header.get("STATUS") != "PASS":
        why.append(f"STATUS={header.get('STATUS')!r}")
    if header.get("WORK_RC") != "0":
        why.append(f"WORK_RC={header.get('WORK_RC')!r}")
    if pod.get("problems") != []:
        why.append(f"pod problems: {str(pod.get('problems'))[:300]}")
    return why


def repeats_of(header, pod):
    raw = header.get("REPEATS", "")
    if raw.isdigit():
        return int(raw)
    r = (pod.get("cfg") or {}).get("repeats")
    if isinstance(r, int) and not isinstance(r, bool):
        return r
    raise RefusedSentinel("no REPEATS header and no pod cfg.repeats: warm-wall count unknown")


def warm_walls(runs, n_warm):
    """`{arm: [walls]}` over pairs >= 1, or None unless every arm has exactly `n_warm` of them, each a
    distinct pair, with no error (a partial cell is never read)."""
    warm, pairs = {}, {}
    arms = {r.get("arm") for r in runs}
    for r in runs:
        p = r.get("pair")
        if not (isinstance(p, int) and p >= 1):
            continue
        if r.get("error") or not isinstance(r.get("wall_s"), (int, float)):
            return None
        warm.setdefault(r["arm"], []).append(float(r["wall_s"]))
        pairs.setdefault(r["arm"], []).append(p)
    if len(arms) < 2 or set(warm) != arms:
        return None
    for arm in arms:
        if len(warm[arm]) != n_warm or sorted(pairs[arm]) != list(range(1, n_warm + 1)):
            return None
    return warm


def cell_readings(key, runs, n_warm):
    warm = warm_walls(runs, n_warm)
    if warm is None:
        return None
    out = {"key": key, "median": {}, "spread": {}, "ratios": {}, "readings": {},
           "pair_ratios": {}}
    for arm, ws in warm.items():
        med = statistics.median(ws)
        out["median"][arm] = med
        out["spread"][arm] = (max(ws) - min(ws)) / med if med > 0 else None
    for a, b in RATIOS:
        if a in out["median"] and b in out["median"] and out["median"][a] > 0:
            r = out["median"][b] / out["median"][a]
            out["ratios"][f"{a}/{b}"] = r
            out["readings"][f"{a}/{b}"] = reading(r)
    by_pair = {}
    for r in runs:
        if isinstance(r.get("pair"), int) and r["pair"] >= 1:
            by_pair.setdefault(r["pair"], {})[r["arm"]] = float(r["wall_s"])
    for p, w in sorted(by_pair.items()):
        out["pair_ratios"][str(p)] = {f"{a}/{b}": w[b] / w[a] for a, b in RATIOS
                                      if a in w and b in w and w[a] > 0}
    # Determinism floor: stock-vs-stock |dE| across every pair (cold included).
    es = [float(r["e"]) for r in runs if r.get("arm") == "stock" and "e" in r and not r.get("error")]
    out["stock_de_floor"] = (max(es) - min(es)) if len(es) >= 2 else None
    # VV10 certificate wall share of the mixed arm's warm walls (wB97M-V only).
    shares = []
    for r in runs:
        cert = (r.get("rec") or {}).get("vv10_cert")
        if (r.get("arm") == "mixed" and (r.get("pair") or 0) >= 1 and isinstance(cert, dict)
                and r.get("wall_s")):
            shares.append(float(cert.get("wall_s") or 0.0) / float(r["wall_s"]))
    out["cert_wall_share"] = statistics.median(shares) if shares else None
    out["noisy"] = any(s is None or s > NOISY_SPREAD for s in out["spread"].values())
    out["under_1s"] = any(w < UNDER_S for ws in warm.values() for w in ws)
    return out


# PREREG-rfcbench-0 Amendment 3: a pod whose GPU already draws more than this before the first group
# (sampled idle, after the probe) is CONTENDED. Its speed readings are reported separately and never
# pooled, and it is re-dispatched once. W1-W3 idled at 57-92 W; W4 drew 576 W.
CONTENDED_IDLE_W = 200.0


def contention(pod):
    """(contended, idle_w): contended is True/False, or None when the pod recorded no readable
    pre-work power (C0 predates the telemetry). Never raises."""
    try:
        first = (pod.get("groups") or ((pod.get("extra") or {}).get("phases")) or [{}])[0]
        st = first.get("gpu_state_before") or {}
        w = float(st["power.draw"])
    except Exception:
        return None, None
    if w != w:                                            # NaN
        return None, None
    return w > CONTENDED_IDLE_W, w


def extract(text, run_id, unquotable=False):
    header, runs, pod = parse(text, run_id)
    why = quotability(header, pod)
    if why and not unquotable:
        raise RefusedSentinel(f"not quotable ({'; '.join(why)}); pass --unquotable to read it "
                              f"with every number labelled UNQUOTABLE")
    if pod.get("extra"):                         # PREREG-rfcbench-7 extra modes
        ex = pod["extra"]
        c, idle = contention(pod)
        return {"header": header, "status": header.get("STATUS"), "quotable": not why,
                "unquotable_why": why, "mode": (pod.get("cfg") or {}).get("mode"),
                "readings": ex.get("readings"), "phases": ex.get("phases"),
                "skipped": ex.get("skipped"), "scfs": len(ex.get("runs") or []),
                "contended": c, "idle_power_w": idle}
    repeats = repeats_of(header, pod)
    verdicts = {v["key"]: v for v in pod.get("cells") or []}
    requested = pod.get("cells_requested") or []
    cells, missing, fit_table, downstream, bridge, prereg8 = [], [], [], [], [], []
    for key in requested:
        v = verdicts.get(key, {})
        kind = key.split("/", 1)[0]
        if kind in ("attrib", "geoopt"):       # PREREG-rfcbench-8: the pod's verdict and readings only
            prereg8.append({"key": key, "status": v.get("status"), "fit": v.get("fit"),
                            "readings": v.get("readings"),
                            "streamed_runs": len(runs.get(key, []))})
            continue
        if kind == "downstream":
            downstream.append({"key": key, "status": v.get("status"),
                               "readings": v.get("readings")})
            continue
        cr = cell_readings(key, runs.get(key, []), 1 if kind == "bridge" else repeats)
        if cr is None:
            missing.append(key)
            continue
        cr.update(status=v.get("status"), fit=v.get("fit"))
        (bridge if kind == "bridge" else cells).append(cr)
        if v.get("status") in ("CACHE_FALLBACK", "NOT_TREATED"):
            mixed = [r.get("rec") or {} for r in runs[key] if r.get("arm") == "mixed"]
            fit_table.append({"key": key, "fit": v.get("status"),
                              "tiers": sorted({str(m.get("ao_cache_tier")) for m in mixed}),
                              "ao_cache": [m.get("ao_cache") for m in mixed][:1]})
    treated = [c for c in cells if c["status"] == "OK"]
    agg = {}
    # One aggregate per (xc, basis): cells of different bases are never pooled (C0 pooled its
    # def2-SVP cell into the r2SCAN mTZVPP geomean before this). Key: "<xc>/<basis> <a>/<b>".
    groups = sorted({tuple(c["key"].split("/")[2:4]) for c in treated})
    for xc, basis in groups:
        sel = [c for c in treated if tuple(c["key"].split("/")[2:4]) == (xc, basis)]
        for a, b in RATIOS:
            g = geomean([c["ratios"].get(f"{a}/{b}") for c in sel]) if all(
                f"{a}/{b}" in c["ratios"] for c in sel) else None
            if g is not None:
                agg[f"{xc}/{basis} {a}/{b}"] = {"geomean": g, "reading": reading(g),
                                                "n": len(sel)}
    n_noisy = sum(c["noisy"] for c in cells)
    canary = next((c["median"].get(CANARY[1]) for c in cells if c["key"] == CANARY[0]), None)
    on_pro = header.get("GPU_CLASS") == "pro6000"          # the band is a PRO 6000 band only
    out = {"header": header, "status": header.get("STATUS"), "quotable": not why,
           "unquotable_why": why, "repeats": repeats, "cells": cells, "bridge": bridge,
           "downstream": downstream, "prereg8": prereg8, "missing": missing, "fit_table": fit_table,
           "aggregates": agg, "noisy": n_noisy,
           "degraded": bool(cells) and n_noisy / len(cells) > DEGRADED_NOISY_FRACTION,
           "canary_s": canary,
           "canary_in_band": (CANARY_BAND_S[0] <= canary <= CANARY_BAND_S[1]
                              if on_pro and canary is not None else None),
           "canary_degraded": (canary >= CANARY_DEGRADED_S
                               if on_pro and canary is not None else None)}
    out["contended"], out["idle_power_w"] = contention(pod)
    if out["contended"] is not False:
        tag = "CONTENDED" if out["contended"] else "CONTENTION-UNKNOWN"
        for a in agg.values():
            a["pool"] = tag                               # reported separately, never pooled
    if why:
        out["label"] = "UNQUOTABLE"
        for c in cells + bridge:
            c["label"] = "UNQUOTABLE"
            c["readings"] = {k: f"UNQUOTABLE {v}" for k, v in c["readings"].items()}
        for a in agg.values():
            a["label"] = "UNQUOTABLE"
            a["reading"] = f"UNQUOTABLE {a['reading']}"
        for d in downstream + prereg8:
            d["label"] = "UNQUOTABLE"
    return out


CSV_FIELDS = ("run_id", "key", "pair", "run", "arm", "wall_s", "e", "cycles", "converged",
              "ngrids", "naux", "nao", "policy", "ao_cache_tier", "xc", "k", "vv10", "fp64_tail",
              "ao_cache_bytes64", "ao_cache_bytes32", "cderi_types", "cderi_bytes", "error")


# PREREG-rfcbench-7 extra modes: one row per SCF of rfcbench_extra (RFCBENCH_JSON "extra").
EXTRA_CSV_FIELDS = ("run_id", "phase", "proc", "seq", "pair", "mol", "xc", "basis", "arm", "wall_s",
                    "t0", "t1", "e", "cycles", "converged", "tier", "xc_s", "jk_s", "dfbuild_s",
                    "eig_s", "nlc_s", "xc_n", "jk_n", "dfbuild_n", "eig_n", "error")


def extra_rows(text, run_id):
    _, _, pod = parse(text, run_id)
    extra = pod.get("extra")
    if not extra:
        raise RefusedSentinel("no 'extra' record: not a PREREG-rfcbench-7 extra-mode sentinel")
    for r in extra.get("runs") or []:
        st = r.get("stages") or {}
        row = {k: r.get(k) for k in EXTRA_CSV_FIELDS if k not in ("run_id",)}
        row.update({k: st.get(k) for k in ("xc_s", "jk_s", "dfbuild_s", "eig_s", "nlc_s",
                                           "xc_n", "jk_n", "dfbuild_n", "eig_n")})
        row["run_id"] = run_id
        yield row


def csv_rows(text, run_id):
    _, runs, _ = parse(text, run_id)
    for key, rs in runs.items():
        for r in rs:
            rec, st = r.get("rec") or {}, r.get("settings") or {}
            yield {"run_id": run_id, "key": key, "pair": r.get("pair"), "run": r.get("run"),
                   "arm": r.get("arm"), "wall_s": r.get("wall_s"), "e": r.get("e"),
                   "cycles": r.get("cycles"), "converged": r.get("converged"),
                   "ngrids": st.get("ngrids"), "naux": st.get("naux"), "nao": st.get("nao"),
                   "policy": rec.get("policy"), "ao_cache_tier": rec.get("ao_cache_tier"),
                   "xc": rec.get("xc"), "k": rec.get("k"), "vv10": rec.get("vv10"),
                   "fp64_tail": rec.get("fp64_tail"), "ao_cache_bytes64": rec.get("ao_cache_bytes64"),
                   "ao_cache_bytes32": rec.get("ao_cache_bytes32"),
                   "cderi_types": ";".join((st.get("cderi") or {}).get("types") or []) or None,
                   "cderi_bytes": (st.get("cderi") or {}).get("nbytes"), "error": r.get("error")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sentinel")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--unquotable", action="store_true",
                    help="read a FAIL / WORK_RC!=0 / problem sentinel, labelling every number")
    a = ap.parse_args(argv)
    text = open(a.sentinel).read()
    try:
        out = extract(text, a.run_id, unquotable=a.unquotable)
    except RefusedSentinel as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if a.csv:
        extra = (parse(text, a.run_id)[2] or {}).get("extra")
        with open(a.csv, "w", newline="") as f:
            if extra:
                w = csv.DictWriter(f, fieldnames=EXTRA_CSV_FIELDS)
                w.writeheader()
                w.writerows(extra_rows(text, a.run_id))
            else:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows(csv_rows(text, a.run_id))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
