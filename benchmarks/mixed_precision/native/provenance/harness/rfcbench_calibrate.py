#!/usr/bin/env python3
"""Build `.github/scripts/rfcbench_calibration.json` from a banked PASS sentinel (C0 for pro6000;
a `cross` sentinel for its card). Stdlib only.

    python3 .github/scripts/rfcbench_calibrate.py <sentinel.txt> --run-id <RUN_ID> [--out PATH]

It refuses anything the extractor would not quote (STATUS != PASS, WORK_RC != 0, pod problems, a
wrong RUN_ID) and any cell without exactly R warm walls per arm. For the sentinel's GPU class it
writes:
  - `cells`: every OK speed cell's measured warm median per arm, x CALIBRATION_MARGIN (1.15);
  - `powerlaw`: per (xc, arm), w = a * size_measure**p fitted in log-log through the three trio cells
    at def2-mTZVPP, then `a` scaled up so the law is >= 1.15 x EVERY measured trio median. The
    estimate is an over-estimate by construction; it is never below a measured median x 1.15.
Other classes already in the file are kept unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rfcbench_extract as X  # noqa: E402
import rfcbench_sets as S  # noqa: E402

CAL_MODES = ("commission", "cross")


def fit_powerlaw(points, margin=S.CALIBRATION_MARGIN):
    """`points`: [(size, wall)], >= 2 distinct sizes. Least squares in log-log, then `a` raised so
    a*size**p >= margin*wall at every point."""
    xs = [math.log(s) for s, _ in points]
    ys = [math.log(w) for _, w in points]
    n = len(points)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 0:
        raise ValueError(f"power law needs >= 2 distinct sizes, got {points}")
    p = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = math.exp(my - p * mx)
    a *= max(margin * w / (a * s ** p) for s, w in points)
    return {"a": a, "p": p}


def calibrate(text, run_id):
    out = X.extract(text, run_id)                     # refuses an unquotable sentinel or RUN_ID
    header = out["header"]
    gpu, mode = header.get("GPU_CLASS"), header.get("RFC_MODE")
    if gpu not in S.GPU_CASCADES:
        raise X.RefusedSentinel(f"GPU_CLASS={gpu!r} is not a class")
    if mode not in CAL_MODES:
        raise X.RefusedSentinel(f"RFC_MODE={mode!r}: calibrate from {CAL_MODES} only")
    if out["missing"]:
        raise X.RefusedSentinel(f"cells without exactly R warm walls: {out['missing']}")
    cells, trio = {}, {}
    for c in out["cells"]:
        if c["status"] != "OK":
            continue
        _kind, mol, xc, basis = c["key"].split("/")
        for arm, med in c["median"].items():
            cells[f"{mol}|{xc}|{basis}|{arm}"] = med * S.CALIBRATION_MARGIN
            if mol in S.TRIO and basis == "mtzvpp":
                trio.setdefault(f"{xc}|{arm}", []).append((S.size_measure(mol), med))
    if not cells:
        raise X.RefusedSentinel("no OK speed cell to calibrate from")
    power = {k: fit_powerlaw(v) for k, v in sorted(trio.items()) if len(v) == len(S.TRIO)}
    return gpu, {"cells": cells, "powerlaw": power,
                 "source": {"run_id": run_id, "sha": header.get("SHA"),
                            "fork_sha": header.get("FORK_SHA"), "rfc_mode": mode,
                            "repeats": out["repeats"], "margin": S.CALIBRATION_MARGIN}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sentinel")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default=S.CALIBRATION_PATH)
    a = ap.parse_args(argv)
    try:
        gpu, cal = calibrate(open(a.sentinel).read(), a.run_id)
    except X.RefusedSentinel as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    data = {}
    if os.path.isfile(a.out):
        with open(a.out) as f:
            data = json.load(f)
    data[gpu] = cal
    S.validate_calibration(data)
    with open(a.out, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"calibrated {gpu}: {len(cal['cells'])} cells, power laws {sorted(cal['powerlaw'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
