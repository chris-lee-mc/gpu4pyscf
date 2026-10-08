#!/usr/bin/env python3
"""Pod side of the `rfcbench` dispatch (docs/upstream/PREREG-rfcbench-0-measurand.md).

It runs on a pod where the pinned stack (gpu4pyscf-cuda12x==1.8.1, the 14-pin lock) and cuTENSOR are
already installed, with cuTENSOR's lib dir on LD_LIBRARY_PATH. Steps, each fail-closed:

  1. Card gate: `nvidia-smi` shows the requested class's single model and no MIG device; for
     `pro6000mig` exactly one MIG device under an RTX PRO 6000 (and, in step 6, a CUDA total memory
     of 20-28 GB). A landing on anything else means no row: the pod stops here.
  2. Fetch the fork at RFCBENCH_FORK_SHA, the v1.8.1 tag and every VALIDATED_FORK_SHAS entry; each
     must resolve.
  3. Validated tree: `git diff --quiet <validated> FORK_SHA -- gpu4pyscf/` against each validated
     commit in order; the first that is identical is recorded (`validated_match`). None identical, or
     a git error, refuses. A mode in MODE_REQUIRED_TREE (PREREG-8 attrib / geoopt) refuses any
     match but its own commit.
  4. Every file the branch changes under gpu4pyscf/ (tests excluded) is byte-identical in the
     installed wheel to its v1.8.1 copy (modified) or absent (added); then the overlay.
     (Copied from g4pport_pod.py steps 1-2.)
  5. The LARGE geometries' SHA256 against rfcbench_sets.LARGE_SHA256, before any SCF.
  6. Environment probe, in its own process: cuTENSOR loaded and observed (engine + self-test), the
     geometry readers import no rdkit, the CUDA-reported total memory. A cuTENSOR failure makes the
     pod INVALID (a FAIL); the groups still run, so the accuracy record is not lost.
  7. One subprocess per xc group (`rfcbench_cells.py --group`), each under
     min(remaining, 1.5 x its estimate). Every RFCBENCH_CELL line is re-emitted as it arrives, so a
     cap kill loses only unfinished cells, which are CAP_KILLED. A group is skipped (its cells
     CAP_KILLED) when too little time is left to start it.
  8. Every requested cell is judged by `rfcbench_cells.judge_cell` from the streamed lines; every
     requested cell must be present exactly once and nothing else may be.

The last two lines are `RFCBENCH_JSON {...}` and a whole-line `RFCBENCH_PASS` or `RFCBENCH_FAIL`.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rfcbench_sets as S  # noqa: E402
import rfcbench_cells as C  # noqa: E402   (its module level is stdlib only)

SRC = "/tmp/rfcfork"
LOCK_PATH = os.path.join(HERE, "scfbench_requirements.lock")
PROBE_TIMEOUT_S = 300
# Time kept back from the work cap to judge, print the JSON and the verdict.
FINISH_RESERVE_S = 60
# A group is not started with less than this left: it could not finish its cold run.
GROUP_MIN_S = 120
GROUP_TIMEOUT_FACTOR = 1.5
# Non-protocol output of a failed group kept for the log (bounded: sizes the sentinel tail).
GROUP_DIAG_LINES = 15
PROTOCOL = ("RFCBENCH_CELL ", "RFCBENCH_CELLEND ", "RFCBENCH_GROUP ", "RFCBENCH_CONTROLS ")

rec = {"problems": []}


def emit(line):
    print(line, flush=True)


def problem(msg):
    rec["problems"].append(msg)
    emit(f"FAIL {msg}")


def sh(args, cwd=None, timeout=600):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def git(*args, timeout=600):
    return sh(["git", "-C", SRC, *args], timeout=timeout)


def cfg_from_env(env=None):
    """The dispatch, as the launcher validated it. The pod RECOMPUTES the cell list from the same
    rfcbench_sets filters, so the coverage check below is against the pre-registered cells, not
    against whatever the cells script happened to run."""
    env = os.environ if env is None else env

    def lst(name):
        raw = env.get(name, "").strip()
        return [x for x in raw.split(",") if x] or None
    cfg = {"mode": env["RFCBENCH_MODE"], "gpu": env["RFCBENCH_GPU_CLASS"],
           "fork": env["RFCBENCH_FORK_REPO"], "sha": env["RFCBENCH_FORK_SHA"],
           "mols": lst("RFCBENCH_MOLS"), "xcs": lst("RFCBENCH_XCS"),
           "bases": lst("RFCBENCH_BASES"), "bridge": env.get("RFCBENCH_BRIDGE", "") == "1",
           "repeats": int(env["RFCBENCH_REPEATS"]),
           "t0": float(env.get("RFCBENCH_WORK_T0") or time.time()),
           "work_cap": float(env["RFCBENCH_WORK_CAP"])}
    cfg["cells"] = S.select_cells(cfg["mode"], cfg["mols"], cfg["xcs"], cfg["bases"],
                                  cfg["bridge"])
    return cfg


# --------------------------------------------------------------------------------------------- #
# 1. card gate
# --------------------------------------------------------------------------------------------- #

def _gpu_line_model(line):
    """'GPU 0: <model> (UUID: ...)' -> '<model>' (exact, stripped); '' if not of that form."""
    head = line.strip()
    if ": " not in head:
        return ""
    model = head.split(": ", 1)[1]
    return model.rsplit(" (UUID", 1)[0].strip()


def card_problems(gpu, names, listing):
    """`names`: `nvidia-smi --query-gpu=name` lines; `listing`: `nvidia-smi -L` output. Pure."""
    gpu_lines = [ln for ln in listing.splitlines() if ln.strip().startswith("GPU ")]
    mig_lines = [ln for ln in listing.splitlines() if ln.strip().startswith("MIG ")]
    names = [n.strip() for n in names if n.strip()]
    if gpu == S.MIG_CLASS:
        probs = []
        want = S.MIG_PARENT_MODEL
        parent = [_gpu_line_model(ln) for ln in gpu_lines]
        if names != [want] or parent != [want]:
            probs.append(f"card: class {gpu} requires exactly one {want!r}; observed names "
                         f"{names}, `nvidia-smi -L` GPUs {gpu_lines}")
        if len(mig_lines) != 1:
            probs.append(f"card: expected exactly one MIG device, `nvidia-smi -L` shows "
                         f"{len(mig_lines)}: {mig_lines}")
        return probs
    want = S.CARD_MODEL.get(gpu)
    probs = []
    if not want or names != [want]:
        probs.append(f"card: observed {names}, class {gpu} requires exactly [{want!r}]")
    if len(gpu_lines) != 1 or mig_lines:
        probs.append(f"card: `nvidia-smi -L` shows {len(gpu_lines)} GPU(s) and {len(mig_lines)} "
                     f"MIG device(s); class {gpu} requires one whole card")
    return probs


def mig_memory_problems(gpu, total_bytes):
    if gpu != S.MIG_CLASS:
        return []
    lo, hi = S.MIG_MEM_GB
    try:
        gb = float(total_bytes) / 1e9
    except Exception:
        return [f"card: CUDA total memory {total_bytes!r} unreadable"]
    if not lo <= gb <= hi:
        return [f"card: CUDA total memory {gb:.2f} GB is outside the MIG band [{lo}, {hi}] GB"]
    return []


# --------------------------------------------------------------------------------------------- #
# 2-4. fetch, validated tree, base identity, overlay (g4pport_pod.py steps 1-2)
# --------------------------------------------------------------------------------------------- #

def fetch(cfg):
    shutil.rmtree(SRC, ignore_errors=True)
    os.makedirs(SRC)
    git("init", "-q")
    git("remote", "add", "origin", f"https://github.com/{cfg['fork']}")
    rcs = []
    for ref in (cfg["sha"],) + tuple(S.VALIDATED_FORK_SHAS):
        rcs.append(git("fetch", "-q", "--depth", "1", "origin", ref)[0])
    rcs.append(git("fetch", "-q", "--depth", "1", "origin",
                   f"refs/tags/{S.BASE_TAG}:refs/tags/{S.BASE_TAG}")[0])
    rcs.append(git("checkout", "-q", cfg["sha"])[0])
    head = git("rev-parse", "HEAD")[1].strip()
    base = git("rev-parse", f"{S.BASE_TAG}^{{commit}}")[1].strip()
    vals = [git("rev-parse", f"{v}^{{commit}}")[1].strip() for v in S.VALIDATED_FORK_SHAS]
    rec["git"] = {"head": head, "base": base, "validated": vals, "rc": rcs}
    ok = True
    checks = [(head, cfg["sha"], "fork HEAD"), (base, S.BASE_SHA, S.BASE_TAG)]
    checks += [(got, want, f"validated {want[:12]}")
               for got, want in zip(vals, S.VALIDATED_FORK_SHAS)]
    for got, want, what in checks:
        if got != want:
            problem(f"fetch: {what} resolves to {got[:12]!r}, not {want[:12]}")
            ok = False
    return ok


def validated_tree_problems(rc, out="", validated=None):
    """One `git diff --quiet VALIDATED FORK -- gpu4pyscf/`: rc 0 = identical, anything else refuses."""
    validated = validated or S.VALIDATED_FORK_SHA
    if rc == 0:
        return []
    if rc == 1:
        return [f"validated tree: the fork's gpu4pyscf/ differs from validated "
                f"{validated[:12]}; only a validated tree may be measured"]
    return [f"validated tree: git diff against {validated[:12]} failed rc={rc} "
            f"({out.strip()[-200:]})"]


def match_validated(mode, results):
    """`(matched, problems)` from `[(validated_sha, rc, out), ...]` in VALIDATED_FORK_SHAS order.
    Pure. `matched` is the first validated commit whose gpu4pyscf/ tree the fork's equals; a git error
    on ANY comparison, no match at all, or (for a MODE_REQUIRED_TREE mode) a match that is not the
    mode's own commit, is a problem. Fail closed: an empty `results` matches nothing."""
    matched, probs = None, []
    for sha, rc, out in results:
        if rc == 0:
            matched = matched or sha
        elif rc != 1:
            probs += validated_tree_problems(rc, out, sha)
    if matched is None:
        probs.append(f"validated tree: the fork's gpu4pyscf/ equals none of the validated trees "
                     f"{[r[0][:12] for r in results]}; only a validated tree may be measured")
    need = S.MODE_REQUIRED_TREE.get(mode)
    if need is not None and matched != need:
        probs.append(f"validated tree: mode {mode} needs the tree of {need[:12]} (its policy fields "
                     f"exist only there); the fork's tree matched "
                     f"{matched[:12] if matched else None}")
    return matched, probs


def validated_tree(cfg):
    results = []
    for v in S.VALIDATED_FORK_SHAS:
        rc, out = git("diff", "--quiet", v, cfg["sha"], "--", "gpu4pyscf/")
        results.append((v, rc, out))
    matched, probs = match_validated(cfg.get("mode"), results)
    rec["validated_tree_rc"] = {v[:12]: rc for v, rc, _ in results}
    rec["validated_match"] = matched
    emit(f"--- validated tree: matched {matched[:12] if matched else None} "
         f"(rc {rec['validated_tree_rc']})")
    for p in probs:
        problem(p)
    return not probs


def changed_files(cfg):
    rc, out = git("diff", "--name-status", S.BASE_TAG, cfg["sha"], "--", "gpu4pyscf/")
    files = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            files.append((parts[0][0], parts[-1]))
    rec["changed"] = files
    return files


def installed_dir():
    spec = importlib.util.find_spec("gpu4pyscf")      # locate WITHOUT importing
    return os.path.dirname(list(spec.submodule_search_locations)[0])


def check_overlay_base(files, site):
    ok = True
    try:
        ver = importlib.metadata.version("gpu4pyscf-cuda12x")
    except Exception as e:
        ver = f"unknown ({e})"
    rec["installed"] = {"site": site, "version": ver}
    if ver != "1.8.1":
        problem(f"base: installed gpu4pyscf-cuda12x is {ver}, not 1.8.1")
        ok = False
    for status, path in files:
        if "/tests/" in path:
            continue
        dst = os.path.join(site, path)
        if status == "M":
            base_bytes = subprocess.run(["git", "-C", SRC, "show", f"{S.BASE_TAG}:{path}"],
                                        capture_output=True).stdout
            try:
                with open(dst, "rb") as f:
                    inst = f.read()
            except OSError as e:
                problem(f"base: {path}: not in the installed wheel ({e})")
                ok = False
                continue
            if inst != base_bytes:
                problem(f"base: {path}: installed copy differs from {S.BASE_TAG}")
                ok = False
        elif status == "A":
            if os.path.exists(dst):
                problem(f"base: {path}: added by the branch but already in the wheel")
                ok = False
        else:
            problem(f"base: {path}: status {status} is not supported by the overlay")
            ok = False
    return ok


def overlay(files, site):
    done = []
    for status, path in files:
        if "/tests/" in path:
            continue
        with open(os.path.join(SRC, path), "rb") as f:
            data = f.read()
        dst = os.path.join(site, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        done.append({"path": path, "sha256": hashlib.sha256(data).hexdigest()[:16]})
    rec["overlaid"] = done
    return done


# --------------------------------------------------------------------------------------------- #
# 4b. the 14 lock pins and cuTENSOR, read off the installed distributions (G1)
# --------------------------------------------------------------------------------------------- #

def lock_pins(text):
    out = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            name, sep, ver = ln.partition("==")
            out[name.strip()] = ver.strip() if sep else None
    return out


def pin_problems(want, version_fn):
    """`(observed, problems)`: every pinned distribution's installed version must equal its pin. An
    unpinned lock line, a missing distribution or any other version is a problem, so an online
    FALLBACK install that resolved other versions FAILS rather than measuring a different stack."""
    observed, probs = {}, []
    if len(want) < 5:
        probs.append(f"pins: the lock holds only {len(want)} entries")
    for name, ver in sorted(want.items()):
        try:
            got = version_fn(name)
        except Exception as e:
            got = f"absent ({type(e).__name__})"
        observed[name] = {"want": ver, "got": got}
        if ver is None or got != ver:
            probs.append(f"pins: {name} is {got!r}, the lock pins {ver!r}")
    return observed, probs


def pins_step():
    try:
        with open(LOCK_PATH) as f:
            want = lock_pins(f.read())
    except OSError as e:
        problem(f"pins: cannot read {LOCK_PATH}: {e}")
        return
    name, _, ver = S.CUTENSOR_PIN.partition("==")
    want[name] = ver
    observed, probs = pin_problems(want, importlib.metadata.version)
    rec["pins"] = observed
    emit(f"--- pins: {len(observed)} checked, {len(probs)} mismatched")
    for p in probs:
        problem(p)


# --------------------------------------------------------------------------------------------- #
# 5. geometries
# --------------------------------------------------------------------------------------------- #

def geometry_problems(cells):
    """`{mol: problem}` for every requested molecule whose committed geometry is missing or, for the
    LARGE set, whose SHA256 is not the pinned one."""
    bad = {}
    for mol in sorted({c["mol"] for c in cells}):
        path = S.geometry_file(mol)
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError as e:
            bad[mol] = f"geometry {mol}: unreadable ({e})"
            continue
        if mol in S.LARGE_NAMES and digest != S.LARGE_SHA256[mol]:
            bad[mol] = (f"geometry {mol}: sha256 {digest[:16]} is not the pinned "
                        f"{S.LARGE_SHA256[mol][:16]}")
    return bad


# --------------------------------------------------------------------------------------------- #
# 6. environment probe (own process)
# --------------------------------------------------------------------------------------------- #

def probe_main():
    out = {"cutensor_preload": C.cutensor_preload()}
    out["contract_engine"] = C.contract_engine_info()
    try:
        from gpu4pyscf.dft.mixed_precision import MixedPrecision
        out["policy_reprs"] = C.policy_repr_table(MixedPrecision)
        try:                                   # PREREG-8 variants: the 09271907 tree only
            out["policy_reprs"].update(C.policy_repr_table(MixedPrecision, variants=True))
        except Exception as e:
            out["policy_variants_error"] = f"{type(e).__name__}: {e}"[:200]
    except Exception as e:
        out["policy_reprs"] = f"unavailable: {type(e).__name__}: {e}"[:200]
    try:
        import cupy
        free, total = cupy.cuda.runtime.memGetInfo()
        out["cuda_mem"] = {"free": int(free), "total": int(total)}
    except Exception as e:
        out["cuda_mem"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    try:
        import scf_headtohead.freeze_rfc_large  # noqa: F401
        import scf_headtohead.freeze_geometries  # noqa: F401
        out["rdkit_imported"] = "rdkit" in sys.modules
    except Exception as e:
        out["rdkit_imported"] = f"reader import failed: {type(e).__name__}: {e}"[:200]
    print("RFCBENCH_PROBE " + json.dumps(out, default=str), flush=True)
    return 0


def probe_problems(gpu, res, mode=None):
    """Pure verdict on the probe's JSON. The campaign reprs are always required; the PREREG-8 variant
    reprs too in a PREREG8_MODES mode (no other key is admitted)."""
    probs = [f"cutensor: {p}" for p in C.cutensor_problems(res.get("contract_engine") or {})]
    if res.get("rdkit_imported") is not False:
        probs.append(f"geometry readers: rdkit imported or reader failed "
                     f"({res.get('rdkit_imported')!r})")
    probs += mig_memory_problems(gpu, (res.get("cuda_mem") or {}).get("total"))
    reprs = res.get("policy_reprs")
    base = {f"{a}|{x}" for a in ("mixed", "cache") for x in S.XCS}
    variants = {f"{v}|{x}" for v in C.POLICY_VARIANTS for x in S.XCS}
    need = base | (variants if mode in S.PREREG8_MODES else set())
    if not (isinstance(reprs, dict) and need <= set(reprs) <= base | variants
            and all(isinstance(v, str) and v for v in reprs.values())):
        probs.append(f"policy: no expected MixedPrecision reprs from the real class ({reprs!r}; "
                     f"{res.get('policy_variants_error')})"[:300])
    return probs


def probe_step(cfg):
    try:
        rc, out = sh([sys.executable, "-u", os.path.abspath(__file__), "--probe"],
                     timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        rc, out = "timeout", ""
    lines = [ln for ln in out.splitlines() if ln.startswith("RFCBENCH_PROBE ")]
    if not lines:
        problem(f"probe produced no result (rc={rc}): {out.strip()[-300:]}")
        return
    try:
        res = json.loads(lines[-1][len("RFCBENCH_PROBE "):])
    except Exception as e:
        problem(f"probe result unreadable: {type(e).__name__}: {e}")
        return
    rec["probe"] = res
    emit(f"--- probe: engine={(res.get('contract_engine') or {}).get('engine')} "
         f"cuda_mem={res.get('cuda_mem')} rdkit_imported={res.get('rdkit_imported')}")
    for p in probe_problems(cfg["gpu"], res, cfg.get("mode")):
        problem(p)


# --------------------------------------------------------------------------------------------- #
# 7. groups
# --------------------------------------------------------------------------------------------- #

def remaining_s(cfg, now=None):
    now = time.time() if now is None else now
    return cfg["t0"] + cfg["work_cap"] - FINISH_RESERVE_S - now


def group_timeout_s(cfg, group, now=None):
    """min(remaining, 1.5 x the group's estimate), or None when too little is left to start."""
    left = remaining_s(cfg, now)
    if left < GROUP_MIN_S:
        return None
    return min(left, GROUP_TIMEOUT_FACTOR * S.group_estimate_s(group, cfg["repeats"], cfg["gpu"]))


def run_group(cfg, group, timeout, popen=subprocess.Popen):
    """Run one group; re-emit protocol lines as they arrive. Returns (lines, rc, diag)."""
    spec = json.dumps({"cells": group["cells"], "repeats": cfg["repeats"]})
    proc = popen([sys.executable, "-u", os.path.join(HERE, "rfcbench_cells.py"), "--group", spec],
                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, diag = [], []

    def reader():
        for ln in proc.stdout:
            ln = ln.rstrip("\n")
            if ln.startswith(PROTOCOL):
                lines.append(ln)
                emit(ln)
            else:
                diag.append(ln)
                del diag[:-GROUP_DIAG_LINES]
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        rc = "timeout"
    t.join(timeout=30)
    return lines, rc, diag


def parse_group(lines):
    """`(runs_by_key, ends_by_key, group_info, problems)` from a group's protocol lines."""
    runs, ends, info, probs = {}, {}, None, []
    for ln in lines:
        tag, _, body = ln.partition(" ")
        try:
            obj = json.loads(body)
        except Exception as e:
            probs.append(f"unreadable {tag} line: {type(e).__name__}")
            continue
        if tag == "RFCBENCH_CELL":
            for r in obj.get("runs") or []:
                r = dict(r)
                if "pair" in obj:
                    r["pair"] = obj["pair"]
                if "run" in obj:
                    r["run"] = obj["run"]
                runs.setdefault(obj.get("key"), []).append(r)
        elif tag == "RFCBENCH_CELLEND":
            k = obj.get("key")
            if k in ends:
                probs.append(f"{k}: reported finished twice")
            ends[k] = obj
        elif tag == "RFCBENCH_GROUP":
            info = obj
    return runs, ends, info, probs


def gpu_mem_line():
    try:
        return sh(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                   "--format=csv,noheader"], timeout=30)[1].strip()
    except Exception as e:
        return f"n/a ({type(e).__name__})"


# Disclosed only, never gated (PREREG-0 Amendment 2): C0 ran gpu4pyscf's stock VV10 kernel ~57 %
# slower than the g4pport pod on the same card model and inputs, and nothing recorded why.
GPU_STATE_FIELDS = ("clocks.sm", "clocks.max.sm", "clocks.mem", "power.draw", "power.limit",
                    "temperature.gpu", "pstate", "clocks_event_reasons.active")


def gpu_state():
    """{field: value} from one nvidia-smi query, or {"error": ...}. Never raises. Drivers that do not
    know `clocks_event_reasons.active` are retried without it."""
    for fields in (GPU_STATE_FIELDS, GPU_STATE_FIELDS[:-1]):
        try:
            rc, out = sh(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                          "--format=csv,noheader,nounits"], timeout=30)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:200]}
        vals = [v.strip() for v in out.strip().splitlines()[0].split(",")] if out.strip() else []
        if rc == 0 and len(vals) == len(fields):
            return dict(zip(fields, vals))
    return {"error": f"nvidia-smi rc={rc}: {out.strip()[:200]}"}


def report_cell(v):
    """The per-cell lines of the log. Bounded: one summary line, at most MAX_PRINTED_PROBLEMS problem
    lines and one 'more' line (rfcbench tail sizing depends on it)."""
    emit(f"--- cell {v['key']}: status={v['status']} fit={v['fit']} fails_run={v['fails_run']} "
         f"problems={len(v['problems'])} readings={json.dumps(v.get('readings') or {}, default=str)}")
    for p in v["problems"][:C.MAX_PRINTED_PROBLEMS]:
        emit(f"FAIL {p}"[:600])
    if len(v["problems"]) > C.MAX_PRINTED_PROBLEMS:
        emit(f"FAIL ... and {len(v['problems']) - C.MAX_PRINTED_PROBLEMS} more (in RFCBENCH_JSON)")


def measure(cfg, bad_geom):
    """Steps 7-8. Returns the verdicts in the requested order."""
    cells = cfg["cells"]
    keys = [S.cell_key(c) for c in cells]
    if len(set(keys)) != len(keys):
        problem(f"coverage: the requested cell list repeats a cell ({keys})")
    runs, ends, groups_rec = {}, {}, []
    for g in S.groups(cells):
        gkey = f"{g['kind']}/{g['xc']}" + (f"/{g['cells'][0]['mol']}" if g["kind"] == "geoopt"
                                           else "")
        todo = [c for c in g["cells"] if c["mol"] not in bad_geom]
        for c in g["cells"]:
            if c["mol"] in bad_geom:
                ends[S.cell_key(c)] = {"key": S.cell_key(c), "error": bad_geom[c["mol"]]}
        if not todo:
            continue
        sub = dict(g, cells=todo)
        timeout = group_timeout_s(cfg, sub)
        grec = {"group": gkey, "cells": len(todo), "timeout_s": timeout,
                "gpu_mem_before": gpu_mem_line(), "gpu_state_before": gpu_state()}
        emit(f"--- group {gkey}: {len(todo)} cells, timeout={timeout} mem={grec['gpu_mem_before']}")
        if timeout is None:
            grec["rc"] = "skipped: too little time left"
            groups_rec.append(grec)
            continue
        t0 = time.time()
        lines, rc, diag = run_group(cfg, sub, timeout)
        r, e, info, probs = parse_group(lines)
        grec.update(rc=rc, wall_s=round(time.time() - t0, 1), info=info,
                    gpu_state_after=gpu_state())
        for k, v in r.items():
            runs.setdefault(k, []).extend(v)
        for k, v in e.items():
            if k in ends:
                probs.append(f"{k}: reported finished twice")
            ends[k] = v
        for p in probs:
            problem(f"group {gkey}: {p}")
        if info is None:
            problem(f"group {gkey}: no RFCBENCH_GROUP line (rc={rc})")
        elif not isinstance(info.get("cutensor_problems"), list):
            problem(f"group {gkey}: cutensor: no cutensor_problems list in RFCBENCH_GROUP")
        else:
            for p in info["cutensor_problems"]:
                problem(f"group {gkey}: cutensor: {p}")
        unfinished = [S.cell_key(c) for c in todo if S.cell_key(c) not in e]
        emit(f"--- group {gkey}: rc={rc} wall={grec['wall_s']}s unfinished={len(unfinished)}")
        if rc != 0 or unfinished:
            for ln in diag:
                emit(f"    | {ln}"[:400])
        groups_rec.append(grec)
    rec["groups"] = groups_rec
    extra = sorted((set(runs) | set(ends)) - set(keys))
    if extra:
        problem(f"coverage: cells reported that were not requested: {extra}")
    verdicts = []
    for c in cells:
        k = S.cell_key(c)
        v = C.judge_cell(cfg["mode"], c, runs.get(k, []), ends.get(k), cfg["repeats"],
                         (rec.get("probe") or {}).get("policy_reprs"))
        verdicts.append(v)
        report_cell(v)
    rec["cells"] = verdicts
    return verdicts


def main(env=None):
    cfg = cfg_from_env(env)
    rec["cfg"] = {k: v for k, v in cfg.items() if k != "cells"}
    rec["cells_requested"] = [S.cell_key(c) for c in cfg["cells"]]
    emit(f"--- rfcbench mode={cfg['mode']} gpu={cfg['gpu']} cells={len(cfg['cells'])} "
         f"repeats={cfg['repeats']} fork={cfg['fork']}@{cfg['sha'][:12]}")
    if not cfg["cells"]:
        problem("coverage: the dispatch selects no cell")
        return finish()

    names = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])[1].splitlines()
    listing = sh(["nvidia-smi", "-L"])[1]
    rec["card"] = {"names": names, "listing": listing.strip().splitlines()}
    emit(f"--- card: {names} | {rec['card']['listing']}")
    probs = card_problems(cfg["gpu"], names, listing)
    for p in probs:
        problem(p)
    if probs:
        return finish()                         # a wrong card means no row

    if not fetch(cfg) or not validated_tree(cfg):
        return finish()
    files = changed_files(cfg)
    emit(f"--- changed under gpu4pyscf/: {files}")
    if not files:
        problem("base: the branch changes nothing under gpu4pyscf/")
        return finish()
    site = installed_dir()
    if not check_overlay_base(files, site):
        return finish()
    overlay(files, site)
    pins_step()

    bad_geom = geometry_problems(cfg["cells"])
    for p in bad_geom.values():
        problem(p)
    probe_step(cfg)
    if cfg["mode"] in S.EXTRA_MODES:            # PREREG-rfcbench-7: process-level phases
        import rfcbench_extra as XM
        if bad_geom:
            return finish()
        XM.measure(cfg, rec, emit, problem, lambda: remaining_s(cfg), gpu_state)
        return finish()
    measure(cfg, bad_geom)
    return finish()


def finish():
    cells = rec.get("cells") or []
    failing = [v["key"] for v in cells if v["fails_run"]]
    if failing:
        rec["problems"].append(f"{len(failing)} cell(s) fail the run: {failing[:8]}")
    ok = not rec["problems"]
    emit("RFCBENCH_JSON " + json.dumps(rec, default=str, separators=(",", ":")))
    emit("RFCBENCH_PASS" if ok else "RFCBENCH_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--probe"]:
        sys.exit(probe_main())
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        rec["problems"].append(f"harness exception: {type(e).__name__}: {e}")
        sys.exit(finish())
