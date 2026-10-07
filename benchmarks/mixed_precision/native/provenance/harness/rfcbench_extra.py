#!/usr/bin/env python3
"""Process-level measurements of PREREG-rfcbench-7: cold start, stage timing, concurrency.

The pod (rfcbench_pod.py) runs steps 1-6 as for every rfcbench mode (card gate, fork overlay, LARGE
geometry hashes, environment probe), then calls `measure` here instead of its per-group cell runner.
`measure` runs a fixed PLAN of phases. A phase is one or more fresh processes
(`python rfcbench_extra.py --phase <json>`) that build every SCF exactly as rfcbench_cells.build_and_run
does, and stream one `RFCBENCH_X {...}` line per SCF. Nothing is judged in a phase process.

  coldstart  CUPY_CACHE_DIR and CUDA_CACHE_PATH point at directories this module owns, wiped before the
             first phase. Phases, each a fresh process, each running paracetamol x {r2SCAN, B3LYP,
             wB97M-V} twice per arm:
               P1 stock first     (nothing has run on the pod's caches yet)
               P2 mixed first     (the first mixed SCFs of the pod)
               P3 mixed again     (a second process: both caches warm)
               P4 mixed after wiping the CuPy kernel cache only
               P5 mixed after wiping the CUDA driver JIT cache only
             Cache sizes are recorded after every phase.
  stages     One process per functional. Every SCF (arms mixed, cache, stock; pairs 0 and 1) is run
             with sync-to-sync timers around: XC (numint.nr_rks and the mixed-precision XC entry
             points), J/K (with_df.get_jk), the DF build (with_df.build) and the eigensolver (mf.eig).
             Times are EXCLUSIVE (a nested timed call is subtracted from its caller). The timers add
             device synchronisations: stage pods are for attribution, never for speed claims.
  concur     A whole card. For each arm separately: S, one process, the 6 cells twice; then C4, four
             processes started behind a barrier after a private untimed warm-up, the 6 cells once
             each; then, if an MPS control daemon can be started, C4 again under MPS.

The gate (`gate_problems`): every planned SCF present exactly once, no error, converged, and every
energy within 1e-8 Ha of the stock energy of the same molecule and functional on the pod.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rfcbench_sets as S  # noqa: E402

ROOT = "/tmp/rfc_extra"
CUPY_DIR = os.path.join(ROOT, "cupy_cache")
CUDA_DIR = os.path.join(ROOT, "cuda_cache")
MPS_PIPE = os.path.join(ROOT, "mps_pipe")
MPS_LOG = os.path.join(ROOT, "mps_log")
BARRIER_WAIT_S = 600
E_TOL = 1e-8
PHASE_DIAG_LINES = 15
CLEANUP_RESERVE_S = 150
# The only reasons a phase may be skipped without failing the pod (PREREG-rfcbench-7, CC1).
SKIP_REASONS_OK = ("MPS unavailable", "MPS unusable", "MPS not engaged (no nvidia-cuda-mps-server seen)")
SKIP_OOM_PREFIX = "out of memory under "     # joins, MPS stop, nvidia-smi after a timed-out phase
MPS_PROBE_TIMEOUT_S = 120
TAG = "RFCBENCH_X"

# --------------------------------------------------------------------------------------------- #
# the plan (pure)
# --------------------------------------------------------------------------------------------- #


def _run(cell, arm, seq, pair=None):
    return {"mol": cell["mol"], "xc": cell["xc"], "basis": cell["basis"], "aux": cell["aux"],
            "arm": arm, "seq": seq, "pair": pair}


def plan(mode, cells, procs=S.CONCUR_PROCS):
    """The phases of a mode, in order. Each: name, procs, runs (per process), wipe (cache dirs to
    empty first), mps, barrier, stage, warmup (runs per process before the barrier, untimed)."""
    phases = []
    if mode == "coldstart":
        def seq(arm):
            out = []
            for c in cells:
                for i in (0, 1):
                    out.append(_run(c, arm, len(out), pair=i))
            return out
        phases = [
            {"name": "P1_stock_first", "wipe": ["cupy", "cuda"], "runs": seq("stock")},
            {"name": "P2_mixed_first", "wipe": [], "runs": seq("mixed")},
            {"name": "P3_mixed_again", "wipe": [], "runs": seq("mixed")},
            {"name": "P4_mixed_wiped_cupy", "wipe": ["cupy"], "runs": seq("mixed")},
            {"name": "P5_mixed_wiped_cuda", "wipe": ["cuda"], "runs": seq("mixed")},
        ]
    elif mode == "stages":
        for xc in [x for x in S.XCS if any(c["xc"] == x for c in cells)]:
            runs = []
            for c in [c for c in cells if c["xc"] == xc]:
                for pair in (0, 1):
                    for arm in c["arms"]:
                        runs.append(_run(c, arm, len(runs), pair=pair))
            phases.append({"name": f"stages_{xc}", "wipe": [], "runs": runs, "stage": True})
    elif mode == "concur":
        for arm in ("mixed", "stock"):
            serial = []
            for i in (0, 1):
                for c in cells:
                    serial.append(_run(c, arm, len(serial), pair=i))
            once = [_run(c, arm, n, pair=1) for n, c in enumerate(cells)]
            warm = []
            for x in [x for x in S.XCS if any(c["xc"] == x for c in cells)]:
                warm.append(_run(next(c for c in cells if c["xc"] == x), arm, -1, pair=0))
            phases.append({"name": f"S_{arm}", "wipe": [], "runs": serial, "free_pool": True})
            phases.append({"name": f"C{procs}_{arm}", "procs": procs, "barrier": True,
                           "warmup": warm, "runs": once, "free_pool": True})
            phases.append({"name": f"C{procs}_{arm}_mps", "procs": procs, "barrier": True,
                           "warmup": warm, "runs": once, "mps": True, "free_pool": True})
    else:
        raise ValueError(f"no extra plan for mode {mode!r}")
    for p in phases:
        p.setdefault("procs", 1)
        p.setdefault("wipe", [])
        p.setdefault("barrier", False)
        p.setdefault("warmup", [])
        p.setdefault("mps", False)
        p.setdefault("stage", False)
        p.setdefault("free_pool", False)
    return phases


def expected_records(phases):
    """(phase, proc, seq) of every SCF the plan must produce."""
    out = []
    for p in phases:
        for proc in range(p["procs"]):
            for r in p["runs"]:
                out.append((p["name"], proc, r["seq"]))
    return out


def phase_estimate_s(p, gpu):
    """An over-estimate of one phase's work."""
    per = sum(S.stock_wall_s(r["mol"], r["xc"], r["basis"], gpu) for r in p["runs"] + p["warmup"])
    factor = 1.6 if p["stage"] else 1.0
    conc = p["procs"] if p["procs"] > 1 else 1      # pessimistic: concurrency buys nothing
    return S.GROUP_COLD_S + per * factor * conc + 5.0 * (len(p["runs"]) + len(p["warmup"]))


def estimate_s(mode, cells, gpu):
    """An over-estimate of the work, for the launcher's budget refusal (PREREG-0 section 5)."""
    return S.POD_FIXED_S + sum(phase_estimate_s(p, gpu) for p in plan(mode, cells))


def phase_cap_s(p, gpu, left):
    """A phase never takes more than twice its over-estimate plus 300 s, so one hang cannot eat the
    phases after it; and never more than the pod has left minus the cleanup reserve."""
    return min(left - CLEANUP_RESERVE_S, 2.0 * phase_estimate_s(p, gpu) + 300.0)


# --------------------------------------------------------------------------------------------- #
# the gate (pure)
# --------------------------------------------------------------------------------------------- #


def gate_problems(phases, records, require_treated=False):
    """Coverage, errors, convergence, |dE| vs stock, and the AO-cache fit read from the fork's own note
    (rfcbench_cells.classify_fit): an override or any non-fit-based note is a problem, and with
    `require_treated` (CS1, ST1: one process on a 96 GB card) so is any fallback."""
    import rfcbench_cells as C
    probs = []
    want = expected_records(phases)
    got = {}
    skipped = [r for r in records if r.get("skipped")]
    records = [r for r in records if not r.get("skipped")]
    for r in records:
        k = (r.get("phase"), r.get("proc"), r.get("seq"))
        if k in got:
            probs.append(f"{k}: reported twice")
        got[k] = r
    skipped_phases = {r.get("phase") for r in skipped if r.get("skipped") in SKIP_REASONS_OK
                      or str(r.get("skipped", "")).startswith(SKIP_OOM_PREFIX)}
    bad_skips = [r for r in skipped if r.get("phase") not in skipped_phases]
    if bad_skips:
        probs.append(f"skips with no allowed reason: {bad_skips[:2]}")
    missing = [k for k in want if k not in got and k[0] not in skipped_phases]
    if missing:
        probs.append(f"{len(missing)} planned SCF(s) missing, first {missing[:4]}")
    extra = sorted(set(got) - set(want), key=str)
    if extra:
        probs.append(f"unplanned records: {extra[:4]}")
    stock_e = {}
    for r in records:
        if r.get("error"):
            probs.append(f"{r.get('phase')}/{r.get('proc')}/{r.get('seq')} {r.get('mol')} {r.get('xc')} "
                         f"{r.get('arm')}: {r['error']}"[:300])
            continue
        if r.get("converged") is not True:
            probs.append(f"{r.get('phase')}/{r.get('proc')}/{r.get('seq')}: did not converge")
        if r.get("arm") == "stock" and isinstance(r.get("e"), float):
            stock_e.setdefault((r["mol"], r["xc"], r["basis"]), []).append(r["e"])
        if r.get("arm") in ("mixed", "cache"):
            tag = f"{r.get('phase')}/{r.get('proc')}/{r.get('seq')} {r.get('mol')} {r.get('xc')} {r['arm']}"
            try:
                fit = C.classify_fit(r["arm"], r["xc"], {"ao_cache_tier": r.get("tier"),
                                                         "ao_cache": r.get("ao_cache")})
                if require_treated and fit != C.TREATED:
                    probs.append(f"{tag}: fit {fit} where TREATED is required ({r.get('ao_cache')!r})"[:300])
            except (ValueError, KeyError) as e:
                probs.append(f"{tag}: {e}"[:300])
    ref = {k: statistics.median(v) for k, v in stock_e.items()}
    for r in records:
        if r.get("error") or not isinstance(r.get("e"), float):
            continue
        k = (r["mol"], r["xc"], r["basis"])
        if k not in ref:
            probs.append(f"{k}: no stock energy on this pod to compare with")
            continue
        if abs(r["e"] - ref[k]) > E_TOL:
            probs.append(f"{r['phase']}/{r['proc']}/{r['seq']} {k} {r['arm']}: |dE| "
                         f"{abs(r['e'] - ref[k]):.2e} > {E_TOL:g}")
    return probs


# --------------------------------------------------------------------------------------------- #
# readings (pure): used by the pod log and by rfcbench_extract
# --------------------------------------------------------------------------------------------- #


def readings(mode, records):
    ok = [r for r in records if not r.get("error") and not r.get("skipped")]
    out = {}
    if mode == "coldstart":
        for r in ok:
            if r["pair"] == 0:
                out[f"{r['phase']}/{r['xc']}/cold_s"] = round(r["wall_s"], 2)
            else:
                out[f"{r['phase']}/{r['xc']}/warm_s"] = round(r["wall_s"], 2)
    elif mode == "stages":
        for r in ok:
            if r["pair"] == 1:
                st = r.get("stages") or {}
                out[f"{r['mol']}/{r['xc']}/{r['arm']}"] = {k: round(v, 3) for k, v in st.items()}
    elif mode == "concur":
        # One time definition throughout: an SCF's span is t1 - t0 (build + kernel + record). T_S is
        # the summed span of the serial phase's warm second pass; T_C is the makespan of a concurrent
        # phase; G = procs * T_S / T_C.
        by = {}
        for r in ok:
            if isinstance(r.get("seq"), int) and r["seq"] >= 0:
                by.setdefault(r["phase"], []).append(r)
        for ph, rs in by.items():
            out[f"{ph}/scfs"] = len(rs)
            if ph.startswith("S_"):
                out[f"{ph}/T_S_s"] = round(sum(r["t1"] - r["t0"] for r in rs if r["pair"] == 1), 3)
            else:
                out[f"{ph}/makespan_s"] = round(max(r["t1"] for r in rs) - min(r["t0"] for r in rs), 3)
        for ph in list(by):
            if not ph.startswith("S_"):
                arm = ph.split("_")[1]
                procs = len({r["proc"] for r in by[ph]})
                ts = out.get(f"S_{arm}/T_S_s")
                tc = out.get(f"{ph}/makespan_s")
                if ts and tc:
                    out[f"{ph}/G"] = round(procs * ts / tc, 4)
    return out


# --------------------------------------------------------------------------------------------- #
# the phase process (GPU)
# --------------------------------------------------------------------------------------------- #


class StageTimer:
    """Exclusive sync-to-sync timers around named callables; nested timed calls are subtracted from
    their caller. `wrap` returns the wrapper; `totals` the seconds and call counts per stage."""

    def __init__(self, sync):
        self.sync = sync
        self.stack = []
        self.tot = {}
        self.calls = {}

    def wrap(self, name, fn):
        def wrapped(*a, **k):
            self.sync()
            t = time.perf_counter()
            self.stack.append(0.0)
            try:
                return fn(*a, **k)
            finally:
                self.sync()
                dt = time.perf_counter() - t
                child = self.stack.pop()
                self.tot[name] = self.tot.get(name, 0.0) + dt - child
                self.calls[name] = self.calls.get(name, 0) + 1
                if self.stack:
                    self.stack[-1] += dt
        return wrapped

    def totals(self):
        return {**{f"{k}_s": v for k, v in self.tot.items()}, **{f"{k}_n": v for k, v in self.calls.items()}}


def _stage_hook(timer):
    """Instance-level wrappers for J/K, the DF build and the eigensolver. XC is NOT wrapped here: an
    instance attribute named nr_rks makes the fork treat XC as user-overridden and switch the FP64 AO
    cache off (mixed_precision._nr_rks_overridden). XC is wrapped at module and class level by
    _patch_xc instead, which keeps that identity check true."""
    def hook(mf):
        wdf = mf.with_df
        wdf.get_jk = timer.wrap("jk", wdf.get_jk)
        wdf.build = timer.wrap("dfbuild", wdf.build)
        mf.eig = timer.wrap("eig", mf.eig)
    return hook


def _patch_xc(timer):
    """Wrap every XC entry point for the duration of one SCF and return an undo:
    numint.nr_rks and NumInt.nr_rks with ONE wrapper object (so `type(ni).nr_rks is numint.nr_rks`
    still holds and `vars(ni)` stays clean), and the mixed-precision entry points rks.get_veff looks
    up at call time. nr_rks_fp64_cached inlines nr_rks rather than calling it, so nothing nests."""
    from gpu4pyscf.dft import mixed_precision as mp
    from gpu4pyscf.dft import numint
    saved = [(numint, "nr_rks", numint.nr_rks), (numint.NumInt, "nr_rks", numint.NumInt.__dict__["nr_rks"])]
    w = timer.wrap("xc", numint.nr_rks)
    numint.nr_rks = w
    numint.NumInt.nr_rks = w
    for name in ("nr_rks_fp32", "nr_rks_fp64_cached"):
        if hasattr(mp, name):
            saved.append((mp, name, getattr(mp, name)))
            setattr(mp, name, timer.wrap("xc", getattr(mp, name)))

    def undo():
        for obj, name, fn in saved:
            setattr(obj, name, fn)
    return undo


def _one_scf(C, run, stage, free_pool=False):
    from gpu4pyscf.dft.mixed_precision import MixedPrecision
    import cupy
    kw = C.policy_kwargs(run["arm"], run["xc"])
    cell = {"kind": "speed", "mol": run["mol"], "xc": run["xc"], "basis": run["basis"],
            "aux": run["aux"], "arms": [run["arm"]]}
    atoms = C.read_atoms(run["mol"])
    timer = StageTimer(cupy.cuda.runtime.deviceSynchronize) if stage else None
    undo = _patch_xc(timer) if stage else (lambda: None)
    C.release_previous()
    t0 = time.time()
    try:
        mf, rec = C.build_and_run(atoms, cell, None if kw is None else MixedPrecision(**kw),
                                  hook=_stage_hook(timer) if stage else None)
        del mf
    finally:
        undo()
    t1 = time.time()
    pool = cupy.get_default_memory_pool()
    pool_bytes = int(pool.total_bytes())
    if free_pool:                    # CC1 only: four processes share one card
        pool.free_all_blocks()
    r = rec.get("rec") or {}
    st = rec.get("settings") or {}
    out = {"wall_s": rec["wall_s"], "e": rec["e"], "cycles": rec["cycles"], "converged": rec["converged"],
           "tier": r.get("ao_cache_tier"), "ao_cache": r.get("ao_cache"), "xc_phases": r.get("xc"),
           "k_phases": r.get("k"), "vv10_phases": r.get("vv10"), "cderi": st.get("cderi"),
           "ngrids": st.get("ngrids"), "pool_bytes": pool_bytes, "t0": t0, "t1": t1}
    if timer is not None:
        out["stages"] = timer.totals()
    return out


def phase_main(spec):
    import rfcbench_cells as C
    t_start = time.time()
    pre = C.cutensor_preload()
    info = C.contract_engine_info()
    C.emit("RFCBENCH_XPROC", {"phase": spec["name"], "proc": spec["proc"], "pid": os.getpid(),
                              "cutensor_preload": pre, "cutensor_problems": C.cutensor_problems(info),
                              "selftest_s": round(time.time() - t_start, 2),
                              "cupy_cache_after_selftest": dir_stats(os.environ.get("CUPY_CACHE_DIR", "")),
                              "cuda_cache_after_selftest": dir_stats(os.environ.get("CUDA_CACHE_PATH", ""))})
    for run in spec["warmup"]:
        out = dict(run, phase=spec["name"], proc=spec["proc"], seq=-1)
        try:
            out.update(_one_scf(C, run, False, spec.get("free_pool", False)))
        except Exception as e:
            out["error"] = f"warm-up: {type(e).__name__}: {e}"[:300]
        C.emit(TAG, out)
    if spec["barrier"]:
        open(os.path.join(spec["barrier_dir"], f"ready.{spec['proc']}"), "w").close()
        deadline = time.time() + BARRIER_WAIT_S
        while len(os.listdir(spec["barrier_dir"])) < spec["procs"] and time.time() < deadline:
            time.sleep(0.05)
    for run in spec["runs"]:
        out = dict(run, phase=spec["name"], proc=spec["proc"])
        try:
            out.update(_one_scf(C, run, spec["stage"], spec.get("free_pool", False)))
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"[:300]
        C.emit(TAG, out)
    return 0


# --------------------------------------------------------------------------------------------- #
# the orchestrator (pod)
# --------------------------------------------------------------------------------------------- #


def dir_stats(path):
    n = b = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            n += 1
            try:
                b += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return {"files": n, "bytes": b}


def _wipe(names):
    for n in names:
        d = {"cupy": CUPY_DIR, "cuda": CUDA_DIR}[n]
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)


def _env():
    env = dict(os.environ, CUPY_CACHE_DIR=CUPY_DIR, CUDA_CACHE_PATH=CUDA_DIR,
               CUDA_CACHE_MAXSIZE=str(4 << 30))
    return env


def mps_env():
    return dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE, CUDA_MPS_LOG_DIRECTORY=MPS_LOG)


def _mps_server_seen():
    """An nvidia-cuda-mps-server process, found by scanning /proc (pgrep may be absent)."""
    try:
        for pid in os.listdir("/proc"):
            if pid.isdigit():
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        if b"nvidia-cuda-mps-server" in f.read():
                            return True
                except OSError:
                    continue
    except OSError:
        pass
    return False


def mps_start(run=subprocess.run):
    """Start the control daemon detached (no inherited pipes, so this cannot block), then confirm it
    answers. Returns (ok, detail)."""
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG, exist_ok=True)
    env = mps_env()
    try:
        p = run(["nvidia-cuda-mps-control", "-d"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=env, timeout=30)
        if p.returncode != 0:
            return False, f"control -d rc={p.returncode}"
        q = run(["nvidia-cuda-mps-control"], input="get_server_list\n", text=True, capture_output=True,
                env=env, timeout=30)
        if q.returncode != 0:
            return False, f"get_server_list rc={q.returncode}: {(q.stdout + q.stderr)[-200:]}"
        return True, "control daemon answering"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:300]


def mps_probe(run=subprocess.run):
    """One short client under the MPS environment. A failure here means 'MPS unusable' without
    starting four clients that could hang on a broken server."""
    code = "import cupy; print(float(cupy.zeros(8).sum().get()))"
    try:
        p = run([sys.executable, "-c", code], env=mps_env(), capture_output=True, text=True,
                timeout=MPS_PROBE_TIMEOUT_S)
        return p.returncode == 0, (p.stdout + p.stderr)[-300:]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:300]


def mps_stop(run=subprocess.run):
    try:
        p = run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=mps_env(), timeout=30,
                capture_output=True)
        return f"rc={p.returncode}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:200]


def _sh(args, env=None, timeout=60):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr)
    except Exception as e:
        return 127, f"{type(e).__name__}: {e}"


def run_phase(phase, timeout, env, popen=subprocess.Popen, live=None):
    """Start the phase's processes together; return (records, proc_infos, rcs, diag). Every
    RFCBENCH_X line is passed to `live` as it arrives, so a work-cap kill keeps what had finished."""
    bdir = os.path.join(ROOT, "barrier", phase["name"])
    shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(bdir, exist_ok=True)
    procs, lines, lock = [], [], threading.Lock()
    for i in range(phase["procs"]):
        spec = dict(phase, proc=i, barrier_dir=bdir)
        procs.append(popen([sys.executable, "-u", os.path.abspath(__file__), "--phase", json.dumps(spec)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env))

    def reader(p):
        for ln in p.stdout:
            ln = ln.rstrip("\n")
            with lock:
                lines.append(ln)
            if live is not None and ln.startswith(TAG + " "):
                live(ln)
    threads = [threading.Thread(target=reader, args=(p,), daemon=True) for p in procs]
    for t in threads:
        t.start()
    deadline = time.time() + timeout
    rcs = []
    for p in procs:
        try:
            rcs.append(p.wait(timeout=max(1.0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            p.kill()
            rcs.append("killed")
    for t in threads:
        t.join(timeout=10)
    records, infos, diag = [], [], []
    for ln in lines:
        if ln.startswith(TAG + " "):
            try:
                records.append(json.loads(ln[len(TAG) + 1:]))
            except ValueError:
                diag.append(ln)
        elif ln.startswith("RFCBENCH_XPROC "):
            try:
                infos.append(json.loads(ln[len("RFCBENCH_XPROC "):]))
            except ValueError:
                diag.append(ln)
        else:
            diag.append(ln)
    return records, infos, rcs, diag[-PHASE_DIAG_LINES:]


def _is_oom(err):
    return isinstance(err, str) and ("OutOfMemory" in err or "out of memory" in err)


def measure(cfg, rec, emit, problem, remaining_s, gpu_state=None, popen=subprocess.Popen, sh=_sh,
            run=subprocess.run, server_seen=_mps_server_seen):
    mode = cfg["mode"]
    phases = plan(mode, cfg["cells"])
    os.makedirs(ROOT, exist_ok=True)
    for d in (CUPY_DIR, CUDA_DIR):
        os.makedirs(d, exist_ok=True)
    home = os.path.expanduser("~")
    prewarm = {"default_cupy_cache": dir_stats(os.path.join(home, ".cupy", "kernel_cache")),
               "default_cuda_cache": dir_stats(os.path.join(home, ".nv", "ComputeCache"))}
    records, phase_recs = [], []

    def skip(prec, reason):
        records.append({"phase": prec["phase"], "skipped": reason, "proc": None, "seq": None})
        phase_recs.append(prec)
        emit(f"--- phase {prec['phase']}: skipped: {reason}")

    for ph in phases:
        prec = {"phase": ph["name"], "procs": ph["procs"], "wipe": ph["wipe"]}
        if gpu_state is not None:
            prec["gpu_state_before"] = gpu_state()
        _wipe(ph["wipe"])
        env = _env()
        left = remaining_s()
        if left < 120 + CLEANUP_RESERVE_S:
            problem(f"phase {ph['name']}: skipped, {left:.0f}s left")
            phase_recs.append(dict(prec, rc="skipped: too little time left"))
            continue
        mps_on = ph["mps"]             # stop it whenever a start was attempted, even a half-failed one
        try:
            if ph["mps"]:
                ok, detail = mps_start(run)
                prec["mps_start"] = detail
                if not ok:
                    skip(prec, "MPS unavailable")
                    continue
                ok, detail = mps_probe(run)
                prec["mps_probe"] = detail
                if not ok:
                    skip(prec, "MPS unusable")
                    continue
                env.update(CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE, CUDA_MPS_LOG_DIRECTORY=MPS_LOG)
            t0 = time.time()
            cap = phase_cap_s(ph, cfg.get("gpu", "pro6000"), remaining_s())
            prec["cap_s"] = round(cap, 1)
            seen = {"v": False}
            done = threading.Event()

            def poll():                # the server may exit with its last client: look while it runs
                while not done.is_set():
                    if server_seen():
                        seen["v"] = True
                        return
                    done.wait(1.0)
            poller = threading.Thread(target=poll, daemon=True) if ph["mps"] else None
            if poller:
                poller.start()
            try:
                recs, infos, rcs, diag = run_phase(ph, cap, env, popen=popen, live=emit)
            finally:
                done.set()
                if poller:
                    poller.join(timeout=5)
            if ph["mps"]:
                prec["mps_server_seen"] = seen["v"] or server_seen()
        finally:
            if mps_on:
                prec["mps_stop"] = mps_stop(run)
                prec["compute_apps_after"] = sh(["nvidia-smi", "--query-compute-apps=pid,name",
                                                 "--format=csv,noheader"])[1][-300:]
        prec.update(wall_s=round(time.time() - t0, 1), rcs=rcs, procs_info=infos,
                    cupy_cache=dir_stats(CUPY_DIR), cuda_cache=dir_stats(CUDA_DIR))
        if gpu_state is not None:
            prec["gpu_state_after"] = gpu_state()
        timed_here = [r for r in recs if isinstance(r.get("seq"), int) and r["seq"] >= 0]
        if ph["mps"] and not any(not r.get("error") for r in timed_here):
            prec["mps_unusable"] = [r.get("error") for r in recs if r.get("error")][:4]
            skip(prec, "MPS unusable")
            continue
        if ph["mps"] and not prec.get("mps_server_seen"):
            skip(prec, "MPS not engaged (no nvidia-cuda-mps-server seen)")
            continue
        if ph["procs"] > 1 and any(_is_oom(r.get("error")) for r in recs):
            prec["oom"] = [r.get("error") for r in recs if _is_oom(r.get("error"))][:4]
            skip(prec, f"out of memory under {ph['procs']} processes")
            continue
        for inf in infos:
            for p in inf.get("cutensor_problems") or []:
                problem(f"phase {ph['name']} proc {inf.get('proc')}: cutensor: {p}")
        if len(infos) != ph["procs"]:
            problem(f"phase {ph['name']}: {len(infos)} of {ph['procs']} processes reported")
        emit(f"--- phase {ph['name']}: procs={ph['procs']} rc={rcs} wall={prec['wall_s']}s "
             f"scfs={len(timed_here)} cupy={prec['cupy_cache']} cuda={prec['cuda_cache']}")
        if any(rc != 0 for rc in rcs):
            for ln in diag:
                emit(f"    | {ln}"[:400])
        records.extend(recs)
        phase_recs.append(prec)
    timed = [r for r in records if not r.get("skipped") and isinstance(r.get("seq"), int) and r["seq"] >= 0]
    warm = [r for r in records if r.get("seq") == -1]
    skipped = [r for r in records if r.get("skipped")]
    require = mode in ("coldstart", "stages") and cfg.get("gpu") == "pro6000"
    for p in gate_problems(phases, timed + skipped, require_treated=require):
        problem(p)
    for r in warm:
        if r.get("error"):
            problem(f"{r['phase']}/{r['proc']}: {r['error']}")
    rec["extra"] = {"plan": [{k: v for k, v in p.items() if k not in ("runs", "warmup")} for p in phases],
                    "prewarm": prewarm, "phases": phase_recs, "runs": timed, "warmups": warm,
                    "skipped": skipped, "readings": readings(mode, timed)}
    emit(f"--- extra readings: {json.dumps(rec['extra']['readings'], default=str)[:3000]}")
    return records


if __name__ == "__main__":
    if sys.argv[1:2] == ["--phase"]:
        sys.exit(phase_main(json.loads(sys.argv[2])))
    print("usage: rfcbench_extra.py --phase <json>  (run by rfcbench_pod.py)")
    sys.exit(2)
