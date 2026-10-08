#!/usr/bin/env python3
"""Ephemeral RunPod GPU pod for the `rfcbench` RFC evidence campaign
(docs/upstream/PREREG-rfcbench-0-measurand.md and the scoped PREREG-rfcbench-1..5).

COPIED from `runpod_g4pport.py` (pre-warm read path, lock pins, base64 work delivery, RUN_ID check,
WORK_CAP < FAILSAFE < TIMEOUT_S) and changed in:
- the inputs: mode, gpu class, fork SHA, molecule / functional / basis filters, repeats, bridge --
  all closed sets, all validated HERE on the free runner before any pod or probe exists;
- the cascades: one single-model class per card, no fallback; `pro6000mig` empty until PREREG-5 is
  amended, so every mig dispatch is refused;
- the work estimate (rfcbench_sets.work_estimate_s) and the budget refusal at WORK_CAP - 300 s;
- the serialisation gate: any OTHER run of this repo `queued` or `in_progress` refuses the dispatch;
- the cuTENSOR install (from runpod_scfbench.py) and the per-run sentinel banking (`bank_sentinel`,
  `ci-results/runs/`, also from runpod_scfbench.py), plus the latest-result pointer;
- `mode=gputypes`: NO POD, EVER. The runner lists RunPod GPU types (PRO 6000 / MIG family) and banks
  the listing as a small sentinel with the same banking code.
Nothing here is shared with, or modifies, any other launcher (CLAUDE.md).

Env:
  RUNPOD_API_KEY, GITHUB_TOKEN and GIT_SHA (required); GITHUB_REPOSITORY, GITHUB_RUN_ID and
    GITHUB_RUN_ATTEMPT (set by Actions).
  RFCBENCH_MODE (required), RFCBENCH_GPU, RFCBENCH_FORK_SHA, RFCBENCH_FORK_REPO, RFCBENCH_MOLS,
  RFCBENCH_XCS, RFCBENCH_BASES, RFCBENCH_REPEATS, RFCBENCH_BRIDGE.

Exit 0 = a PASS sentinel was observed; non-zero otherwise.
"""
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rfcbench_sets as S  # noqa: E402   (stdlib only: this runner has no numpy)

RP_API = "https://rest.runpod.io/v1"
RP_GQL = "https://api.runpod.io/graphql"
GH_API = "https://api.github.com"
RP_KEY = os.environ["RUNPOD_API_KEY"]
GH_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ.get("GITHUB_REPOSITORY", "chris-lee-mc/gpu-conformer-engine")
SHA = os.environ["GIT_SHA"]
GH_RUN_ID = os.environ.get("GITHUB_RUN_ID", "")

RUN_ID = (f"{GH_RUN_ID or str(int(time.time()))}"
          f"-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}-{SHA[:8]}")

# The User-Agent is REQUIRED (see runpod_g4pport.py / runpod_gpucatalog.py): urllib's default UA is
# refused at RunPod's Cloudflare edge with HTTP 403 `error code: 1010`.
RP_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/126.0.0.0 Safari/537.36")
RP_HDR = {"Authorization": f"Bearer {RP_KEY}", "Content-Type": "application/json",
          "User-Agent": RP_UA}
GH_HDR = {"Authorization": f"Bearer {GH_TOKEN}",
          "Accept": "application/vnd.github+json",
          "Content-Type": "application/json"}

RESULT_BRANCH = "ci-bus"
RUN_RESULT_DIR = "ci-results/runs"
# Latest-result pointer and this dispatch's own immutable copy (runpod_scfbench.py, #181).
RESULT_PATH = f"ci-results/rfcbench-{SHA[:12]}.txt"
RUN_RESULT_PATH = f"{RUN_RESULT_DIR}/rfcbench-{SHA[:12]}-{RUN_ID}.txt"
GPUTYPES_RESULT_PATH = f"ci-results/rfcbench-gputypes-{SHA[:12]}.txt"
GPUTYPES_RUN_RESULT_PATH = f"{RUN_RESULT_DIR}/rfcbench-gputypes-{SHA[:12]}-{RUN_ID}.txt"

# --------------------------------------------------------------------------------------------- #
# budgets -- one budget, two derived caps (post-#139), as g4pport
# --------------------------------------------------------------------------------------------- #
TIMEOUT_S = 4800
WORK_CAP = max(120, TIMEOUT_S - 360)   # 300 s to clone/commit/push the bus + 60 s kill-after margin
FAILSAFE = max(180, TIMEOUT_S - 60)    # backstop only; the pod should self-stop first
DISK_GB = 80                            # the LARGE set's DF tensors may spill to host
# PREREG-0 section 5: the free runner refuses a dispatch whose over-estimate exceeds this.
BUDGET_MARGIN_S = 300
BUDGET_LIMIT_S = WORK_CAP - BUDGET_MARGIN_S
# Pre-flight free-VRAM floor: refuse a card another tenant already fills. 8 GiB covers the trio's
# largest cache (10.3 GiB is NOT required to fit: a cache that does not fit is classified, not
# crashed) plus one DF SCF; the MIG instance is ~24 GB in total.
VRAM_MIN_MB = 8192

TRANSIENT_NET = (urllib.error.URLError, ConnectionError, TimeoutError, OSError)
MAX_CONSECUTIVE_NET_ERRORS = 10
DELETE_ATTEMPTS = 4

# --------------------------------------------------------------------------------------------- #
# sentinel window -- the worst case the emitters can produce, not the average (defect family 10)
# --------------------------------------------------------------------------------------------- #
# Per cell: up to REPEATS_MAX+1 pair lines (a downstream cell: 4 run lines + 1 controls line; a
# bridge cell: 1), one CELLEND line, and rfcbench_pod.report_cell's summary + at most
# MAX_PRINTED_PROBLEMS (12) problem lines + one 'more' line = 21; 24 with slack for a duplicate or
# unreadable line. Per group: two '--- group' lines, the RFCBENCH_GROUP line, a cuTENSOR problem
# line or two and up to GROUP_DIAG_LINES (15) lines of the group's own output = 20; 24 with slack.
# Fixed: the out.txt preamble (prewarm, install tail, clone, nvidia-smi, cuTENSOR: ~20), the pod's
# header / card / fetch / per-file base / pins (up to 16) / probe / geometry lines (~80 at the
# worst plausible fork),
# a traceback (~40), RFCBENCH_JSON + the verdict, and the work script's trailer (~12). A host test
# drives the real emitters at their all-FAIL worst case against this window.
TAIL_FLOOR = 60
TAIL_PER_CELL = 24
TAIL_PER_GROUP = 24
TAIL_FIXED = 180


def tail_lines(n_cells: int, n_groups: int) -> int:
    return max(TAIL_FLOOR, TAIL_PER_CELL * int(n_cells) + TAIL_PER_GROUP * int(n_groups)
               + TAIL_FIXED)


# --------------------------------------------------------------------------------------------- #
# pre-warm keys -- READ ONLY (copied from runpod_g4pport.py)
# --------------------------------------------------------------------------------------------- #
LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scfbench_requirements.lock")
PREWARM_PY_TAG = "py310"
PREWARM_CUDA_TAG = "cu128"


def lock_text() -> str:
    try:
        with open(LOCK_PATH) as fh:
            text = fh.read()
    except OSError as e:
        raise SystemExit(f"::error::cannot read the requirements lock at {LOCK_PATH}: {e}")
    reqs = [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    if len(reqs) < 5:
        raise SystemExit(f"::error::{LOCK_PATH} holds only {len(reqs)} requirement(s)")
    loose = [r for r in reqs if "==" not in r]
    if loose:
        raise SystemExit(f"::error::{LOCK_PATH} is not exactly pinned: {loose!r}")
    return text


def lock_pins() -> dict:
    out = {}
    for ln in lock_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "==" in ln:
            name, _, ver = ln.partition("==")
            out[name.strip()] = ver.strip()
    return out


def req_hash() -> str:
    return hashlib.sha256(lock_text().encode()).hexdigest()[:8]


def wheelhouse_tag() -> str:
    return f"prewarm-wh-{req_hash()}-{PREWARM_PY_TAG}-{PREWARM_CUDA_TAG}"


def kernel_cache_tag_prefix() -> str:
    pins = lock_pins()
    try:
        cupy = pins["cupy-cuda12x"]
    except KeyError:
        raise SystemExit(f"::error::{LOCK_PATH} pins no cupy-cuda12x")
    return f"prewarm-kc-cupy{cupy}-{PREWARM_CUDA_TAG}-sm"


# --------------------------------------------------------------------------------------------- #
# validation -- all of it on the FREE runner, before any pod or probe exists
# --------------------------------------------------------------------------------------------- #

def _env(name):
    return os.environ.get(name, "").strip()


def resolve_mode() -> str:
    mode = _env("RFCBENCH_MODE")
    if mode not in S.MODES:
        raise SystemExit(f"::error::RFCBENCH_MODE={mode!r} is not one of {S.MODES}")
    return mode


def resolve_gpu(mode: str) -> str:
    want = _env("RFCBENCH_GPU")
    allowed = S.MODE_CLASSES[mode]
    if mode == "gputypes":
        if want:
            raise SystemExit("::error::RFCBENCH_GPU must be blank for mode=gputypes (no pod)")
        return ""
    if not want:
        if len(allowed) != 1:
            raise SystemExit(f"::error::mode={mode} needs an explicit RFCBENCH_GPU, one of {allowed}")
        want = allowed[0]
    if want not in S.GPU_CASCADES:
        raise SystemExit(f"::error::RFCBENCH_GPU={want!r} is not one of {sorted(S.GPU_CASCADES)}")
    if want not in allowed:
        raise SystemExit(f"::error::mode={mode} runs on {allowed} only, not {want!r}")
    if not S.GPU_CASCADES[want]:
        raise SystemExit(
            f"::error::RFCBENCH_GPU={want!r} has an EMPTY cascade: the RunPod GPU type id of the RTX "
            f"PRO 6000 MIG instance must be pinned by a PREREG-rfcbench-5 amendment first (read it "
            f"with mode=gputypes, which launches no pod). No mig dispatch is possible until then.")
    return want


def resolve_fork() -> tuple:
    repo = _env("RFCBENCH_FORK_REPO") or S.FORK_REPO_DEFAULT
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise SystemExit(f"::error::RFCBENCH_FORK_REPO={repo!r} is not owner/name")
    sha = _env("RFCBENCH_FORK_SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"::error::RFCBENCH_FORK_SHA={sha!r} must be a full 40-hex commit SHA")
    return repo, sha


def _list(name: str, mode: str, allowed: list):
    raw = _env(name)
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",")]
    if mode in S.FIXED_MODES and tuple(items) != S.FIXED_MODE_RERUNS.get((mode, name)):
        raise SystemExit(f"::error::{name} must be blank in mode={mode}: its cells are fixed by "
                         f"its PREREG (the only admitted re-run filter is "
                         f"{S.FIXED_MODE_RERUNS.get((mode, name))})")
    if any(not x for x in items):
        raise SystemExit(f"::error::{name}={raw!r} has an empty entry")
    dupes = sorted({x for x in items if items.count(x) > 1})
    if dupes:
        raise SystemExit(f"::error::{name} repeats {dupes}")
    for x in items:
        if x not in allowed:
            raise SystemExit(f"::error::{name}={x!r} is not admissible in mode={mode}: "
                             f"{sorted(allowed)}")
    return items


def resolve_repeats() -> int:
    raw = _env("RFCBENCH_REPEATS")
    if not raw:
        return S.REPEATS_DEFAULT
    if not re.fullmatch(r"[0-9]+", raw):
        raise SystemExit(f"::error::RFCBENCH_REPEATS={raw!r} is not an int")
    n = int(raw)
    if not S.REPEATS_MIN <= n <= S.REPEATS_MAX:
        raise SystemExit(f"::error::RFCBENCH_REPEATS={n} is outside "
                         f"[{S.REPEATS_MIN}, {S.REPEATS_MAX}]")
    return n


# R3: process gates, on the free runner. A mode whose PREREG still owes an amendment is refused.
DOCS = os.path.join(S.REPO_ROOT, "docs", "upstream")
PREREG3_PATH = os.path.join(DOCS, "PREREG-rfcbench-3-large-basis.md")
PREREG5_PATH = os.path.join(DOCS, "PREREG-rfcbench-5-mig.md")
PREREG6_PATH = os.path.join(DOCS, "PREREG-rfcbench-6-mig-fit.md")
PREREG7_PATH = os.path.join(DOCS, "PREREG-rfcbench-7-coldstart-stages-concurrency.md")
PREREG8_PATH = os.path.join(DOCS, "PREREG-rfcbench-8-geometry-safety.md")


def amendment_text(path) -> str:
    """The text of a PREREG between its first amendments heading and `## RESULT`: where a
    dated amendment lives. Empty when either heading is missing or the file is unreadable."""
    try:
        text = open(path).read()
    except OSError:
        return ""
    head, sep, _ = text.partition("\n## RESULT")
    if not sep:
        return ""
    # `## Amendments` or `## Amendment 1 (...)`, never `## Amendment rule` (which states the rule).
    m = re.search(r"^## Amendment(s[ \t]*$| [0-9]| \()", head, flags=re.M)
    return head[m.start():] if m else ""


def process_gate(mode: str, gpu: str):
    if mode == "large":
        amend = amendment_text(PREREG3_PATH)
        absent = [n for n, h in S.LARGE_SHA256.items() if h not in amend]
        if absent:
            raise SystemExit(
                f"::error::mode=large is refused until PREREG-rfcbench-3 carries, in an amendment "
                f"above RESULT, every pinned LARGE SHA256 (rfcbench_sets.LARGE_SHA256); missing "
                f"for {absent}")
    if mode == "mig":
        amend = amendment_text(PREREG5_PATH)
        for cls in S.MODE_CLASSES["mig"]:
            cascade = S.GPU_CASCADES.get(cls) or []
            if not cascade or not cascade[0] or f"`{cascade[0]}`" not in amend:
                raise SystemExit(
                    f"::error::mode=mig is refused until the {cls} cascade is pinned AND "
                    f"PREREG-rfcbench-5 carries that exact RunPod type id, in backticks, in an "
                    f"amendment above RESULT (cascade={cascade!r})")


def prereg7_gate():
    """The PREREG-rfcbench-7 modes run only while that protocol is committed with no RESULT yet."""
    try:
        text = open(PREREG7_PATH).read()
    except OSError:
        raise SystemExit("::error::this mode is refused: PREREG-rfcbench-7 is not committed")
    m = re.search(r"^## RESULT[ \t]*$", text, flags=re.M)
    if not m:
        raise SystemExit("::error::this mode is refused: PREREG-rfcbench-7 has no RESULT heading")
    if text[m.end():].strip():
        raise SystemExit("::error::this mode is refused: PREREG-rfcbench-7 already has a RESULT")


def prereg8_gate():
    """The PREREG-rfcbench-8 modes (attrib, geoopt) run only while that protocol is committed with no
    RESULT yet."""
    try:
        text = open(PREREG8_PATH).read()
    except OSError:
        raise SystemExit("::error::this mode is refused: PREREG-rfcbench-8 is not committed")
    m = re.search(r"^## RESULT[ \t]*$", text, flags=re.M)
    if not m:
        raise SystemExit("::error::this mode is refused: PREREG-rfcbench-8 has no RESULT heading")
    if text[m.end():].strip():
        raise SystemExit("::error::this mode is refused: PREREG-rfcbench-8 already has a RESULT")


def required_tree_gate(mode: str, sha: str):
    """A mode whose policy fields exist on one validated tree only (S.MODE_REQUIRED_TREE). The free
    runner cannot read trees, so it requires that exact commit; the pod then checks the tree."""
    need = S.MODE_REQUIRED_TREE.get(mode)
    if need is not None and sha != need:
        raise SystemExit(f"::error::mode={mode} needs RFCBENCH_FORK_SHA={need} (its policy fields "
                         f"exist only on that tree), not {sha}")


def migfit_gate():
    """mode=migfit runs only while PREREG-rfcbench-6 is committed with no RESULT yet written, and its
    pre-registered text (above RESULT) names both cascade ids in backticks and every geometry SHA256
    of the molecules it measures."""
    try:
        text = open(PREREG6_PATH).read()
    except OSError:
        raise SystemExit("::error::mode=migfit is refused: PREREG-rfcbench-6 is not committed")
    m = re.search(r"^## RESULT[ \t]*$", text, flags=re.M)
    if not m:
        raise SystemExit("::error::mode=migfit is refused: PREREG-rfcbench-6 has no RESULT heading")
    head, tail = text[:m.start()], text[m.end():]
    if tail.strip():
        raise SystemExit("::error::mode=migfit is refused: PREREG-rfcbench-6 already has a RESULT")
    absent = [f"`{S.GPU_CASCADES[c][0]}`" for c in S.MODE_CLASSES["migfit"]
              if not S.GPU_CASCADES.get(c) or f"`{S.GPU_CASCADES[c][0]}`" not in head]
    absent += [m for m in S.MIGFIT_MOLS if S.LARGE_SHA256[m] not in head]
    if absent:
        raise SystemExit(f"::error::mode=migfit is refused: PREREG-rfcbench-6 does not pin {absent}")


def resolve_cfg() -> dict:
    """Every input, validated, and the dispatch's cells and budget. Raises SystemExit on any
    refusal; nothing has been spent when it does."""
    mode = resolve_mode()
    gpu = resolve_gpu(mode)
    if mode == "gputypes":
        for name in ("RFCBENCH_MOLS", "RFCBENCH_XCS", "RFCBENCH_BASES", "RFCBENCH_BRIDGE",
                     "RFCBENCH_REPEATS"):
            if _env(name):
                raise SystemExit(f"::error::{name} must be blank for mode=gputypes (no pod)")
        return {"mode": mode, "gpu": "", "cells": []}
    process_gate(mode, gpu)
    if mode == "migfit":
        migfit_gate()
    if mode in S.EXTRA_MODES:
        prereg7_gate()
    if mode in S.PREREG8_MODES:
        prereg8_gate()
        if _env("RFCBENCH_REPEATS"):
            raise SystemExit(f"::error::RFCBENCH_REPEATS must be blank in mode={mode}: it runs "
                             f"no warm pairs (PREREG-rfcbench-8)")
    repo, sha = resolve_fork()
    required_tree_gate(mode, sha)
    adm = S.admissible(mode)
    mols = _list("RFCBENCH_MOLS", mode, adm["mols"])
    xcs = _list("RFCBENCH_XCS", mode, adm["xcs"])
    bases = _list("RFCBENCH_BASES", mode, adm["bases"])
    if mols is not None:
        mols = S.expand_mols(mode, mols)
        dupes = sorted({m for m in mols if mols.count(m) > 1})
        if dupes:
            raise SystemExit(f"::error::RFCBENCH_MOLS names {dupes} twice (directly and by tier)")
    if xcs is None and mode in S.DEFAULT_XCS:
        xcs = list(S.DEFAULT_XCS[mode])
    bridge = _env("RFCBENCH_BRIDGE")
    if bridge not in ("", "1"):
        raise SystemExit(f"::error::RFCBENCH_BRIDGE={bridge!r} is not one of ('', '1')")
    if bridge and mode != "ladder":
        raise SystemExit("::error::RFCBENCH_BRIDGE=1 is a ladder-only leg (PREREG-1)")
    if bridge and xcs is not None and "r2scan" not in xcs:
        raise SystemExit("::error::RFCBENCH_BRIDGE=1 needs r2scan among RFCBENCH_XCS")
    repeats = resolve_repeats()
    cells = S.select_cells(mode, mols, xcs, bases, bridge == "1")
    if not cells:
        raise SystemExit(f"::error::the filters select no cell of mode={mode}")
    for mol in sorted({c["mol"] for c in cells}):
        if not os.path.isfile(S.geometry_file(mol)):
            raise SystemExit(f"::error::{mol} has no committed geometry at {S.geometry_file(mol)}")
    if mode in S.EXTRA_MODES:
        import rfcbench_extra as XM
        est = XM.estimate_s(mode, cells, gpu)
        n_groups = len(XM.plan(mode, cells))
    else:
        est = S.work_estimate_s(cells, repeats, gpu)
        n_groups = len(S.groups(cells))
    if est > BUDGET_LIMIT_S:
        raise SystemExit(
            f"::error::this dispatch's over-estimate is {est:.0f}s ({len(cells)} cells, "
            f"{n_groups} groups, R={repeats}, gpu={gpu}), above WORK_CAP - "
            f"{BUDGET_MARGIN_S} = {BUDGET_LIMIT_S}s (PREREG-0 section 5). Split it: a narrower "
            f"RFCBENCH_MOLS / RFCBENCH_XCS selects fewer cells.")
    return {"mode": mode, "gpu": gpu, "fork_repo": repo, "fork_sha": sha, "mols": mols,
            "xcs": xcs, "bases": bases, "bridge": bridge, "repeats": repeats, "cells": cells,
            "n_groups": n_groups, "work_estimate_s": est}


def _csv(x):
    return ",".join(x) if x else ""


def _env_exports(cfg) -> str:
    """The pod's RFCBENCH_* exports. Every value is from a closed set validated above."""
    rows = [("RFCBENCH_MODE", cfg["mode"]), ("RFCBENCH_GPU_CLASS", cfg["gpu"]),
            ("RFCBENCH_FORK_REPO", cfg["fork_repo"]), ("RFCBENCH_FORK_SHA", cfg["fork_sha"]),
            ("RFCBENCH_MOLS", _csv(cfg["mols"])), ("RFCBENCH_XCS", _csv(cfg["xcs"])),
            ("RFCBENCH_BASES", _csv(cfg["bases"])), ("RFCBENCH_BRIDGE", cfg["bridge"]),
            ("RFCBENCH_REPEATS", str(cfg["repeats"])), ("RFCBENCH_WORK_CAP", str(WORK_CAP))]
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in rows)


# --------------------------------------------------------------------------------------------- #
# the pod's work script
# --------------------------------------------------------------------------------------------- #
# (Never name a template placeholder literally in a comment anywhere in this string: the assembly
# macro-substitutes EVERY occurrence, comments included -- defect family 11.)
RFCBENCH_WORK = r"""
export RFCBENCH_WORK_T0=$(date -u +%s)
@@PREWARM_BLOCK@@
# cuTENSOR (PREREG-0 section 1: on, and observed), copied from runpod_scfbench.py: installed after
# the locked install, never part of the lock, its lib dir on the loader path BEFORE any python
# starts. Whether it LOADED is read off gpu4pyscf's own module and gated by rfcbench_pod.py.
python3 -m pip install -q --no-cache-dir "__CUTENSOR_PIN__" >> /tmp/install.log 2>&1
RC_CUTENSOR=$?
CT_VER=$(python3 -c "import importlib.metadata as m; print(m.version('cutensor-cu12'))" 2>/dev/null || echo absent)
CT_LIB=$(python3 -c "import importlib.metadata as m; print(m.distribution('cutensor-cu12').locate_file('cutensor/lib'))" 2>/dev/null || echo "")
export LD_LIBRARY_PATH="${CT_LIB:+$CT_LIB:}/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "=== cutensor install rc=$RC_CUTENSOR pin=__CUTENSOR_PIN__ installed=cutensor-cu12==$CT_VER lib=${CT_LIB:-absent} ===" >> /tmp/out.txt 2>&1

git clone https://x-access-token:${GH_TOKEN}@github.com/__REPO__.git /tmp/r >/dev/null 2>&1
cd /tmp/r && git checkout -q __SHA__
rc_clone=$?

cd /tmp/r
export PYTHONPATH=/tmp/r/src
export OMP_NUM_THREADS=8
__ENV_EXPORTS__

{ echo "=== install.log tail ==="; tail -8 /tmp/install.log;
  echo "=== clone_rc=$rc_clone ===";
  echo "=== nvidia-smi ==="; nvidia-smi 2>&1 | head -6; } >> /tmp/out.txt 2>&1

# -u: unbuffered, so a cap kill cannot discard what has already been printed.
python3 -u /tmp/r/.github/scripts/rfcbench_pod.py >> /tmp/out.txt 2>&1
RUN_RC=$?
# THE VERDICT: matched whole (grep -x) in the last 3 lines, and RUN_RC must agree.
VERDICT=$(tail -n 3 /tmp/out.txt | grep -x 'RFCBENCH_PASS' | tail -n 1)
LAST_LINE=$(tail -n 1 /tmp/out.txt | cut -c1-200)
if [ "$RUN_RC" -eq 0 ] && [ "$VERDICT" = "RFCBENCH_PASS" ]; then
  echo PASS > /tmp/status
else
  echo FAIL > /tmp/status
fi
{
  echo "=== rc: selfupgrade=${rc_selfupgrade:-U} base=${rc_base:-U} g4p=${rc_g4p:-U} cutensor=${RC_CUTENSOR:-U} clone=${rc_clone:-U} run=$RUN_RC ==="
  if [ "${rc_g4p:-1}" != "0" ]; then
    echo "=== gpu4pyscf install FAILED: install.log tail -12 ==="
    tail -12 /tmp/install.log
  fi
  echo "=== verdict: run_rc=$RUN_RC status=$(cat /tmp/status) token_in_last3=[$VERDICT] last_line=[$LAST_LINE] ==="
  echo "=== prewarm: $PREWARM_LINE ==="
  echo "=== gpu: class=__GPU_CLASS__ head=__GPU_HEAD__ observed=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
} >> /tmp/out.txt 2>&1
"""

# bank_sentinel, copied from runpod_scfbench.py: one sentinel, two copies (this run's immutable
# per-run file + the latest-result pointer), one commit; BANK_REFUSED on a per-run collision. The
# SAME text banks the pod's sentinel and, on the runner, the gputypes listing.
BANK_SH = r"""
bank_sentinel(){
  # $1 sentinel file, $2 checkout dir, $3 remote url. Returns 0 banked; 2 REFUSED because the
  # per-run path already exists (a per-run sentinel is never overwritten); 1 did not land.
  local src="$1" bus="$2" remote="$3" i status
  status=$(sed -n 's/^STATUS=//p' "$src" | head -1)
  if git ls-remote --exit-code --heads "$remote" __RESULT_BRANCH__ >/dev/null 2>&1; then
    git clone --branch __RESULT_BRANCH__ --depth 1 "$remote" "$bus" >/dev/null 2>&1 || return 1
    cd "$bus" || return 1
  else
    git clone --depth 1 "$remote" "$bus" >/dev/null 2>&1 || return 1
    cd "$bus" || return 1
    git checkout --orphan __RESULT_BRANCH__ >/dev/null 2>&1
    git rm -rf . >/dev/null 2>&1
  fi
  git config user.email ci@runpod && git config user.name "RunPod CI"
  for i in 1 2 3 4 5; do
    if [ -e "__RUN_RESULT_PATH__" ]; then
      echo "BANK_REFUSED: __RUN_RESULT_PATH__ is already on the bus (RUN_ID collision); not overwriting it"
      return 2
    fi
    mkdir -p ci-results __RUN_RESULT_DIR__
    cp "$src" __RUN_RESULT_PATH__ && cp "$src" __RESULT_PATH__ || return 1
    git add __RUN_RESULT_PATH__ __RESULT_PATH__
    git commit -q -m "ci-bus: __BANK_LABEL__ ${status:-FAIL} for __SHA7__ (run __RUN_ID__)" >/dev/null 2>&1 || return 1
    git push -q origin __RESULT_BRANCH__ >/dev/null 2>&1 && return 0
    # Rejected: another writer pushed first. Rebuild on the new remote head (no rebase); the
    # per-run guard above re-checks against the tree actually being pushed onto.
    git fetch -q origin __RESULT_BRANCH__ >/dev/null 2>&1 && git reset -q --hard FETCH_HEAD >/dev/null 2>&1
    sleep 3
  done
  echo "BANK_FAILED: __RUN_RESULT_PATH__ did not land in 5 attempts"
  return 1
}
"""


def bank_sh(run_path: str, pointer_path: str, label: str) -> str:
    return (BANK_SH.replace("__RESULT_BRANCH__", RESULT_BRANCH)
            .replace("__RUN_RESULT_DIR__", RUN_RESULT_DIR)
            .replace("__RUN_RESULT_PATH__", run_path)
            .replace("__RESULT_PATH__", pointer_path)
            .replace("__BANK_LABEL__", label)
            .replace("__SHA7__", SHA[:7])
            .replace("__RUN_ID__", RUN_ID))


# The pod script (from runpod_g4pport.py's POD_TEMPLATE, post-#139), with scfbench's banking.
POD_TEMPLATE = r"""
set +e
POD_T0=$(date -u +%s)
: > /tmp/pod.txt
( while :; do read -r _u _ < /proc/uptime; echo "$_u"; sleep 5; done >> /tmp/hb.txt ) 2>/dev/null &
pmark(){ read -r _up _ < /proc/uptime; echo "[pod +$(( $(date -u +%s) - POD_T0 ))s mono=${_up%.*}s] $*" >> /tmp/pod.txt; }
pmark "start $(date -u) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
pmark "host nproc=$(nproc 2>/dev/null) cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | tr '\n' ' ')"
nohup bash -c 'sleep __FAILSAFE__ && runpodctl stop pod $RUNPOD_POD_ID' >/tmp/k.log 2>&1 &
pmark "preamble: failsafe backgrounded"
GH_TOKEN=$(printf '%s' "$GH_TOKEN_B64" | base64 -d)
export GH_TOKEN
pmark "preamble: token decoded"
# The work script arrives base64-encoded in the pod's environment (an inline literal stalled a
# median of 210 s; delivery through env measured 0 s).
printf '%s' "$WORK_B64" | base64 -d > /tmp/work.sh
pmark "preamble: work.sh decoded from env ($(wc -c < /tmp/work.sh) bytes)"
VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
VRAM_TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
pmark "vram free=${VRAM_FREE_MB:-?}MB of ${VRAM_TOTAL_MB:-?}MB"
pmark "heartbeat max gap=$(awk 'NR>1{d=$1-p; if(d>m)m=d} {p=$1} END{printf "%d", m+0}' /tmp/hb.txt 2>/dev/null)s of $(wc -l < /tmp/hb.txt 2>/dev/null) ticks"
if [ -n "$VRAM_FREE_MB" ] && [ "$VRAM_FREE_MB" -lt __VRAM_MIN_MB__ ] 2>/dev/null; then
  pmark "ABORT: only ${VRAM_FREE_MB}MB free -- another tenant holds this GPU; not computing on it"
  echo "POD_PREFLIGHT_FAIL: ${VRAM_FREE_MB}MB free of ${VRAM_TOTAL_MB}MB, need >=__VRAM_MIN_MB__MB" > /tmp/out.txt
  echo FAIL > /tmp/status
  WORK_RC=99
else
pmark "work start (cap __WORK_CAP__s)"
timeout --kill-after=60 __WORK_CAP__ bash /tmp/work.sh
WORK_RC=$?
fi
pmark "work end rc=$WORK_RC"
if [ $WORK_RC -eq 124 ] || [ $WORK_RC -eq 137 ]; then
  pmark "WORK HIT THE __WORK_CAP__s CAP -- the log above is where it stopped, not where it failed"
fi
if [ -s /tmp/out.txt ]; then
  pmark "out.txt $(wc -l < /tmp/out.txt) lines -- the work block ran; read the tail"
else
  pmark "out.txt ABSENT/EMPTY -- died before the prewarm block finished: apt/pip or provisioning"
fi
STATUS=$(cat /tmp/status 2>/dev/null || echo FAIL)
if [ "$WORK_RC" != "0" ]; then STATUS=FAIL; fi
{ echo "STATUS=$STATUS"; echo "RUN_ID=__RUN_ID__"; echo "MODE=rfcbench"; echo "SHA=__SHA__";
  echo "FORK_SHA=__FORK_SHA__"; echo "GPU_CLASS=__GPU_CLASS__"; echo "RFC_MODE=__RFC_MODE__";
  echo "XCS=__XCS__"; echo "MOLS=__MOLS__"; echo "BASIS=__BASES__"; echo "REPEATS=__REPEATS__";
  echo "WORK_RC=$WORK_RC"; echo "---";
  cat /tmp/out.txt 2>/dev/null | tail -__TAIL_LINES__; echo "--- pod ---"; cat /tmp/pod.txt 2>/dev/null; } > /tmp/sentinel.txt
__BANK_SH__
bank_sentinel /tmp/sentinel.txt /tmp/bus "https://x-access-token:${GH_TOKEN}@github.com/__REPO__.git"
echo "bank rc=$?"
runpodctl stop pod $RUNPOD_POD_ID 2>/dev/null || true
sleep infinity
"""


def rp(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{RP_API}{path}", data=data, headers=RP_HDR, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            t = r.read().decode(); return r.status, (json.loads(t) if t else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:2000]}


def gh(method, path, body=None, accept=None):
    data = json.dumps(body).encode() if body is not None else None
    hdr = dict(GH_HDR, Accept=accept) if accept else GH_HDR
    req = urllib.request.Request(f"{GH_API}{path}", data=data, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            t = r.read().decode(errors="replace")
            if accept and "raw" in accept:
                return r.status, t
            return r.status, (json.loads(t) if t else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:300]}


# --------------------------------------------------------------------------------------------- #
# serialisation gate (PREREG-0 sections 5-6; CLAUDE.md "check both")
# --------------------------------------------------------------------------------------------- #

def other_active_runs(gh_fn=None):
    """`(runs, error)`: every OTHER workflow run of this repo that is `queued` or `in_progress`.
    `error` is set when either listing could not be read -- the caller refuses then too, because an
    unreadable listing is not evidence that nothing is running."""
    gh_fn = gh_fn or gh
    found = []
    for status in ("queued", "in_progress"):
        code, resp = gh_fn("GET", f"/repos/{REPO}/actions/runs?status={status}&per_page=100")
        if code != 200 or not isinstance(resp, dict) or not isinstance(resp.get("workflow_runs"),
                                                                         list):
            return None, f"listing ?status={status} unreadable (HTTP {code})"
        for r in resp["workflow_runs"]:
            if str(r.get("id")) == str(GH_RUN_ID):
                continue
            found.append({"id": r.get("id"), "name": r.get("name"), "status": status})
        # Only a FULL page means there may be more. GitHub's total_count can lag for a moment (this
        # very run moving from queued to in_progress), and reading "count > listed" as "more
        # than one page" refused D1's first dispatch (run 37013682529) with nothing queued.
        if len(resp["workflow_runs"]) >= 100:
            found.append({"id": None, "name": "more than one page", "status": status})
    return found, None


def serialisation_gate(gh_fn=None):
    runs, err = other_active_runs(gh_fn)
    if err:
        raise SystemExit(f"::error::serialisation gate: {err}; refusing to launch anything")
    if runs:
        raise SystemExit(f"::error::serialisation gate: {len(runs)} other run(s) queued or in "
                         f"progress: {runs[:5]}. One GPU job at a time; re-dispatch when they "
                         f"finish.")


# --------------------------------------------------------------------------------------------- #
# mode=gputypes -- no pod, ever
# --------------------------------------------------------------------------------------------- #
GPUTYPES_QUERY = "{ gpuTypes { id displayName memoryInGb secureCloud communityCloud } }"
GPUTYPES_MATCH = re.compile(r"PRO 6000|6000|MIG", re.IGNORECASE)
GPUTYPES_REST_PATHS = ("/gputypes", "/gpuTypes", "/gpu-types")


def _rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        d = payload.get("data")
        if isinstance(d, dict) and isinstance(d.get("gpuTypes"), list):
            return d["gpuTypes"]
        for key in ("data", "gpuTypes", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _rp_req(method, url, body=None):
    """(status, payload). Never prints, never returns headers; the key is only in RP_HDR."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=RP_HDR, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            t = r.read().decode(errors="replace")
            return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode(errors="replace") or "{}")
        except ValueError:
            return e.code, {}
    except Exception:
        return 0, {}


def gputypes_listing(req=None):
    """`(status, lines)`: GraphQL first, then the REST listing paths of the launcher's API family.
    PASS only on a non-empty catalog: an empty one is INCONCLUSIVE, never 'no MIG type exists'."""
    req = req or _rp_req
    lines, rows, source = [], [], None
    code, payload = req("POST", RP_GQL, {"query": GPUTYPES_QUERY})
    rows = _rows(payload)
    lines.append(f"GRAPHQL HTTP {code} entries={len(rows)}")
    if code == 200 and rows:
        source = "graphql"
    else:
        for path in GPUTYPES_REST_PATHS:
            code, payload = req("GET", f"{RP_API}{path}")
            rows = _rows(payload)
            lines.append(f"REST {path} HTTP {code} entries={len(rows)}")
            if code == 200 and rows:
                source = f"rest {path}"
                break
    if not source:
        lines.append("GPUTYPES INCONCLUSIVE: no catalog could be read (this is not evidence of "
                     "absence)")
        return "FAIL", lines
    lines.append(f"GPUTYPES source={source} count={len(rows)}")
    n = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        gid, name = str(r.get("id", "")), str(r.get("displayName", ""))
        if GPUTYPES_MATCH.search(gid) or GPUTYPES_MATCH.search(name):
            n += 1
            lines.append(f"GPUTYPE id={gid!r} displayName={name!r} memoryInGb={r.get('memoryInGb')!r} "
                         f"secureCloud={r.get('secureCloud')!r} "
                         f"communityCloud={r.get('communityCloud')!r}")
    lines.append(f"GPUTYPES matched={n} of {len(rows)} (pattern PRO 6000|6000|MIG, case-insensitive)")
    return "PASS", lines


def gputypes_sentinel(status, lines) -> str:
    head = [f"STATUS={status}", f"RUN_ID={RUN_ID}", "MODE=rfcbench", f"SHA={SHA}",
            "RFC_MODE=gputypes", "---"]
    return "\n".join(head + list(lines)) + "\n"


def bank_on_runner(text: str, run_path: str, pointer_path: str, label: str, run=subprocess.run):
    """Bank a sentinel from the RUNNER with the pod's own `bank_sentinel` text. Returns its rc."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "sentinel.txt")
        with open(src, "w") as f:
            f.write(text)
        script = (bank_sh(run_path, pointer_path, label)
                  + f'\nbank_sentinel {shlex.quote(src)} {shlex.quote(os.path.join(d, "bus"))} '
                    f'"https://x-access-token:${{GH_TOKEN}}@github.com/{REPO}.git"\n')
        env = dict(os.environ, GH_TOKEN=GH_TOKEN)
        r = run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=300)
        return r.returncode


def main_gputypes(req=None, bank=None):
    status, lines = gputypes_listing(req)
    for ln in lines:
        print(ln)
    text = gputypes_sentinel(status, lines)
    rc = (bank or bank_on_runner)(text, GPUTYPES_RUN_RESULT_PATH, GPUTYPES_RESULT_PATH,
                                  "rfcbench-gputypes")
    print(f"::notice::[gputypes] banked rc={rc} at {GPUTYPES_RUN_RESULT_PATH}@{RESULT_BRANCH}")
    return 0 if status == "PASS" and rc == 0 else 1


# --------------------------------------------------------------------------------------------- #
# the pod
# --------------------------------------------------------------------------------------------- #

def prewarm_block() -> str:
    return PREWARM_BLOCK


def vram_min_mb(gpu):
    return VRAM_MIN_MB


def build_script(cfg):
    """The pod script. The work script is NOT embedded here; it travels as base64 in the pod env."""
    return (POD_TEMPLATE
            .replace("__BANK_SH__", bank_sh(RUN_RESULT_PATH, RESULT_PATH, "rfcbench"))
            .replace("__FAILSAFE__", str(FAILSAFE))
            .replace("__WORK_CAP__", str(WORK_CAP))
            .replace("__VRAM_MIN_MB__", str(vram_min_mb(cfg["gpu"])))
            .replace("__REPO__", REPO)
            .replace("__SHA__", SHA)
            .replace("__FORK_SHA__", cfg["fork_sha"])
            .replace("__GPU_CLASS__", cfg["gpu"])
            .replace("__RFC_MODE__", cfg["mode"])
            .replace("__XCS__", _csv(cfg["xcs"]))
            .replace("__MOLS__", _csv(cfg["mols"]))
            .replace("__BASES__", _csv(cfg["bases"]))
            .replace("__REPEATS__", str(cfg["repeats"]))
            .replace("__TAIL_LINES__", str(tail_lines(len(cfg["cells"]), cfg["n_groups"])))
            .replace("__RUN_ID__", RUN_ID))


def work_text(cfg) -> str:
    return (RFCBENCH_WORK
            .replace("@@PREWARM_BLOCK@@", prewarm_block())
            .replace("__REPO__", REPO)
            .replace("__SHA__", SHA)
            .replace("__REQHASH__", req_hash())
            .replace("__WH_TAG__", wheelhouse_tag())
            .replace("__KC_TAG_PREFIX__", kernel_cache_tag_prefix())
            .replace("__CUTENSOR_PIN__", S.CUTENSOR_PIN)
            .replace("__GPU_CLASS__", cfg["gpu"])
            .replace("__GPU_HEAD__", shlex.quote(S.GPU_CASCADES[cfg["gpu"]][0]))
            .replace("__ENV_EXPORTS__", _env_exports(cfg)))


def work_b64(cfg):
    return base64.b64encode(work_text(cfg).encode()).decode()


def pod_env(cfg, token_b64):
    return {"GH_TOKEN_B64": token_b64, "WORK_B64": work_b64(cfg),
            "LOCK_B64": base64.b64encode(lock_text().encode()).decode()}


# RunPod's REST v1 create-pod schema does not list the PRO 6000 MIG type (its gpuTypeIds enum skips
# "...Server Edition MIG 1g.24gb"; run 37094752419), although GraphQL's gpuTypes lists it (run
# 36951667656). These classes are created through GraphQL podFindAndDeployOnDemand instead; every
# other class keeps the REST path unchanged. PREREG-rfcbench-5 Amendment 2.
GQL_DEPLOY_CLASSES = ("pro6000mig",)
GQL_DEPLOY = ("mutation Deploy($input: PodFindAndDeployOnDemandInput) "
              "{ podFindAndDeployOnDemand(input: $input) { id machineId } }")
GQL_TERMINATE = "mutation Terminate($input: PodTerminateInput!) { podTerminate(input: $input) }"
# The REST path hands the bootstrap script over as dockerStartCmd ["bash", "-c", script]. GraphQL
# takes one dockerArgs string, so the same script travels base64 in the env (as WORK_B64 already
# does) and this fixed command decodes and runs it. It parses the same whether RunPod splits the
# string into argv or hands it to `sh -c`.
GQL_BOOT_CMD = "bash -c 'echo \"$BOOT_B64\" | base64 -d > /tmp/boot.sh && exec bash /tmp/boot.sh'"


def gql_deploy_input(cfg, script, env, gpu, cloud):
    env = dict(env, BOOT_B64=base64.b64encode(script.encode()).decode())
    return {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu,
            "name": f"ci-rfcbench-{cfg['mode']}-{SHA[:7]}",
            "imageName": "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            "containerDiskInGb": DISK_GB, "volumeInGb": 0,
            "dockerArgs": GQL_BOOT_CMD,
            "env": [{"key": k, "value": v} for k, v in sorted(env.items())]}


def _gql_error(code, payload):
    errs = payload.get("errors") if isinstance(payload, dict) else None
    text = json.dumps(errs) if errs else json.dumps(payload)[:2000]
    return f"HTTP {code}: " + " ".join(text.split())[:1500]


def gql_launch_one(cfg, script, env, gpu, cloud, req=None):
    """(pod id or None, error text). Never raises, never prints the key."""
    req = req or _rp_req
    code, payload = req("POST", RP_GQL, {"query": GQL_DEPLOY, "variables": {
        "input": gql_deploy_input(cfg, script, env, gpu, cloud)}})
    pod = None
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        pod = payload["data"].get("podFindAndDeployOnDemand")
    if code == 200 and isinstance(pod, dict) and pod.get("id") and not payload.get("errors"):
        return str(pod["id"]), ""
    return None, _gql_error(code, payload)


def launch(cfg, token_b64, rp_fn=None, gql_req=None):
    rp_fn = rp_fn or rp
    script = build_script(cfg)
    env = pod_env(cfg, token_b64)
    cascade = S.GPU_CASCADES[cfg["gpu"]]
    if not cascade:                                   # resolve_gpu refuses first; belt and braces
        raise SystemExit(f"::error::empty cascade for {cfg['gpu']}")
    print(f"::notice::[rfcbench] mode={cfg['mode']} gpu={cfg['gpu']} head={cascade[0]} "
          f"fork={cfg['fork_repo']}@{cfg['fork_sha'][:12]} cells={len(cfg['cells'])} "
          f"groups={cfg['n_groups']} R={cfg['repeats']} est={cfg['work_estimate_s']:.0f}s "
          f"(limit {BUDGET_LIMIT_S}s) tail={tail_lines(len(cfg['cells']), cfg['n_groups'])}")
    for cloud in ("SECURE", "COMMUNITY"):
        for gpu in cascade:
            if cfg["gpu"] in GQL_DEPLOY_CLASSES:
                pid, err = gql_launch_one(cfg, script, env, gpu, cloud, req=gql_req)
                if pid:
                    print(f"::notice::[rfcbench] launched pod {pid} on {gpu} ({cloud}, graphql)")
                    return pid
                print(f"::warning::{gpu} unavailable ({cloud}, graphql): {err}")
                continue
            status, resp = rp_fn("POST", "/pods", {
                "name": f"ci-rfcbench-{cfg['mode']}-{SHA[:7]}",
                "imageName": "nvidia/cuda:12.8.1-devel-ubuntu22.04",
                "gpuTypeIds": [gpu], "gpuCount": 1, "cloudType": cloud,
                "containerDiskInGb": DISK_GB, "volumeInGb": 0,
                "dockerStartCmd": ["bash", "-c", script],
                "env": env,
            })
            if status in (200, 201):
                print(f"::notice::[rfcbench] launched pod {resp.get('id')} on {gpu} ({cloud})")
                return resp.get("id")
            # The whole RunPod error, on one line: M1a (run 37074824098) was refused with a request
            # schema error cut to 80 characters, so its cause could not be read. RunPod's error
            # body never carries the API key (it is sent only in the Authorization header).
            err = " ".join(str(resp.get("error", "")).split())[:1500]
            print(f"::warning::{gpu} unavailable ({cloud}, HTTP {status}): {err}")
    return None


def poll_result():
    """THIS dispatch's per-run sentinel, read RAW (no 1 MB contents-API ceiling), accepted only when
    its line 2 is this run's RUN_ID."""
    status, text = gh("GET", f"/repos/{REPO}/contents/{RUN_RESULT_PATH}?ref={RESULT_BRANCH}",
                      accept="application/vnd.github.raw")
    if status == 200 and isinstance(text, str):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[1].strip() == f"RUN_ID={RUN_ID}":
            return text
    return None


def bus_diagnosis():
    st, listing = gh("GET", f"/repos/{REPO}/contents/{RUN_RESULT_DIR}?ref={RESULT_BRANCH}")
    names = ([e.get("name", "") for e in listing]
             if st == 200 and isinstance(listing, list) else [])
    me = os.path.basename(RUN_RESULT_PATH)
    return (f"bus lists {len(names)} per-run sentinels; this run's {RUN_RESULT_PATH} "
            f"{'IS PRESENT (read-path failure, not a timeout)' if me in names else 'is absent'}")


def main():
    cfg = resolve_cfg()                         # every refusal is free and happens first
    serialisation_gate()                        # ... and so does this one
    if cfg["mode"] == "gputypes":
        sys.exit(main_gputypes())
    print(f"::notice::mode=rfcbench/{cfg['mode']} sha={SHA[:12]} result={RUN_RESULT_PATH}@"
          f"{RESULT_BRANCH} (latest pointer {RESULT_PATH})")
    gh_token_b64 = base64.b64encode(GH_TOKEN.encode()).decode()
    pid = None
    t_start = time.time()
    try:
        pid = launch(cfg, gh_token_b64)
        if not pid:
            print("::error::No GPU available in cascade (single-model class: no fallback)")
            sys.exit(1)
        deadline = time.time() + TIMEOUT_S
        result, polls, net_errors = None, 0, 0
        while time.time() < deadline:
            time.sleep(20)
            polls += 1
            try:
                result = poll_result()
                if result:
                    print(f"::notice::[runner] sentinel seen after {polls} polls, "
                          f"{time.time() - t_start:.0f}s since start")
                    break
                _, pod = rp("GET", f"/pods/{pid}")
                exited_empty = (pod or {}).get("desiredStatus") == "EXITED" and not poll_result()
                net_errors = 0
            except TRANSIENT_NET as e:
                net_errors += 1
                result = None
                print(f"::warning::[runner] poll {polls}: network error {type(e).__name__} "
                      f"({net_errors}/{MAX_CONSECUTIVE_NET_ERRORS} consecutive)")
                if net_errors >= MAX_CONSECUTIVE_NET_ERRORS:
                    print("::error::[runner] API unreachable for too many consecutive polls")
                    break
                continue
            if exited_empty:
                print(f"::notice::[runner] pod EXITED at {time.time() - t_start:.0f}s "
                      f"with no sentinel visible after {polls} polls")
                time.sleep(15)
                result = poll_result()
                break
        if not result:
            print(f"::notice::[runner] gave up after {polls} polls / {time.time() - t_start:.0f}s; "
                  f"{bus_diagnosis()}")
            print(f"::error::no result sentinel within {TIMEOUT_S}s")
            sys.exit(1)
        print("::group::pod result")
        print(result)
        print("::endgroup::")
        passed = result.splitlines()[0].strip() == "STATUS=PASS"
        print(f"::notice::result = {'PASS' if passed else 'FAIL'}")
        sys.exit(0 if passed else 1)
    finally:
        if pid:
            delete_pod(pid)


def delete_pod(pid, rp_fn=None, gql_req=None, sleep=time.sleep):
    """Delete pod `pid`, VERIFIED: REST DELETE must answer 2xx, else GraphQL podTerminate must
    answer without errors. A pod created through GraphQL (GQL_DEPLOY_CLASSES) must never be left
    running because the REST delete silently did not apply to it. Returns True when deleted."""
    rp_fn = rp_fn or rp
    gql_req = gql_req or _rp_req
    for attempt in range(1, DELETE_ATTEMPTS + 1):
        try:
            status, _ = rp_fn("DELETE", f"/pods/{pid}")
            if 200 <= int(status) < 300:
                print(f"::notice::deleted pod {pid}")
                return True
            code, payload = gql_req("POST", RP_GQL, {"query": GQL_TERMINATE,
                                                      "variables": {"input": {"podId": pid}}})
            if code == 200 and isinstance(payload, dict) and not payload.get("errors"):
                print(f"::notice::terminated pod {pid} (graphql; REST delete answered {status})")
                return True
            print(f"::warning::[runner] delete attempt {attempt} for pod {pid}: REST {status}, "
                  f"{_gql_error(code, payload)}")
        except TRANSIENT_NET as e:
            print(f"::warning::[runner] delete attempt {attempt} for pod {pid}: "
                  f"{type(e).__name__}")
        if attempt < DELETE_ATTEMPTS:
            sleep(5)
    print(f"::error::[runner] could NOT delete pod {pid} -- delete it by hand")
    return False


# The pre-warm READ block, copied verbatim from runpod_g4pport.py's work script (lines 526-730 at
# 17be2f31) except: no `geometric` online-fallback install (rfcbench never optimises a geometry), and
# the labels say rfcbench. It ends by CREATING /tmp/out.txt with the prewarm line.
PREWARM_BLOCK = r"""
PW_T0=$(date -u +%s)
apt-get update -qq && apt-get install -y -qq git python3-pip curl >/dev/null 2>&1
T_APT=$(( $(date -u +%s) - PW_T0 ))
export CUDA_PATH=/usr/local/cuda
export PATH="$CUDA_PATH/bin:$PATH"

PW_REQHASH=__REQHASH__
PW_WH_TAG=__WH_TAG__
PW_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
[ -z "$PW_ARCH" ] && PW_ARCH=unknown
PW_KC_TAG="__KC_TAG_PREFIX__$PW_ARCH"
PW_WH_ASSET=wheelhouse.tar
PW_KC_ASSET=kc.tar.gz
PW_WH=MISS
PW_KC=MISS
PW_WH_BYTES=0
PW_KC_BYTES=0
PW_KC_FILES=0
T_DL=0
T_INSTALL=0
T_KC=0
PW_CACHE_DIR="${HOME:-/root}/.cupy/kernel_cache"
# The lock arrives in the pod ENV, like the work script and the token, and is decoded here. It is
# never materialised inline: an inline literal stalled a median of 210 s (up to 2569 s) for reasons
# still unnamed, and delivery through `env` measured 0 s.
printf '%s' "$LOCK_B64" | base64 -d > /tmp/req.lock
PW_LOCK_PINS=$(grep -v '^[[:space:]]*#' /tmp/req.lock 2>/dev/null | grep -c '==')
[ -z "$PW_LOCK_PINS" ] && PW_LOCK_PINS=0

# Download asset $2 of release tag $1 to $3. Checked on curl's OWN exit status, never on the file
# merely being non-empty: `-o` keeps whatever bytes arrived, so an interrupted transfer leaves a
# large, non-empty, WRONG file that a size test accepts and the digest then rejects as if the stored
# asset were corrupt. Two attempts SHARE ONE 1800 s budget, each with a per-attempt 900 s cap and a
# 50 KiB/s speed floor so a zero-byte stall cannot eat the whole budget in one try.
pw_get(){
  pw_tag="$1"; pw_asset="$2"; pw_dest="$3"
  curl -sS -m 120 -H "Authorization: Bearer ${GH_TOKEN}" \
       -H "Accept: application/vnd.github+json" \
       "https://api.github.com/repos/__REPO__/releases/tags/${pw_tag}" > /tmp/pw_rel.json 2>/dev/null
  [ -s /tmp/pw_rel.json ] || return 1
  pw_id=$(python3 -c "
import json, sys
try:
    d = json.load(open('/tmp/pw_rel.json'))
except Exception:
    raise SystemExit(0)
for a in (d.get('assets') or []):
    if a.get('name') == sys.argv[1]:
        print(a.get('id'))
        break
" "$pw_asset" 2>/dev/null)
  [ -z "$pw_id" ] && return 1
  pw_try=1
  pw_end=$(( $(date -u +%s) + 1800 ))
  while [ "$pw_try" -le 2 ]; do
    rm -f "$pw_dest"
    pw_left=$(( pw_end - $(date -u +%s) ))
    if [ "$pw_left" -lt 60 ]; then
      echo "pw_get asset=$pw_asset tag=$pw_tag try=$pw_try abandoned: ${pw_left}s left of the shared 1800s budget" >&2
      break
    fi
    pw_m="$pw_left"; [ "$pw_m" -gt 900 ] && pw_m=900
    curl -sSL --fail -m "$pw_m" --speed-time 60 --speed-limit 51200 \
         -H "Authorization: Bearer ${GH_TOKEN}" \
         -H "Accept: application/octet-stream" \
         -o "$pw_dest" "https://api.github.com/repos/__REPO__/releases/assets/${pw_id}" 2>/dev/null
    pw_rc=$?
    pw_bytes=$(wc -c < "$pw_dest" 2>/dev/null)
    echo "pw_get asset=$pw_asset tag=$pw_tag try=$pw_try rc_curl=$pw_rc bytes=${pw_bytes:-0} cap=${pw_m}s budget_left=${pw_left}s" >&2
    if [ "$pw_rc" -eq 0 ] && [ -s "$pw_dest" ]; then
      return 0
    fi
    rm -f "$pw_dest"
    pw_try=$(( pw_try + 1 ))
  done
  return 1
}

# Content check on a downloaded asset. A mismatch discards the payload and degrades: a half-written
# wheelhouse installed with no index is a broken environment that would be blamed on the benchmark.
pw_verify(){
  pw_want=$(tr -d ' \n' < "$2" 2>/dev/null | cut -c1-64)
  pw_have=$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1)
  [ -n "$pw_want" ] && [ "$pw_want" = "$pw_have" ]
}

{
  echo "=== prewarm $(date -u) reqhash=$PW_REQHASH pins=$PW_LOCK_PINS wh_tag=$PW_WH_TAG kc_tag=$PW_KC_TAG publish=never(rfcbench) ==="
  echo "=== pip $(date -u) ==="
  python3 -m pip install -q --upgrade pip setuptools wheel
  rc_selfupgrade=$?; echo "rc_selfupgrade=$rc_selfupgrade"

  PW_T0=$(date -u +%s)
  if pw_get "$PW_WH_TAG" "$PW_WH_ASSET" /tmp/wheelhouse.tar &&
     pw_get "$PW_WH_TAG" "${PW_WH_ASSET}.sha256" /tmp/wheelhouse.tar.sha256; then
    T_DL=$(( $(date -u +%s) - PW_T0 ))
    PW_WH_BYTES=$(wc -c < /tmp/wheelhouse.tar)
    if pw_verify /tmp/wheelhouse.tar /tmp/wheelhouse.tar.sha256; then
      mkdir -p /tmp/wh
      tar -xf /tmp/wheelhouse.tar -C /tmp/wh
      if [ $? -eq 0 ]; then PW_WH=HIT; else PW_WH=FALLBACK:untar; rm -rf /tmp/wh; fi
    else
      PW_WH=FALLBACK:sha256
      rm -rf /tmp/wh /tmp/wheelhouse.tar
    fi
  else
    T_DL=$(( $(date -u +%s) - PW_T0 ))
    rm -f /tmp/wheelhouse.tar
  fi
  echo "rc_wh=$PW_WH bytes=$PW_WH_BYTES t_dl=${T_DL}s"

  PW_T0=$(date -u +%s)
  rc_lock=1
  if [ "$PW_WH" = "HIT" ]; then
    python3 -m pip install -q --no-index --find-links /tmp/wh -r /tmp/req.lock
    rc_lock=$?
    [ $rc_lock -ne 0 ] && PW_WH=FALLBACK:install
  fi
  if [ "$PW_WH" != "HIT" ]; then
    mkdir -p /tmp/wh
    python3 -m pip download -q -d /tmp/wh -r /tmp/req.lock
    rc_dl=$?
    python3 -m pip download -q -d /tmp/wh --only-binary=:all: setuptools wheel >/dev/null 2>&1
    if [ $rc_dl -eq 0 ]; then
      python3 -m pip install -q --no-index --find-links /tmp/wh -r /tmp/req.lock
      rc_lock=$?
    fi
    if [ $rc_lock -ne 0 ]; then
      case "$PW_WH" in FALLBACK:*) ;; *) PW_WH=FALLBACK:online ;; esac
      python3 -m pip install -q --no-cache-dir "numpy>=1.24" scipy pyscf
      rc_base=$?; echo "rc_base=$rc_base"
      python3 -m pip install -q --no-cache-dir "gpu4pyscf-cuda12x==1.8.1"
      rc_g4p=$?; echo "rc_g4p=$rc_g4p"
    fi
  fi
  T_INSTALL=$(( $(date -u +%s) - PW_T0 ))
  if [ $rc_lock -eq 0 ]; then
    rc_base=0
    rc_g4p=0
    echo "rc_base=$rc_base"
    echo "rc_g4p=$rc_g4p"
  fi
  echo "rc_lock=$rc_lock wh=$PW_WH t_apt=${T_APT}s t_install=${T_INSTALL}s"

  # The cupy kernel cache, seeded BEFORE anything imports cupy. Safe by cupy's own construction: the
  # cache filename is a sha1 over arch + options + NVRTC version + backend + source + header
  # checksum and the payload is SHA1-checked on load. This pod only ever READS this asset.
  PW_T0=$(date -u +%s)
  if [ "$PW_ARCH" = "unknown" ]; then
    PW_KC=FALLBACK:noarch
  elif pw_get "$PW_KC_TAG" "$PW_KC_ASSET" /tmp/kc.tar.gz &&
     pw_get "$PW_KC_TAG" "${PW_KC_ASSET}.sha256" /tmp/kc.tar.gz.sha256; then
    PW_KC_BYTES=$(wc -c < /tmp/kc.tar.gz)
    if pw_verify /tmp/kc.tar.gz /tmp/kc.tar.gz.sha256; then
      mkdir -p "$PW_CACHE_DIR"
      tar -xzf /tmp/kc.tar.gz -C "$PW_CACHE_DIR"
      if [ $? -eq 0 ]; then
        PW_KC_FILES=$(ls -1 "$PW_CACHE_DIR" 2>/dev/null | wc -l)
        if [ "$PW_KC_FILES" -gt 0 ]; then
          PW_KC=HIT
        else
          PW_KC=FALLBACK:empty
        fi
      else
        rm -rf "$PW_CACHE_DIR"
        PW_KC=FALLBACK:untar
      fi
    else
      PW_KC=FALLBACK:sha256
      rm -f /tmp/kc.tar.gz
    fi
  else
    rm -f /tmp/kc.tar.gz
  fi
  T_KC=$(( $(date -u +%s) - PW_T0 ))
  echo "rc_kc=$PW_KC files=$PW_KC_FILES bytes=$PW_KC_BYTES t_kc=${T_KC}s"
  echo "=== resolved versions ==="
  python3 -c "
import importlib.metadata as _im
CANDIDATES = (('gpu4pyscf', ('gpu4pyscf-cuda12x', 'gpu4pyscf-cuda11x', 'gpu4pyscf')),
              ('pyscf', ('pyscf',)),
              ('cupy', ('cupy-cuda12x', 'cupy-cuda11x', 'cupy')),
              ('numpy', ('numpy',)),
              ('scipy', ('scipy',)))
for name, dists in CANDIDATES:
    for dist in dists:
        try:
            print(f'{name}: {dist}=={_im.version(dist)}')
            break
        except Exception:
            continue
    else:
        print(f'{name}=unknown (none of {dists} is installed)')
"
} > /tmp/install.log 2>&1

PREWARM_LINE="WH=$PW_WH KC=$PW_KC arch=sm_$PW_ARCH reqhash=$PW_REQHASH pins=$PW_LOCK_PINS t_apt=${T_APT}s t_dl=${T_DL}s t_install=${T_INSTALL}s t_kc=${T_KC}s wh_bytes=$PW_WH_BYTES kc_bytes=$PW_KC_BYTES kc_files=$PW_KC_FILES publish=never(rfcbench) keys=$PW_WH_TAG,$PW_KC_TAG"
export RFCBENCH_PREWARM="$PREWARM_LINE"
# UNCONDITIONAL FLUSH, which is also what CREATES out.txt: a run killed by the cap between here and
# the end-of-run trailer would otherwise report nothing about the download that decided it.
{ echo "=== prewarm (flushed before the clone: a cap kill after this point cannot take it): $PREWARM_LINE ==="; } > /tmp/out.txt 2>&1
"""


if __name__ == "__main__":
    main()
