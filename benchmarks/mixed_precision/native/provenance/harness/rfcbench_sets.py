"""Closed sets, cell lists and work estimates of the `rfcbench` campaign. STDLIB ONLY.

Shared by the free runner (`runpod_rfcbench.py`, a bare hosted runner with no numpy) and the pod
(`rfcbench_pod.py`, `rfcbench_cells.py`), so the two cannot drift apart: one source of truth instead
of duplicate-and-pin-equal.

Binding spec: docs/upstream/PREREG-rfcbench-0-measurand.md (measurand, gates, rules) and the scoped
documents -1 .. -5. Every set below is closed; anything else is refused on the free runner.
"""
from __future__ import annotations

import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# --------------------------------------------------------------------------------------------- #
# the measured tree
# --------------------------------------------------------------------------------------------- #
# PREREG-0: only a fork whose gpu4pyscf/ tree is byte-identical to this commit's may be measured.
VALIDATED_FORK_SHA = "63af0568d4fd19935bef51b7fd71f161a9cee56f"
FORK_REPO_DEFAULT = "chris-lee-mc/gpu4pyscf"
BASE_TAG = "v1.8.1"
BASE_SHA = "5b284c258a4260baef80e3d150b4e7a81a9dbd57"     # gpu4pyscf v1.8.1 == the installed wheel
CUTENSOR_PIN = "cutensor-cu12==2.3.1"                       # as runpod_scfbench.py

# --------------------------------------------------------------------------------------------- #
# modes, arms, functionals, bases
# --------------------------------------------------------------------------------------------- #
MODES = ("commission", "ladder", "cross", "large", "basis", "downstream", "mig", "migfit",
         "gputypes")
# PREREG-0 section 2: a CACHE_FALLBACK / NOT_TREATED cell is a FAIL here ...
FALLBACK_FAIL_MODES = ("commission", "ladder", "cross", "downstream")
# ... and a pre-registered outcome (kept, listed in the fit table, out of the treated aggregates) here.
FALLBACK_OUTCOME_MODES = ("large", "basis", "mig", "migfit")

# Order within every pair: mixed, then cache (if present), then stock. Stock LAST, so the warm pool
# favours the baseline and the bias runs against the claim (defect family 4). `arms_for` returns a
# sub-sequence of this tuple and nothing else.
ARM_ORDER = ("mixed", "cache", "stock")
CACHE_ARM_MODES = ("commission", "cross")

XCS = ("r2scan", "b3lyp", "wb97m-v")
VV10_XC = "wb97m-v"

# Basis key -> (pyscf basis, default auxbasis). The basis scope uses def2-universal-jkfit for EVERY
# basis (PREREG-3), the other scopes def2-tzvpp-jkfit at def2-mTZVPP (PREREG-0 section 1).
BASES = {"mtzvpp": "def2-mtzvpp", "svp": "def2-svp", "tzvp": "def2-tzvp"}
AUX_DEFAULT = "def2-tzvpp-jkfit"
AUX_UNIVERSAL = "def2-universal-jkfit"

# Campaign settings (PREREG-0 section 1), read back off the built objects on the pod.
GRID_LEVEL = 3
CONV_TOL = 1e-9
MAX_CYCLE = 60
BRIDGE_CONV_TOL_GRAD = 1e-5        # PREREG-1 bridge leg only
# PREREG-4 reference run R.
REF_CONV_TOL = 1e-11
REF_CONV_TOL_GRAD = 3e-6
REF_MAX_CYCLE = 100

REPEATS_DEFAULT = 3
REPEATS_MIN = 2
REPEATS_MAX = 5

# --------------------------------------------------------------------------------------------- #
# molecules
# --------------------------------------------------------------------------------------------- #
TRIO = ("paracetamol", "propranolol", "celecoxib")
DOWNSTREAM_MOLS = TRIO + ("fluconazole", "warfarin", "omeprazole")
BASIS_MOLS = TRIO + ("sildenafil",)
TIERS = ("S", "M", "L")

DRUGSET_PATH = os.path.join(REPO_ROOT, "src", "jqc_conformer", "drugset.py")
GEOM_DIR = os.path.join(REPO_ROOT, "src", "scf_headtohead", "geometries")
LARGE_DIR = os.path.join(REPO_ROOT, "src", "scf_headtohead", "geometries_rfc_large")

# PREREG-3 LARGE set: (name, n_atoms). Pinned against freeze_rfc_large.LARGE by a host test.
# ritonavir is C37H48N6O5S2, 98 atoms (the committed file and freeze_rfc_large agree; PREREG-3's
# table row "C37H48N6O5S, 97" is a typo for the same molecule -- see the hand-off report).
LARGE = (("sildenafil", 63, 33), ("imatinib", 68, 37), ("atorvastatin", 76, 41),
         ("montelukast", 77, 41), ("lopinavir", 94, 46), ("ritonavir", 98, 50))   # (name, atoms, heavy)
LARGE_NAMES = tuple(r[0] for r in LARGE)
LARGE_WB97MV = ("sildenafil", "atorvastatin")
# The committed SHA256SUMS of geometries_rfc_large/ (commit 17be2f31), checked on the pod BEFORE any
# SCF and by a host test against the files in the tree.
LARGE_SHA256 = {
    "sildenafil": "13e1f2a31ab3ed2a19b28407f378995c20bbd7dfa98391e277c16a2d07e92160",
    "imatinib": "8954647d0f2a6b10c622519b326ee7a714be8cae0413bd4d88116027483155ac",
    "atorvastatin": "3de1abd52b70e63f904dc9990b0e6db2caa0a7d0e864f96a2ea9a039750e2368",
    "montelukast": "a025eda852d38ef3de8aca7af22c3e814a54535470679524d038f5a701d8d7b8",
    "lopinavir": "1ab60a8be8432b0dbc091c713e515213fc143ebcde68c7f6c072c3c9753467d5",
    "ritonavir": "7bf7b4de222d2901f88e37c3f5d548c0a45b0773fb358bf12bd2956d1450aa6f",
}


def drug_table() -> dict:
    """`{name: (n_atoms, tier, n_heavy)}` for the 24-drug ladder, PARSED out of drugset.py (never imported:
    the free runner has no numpy). Fails closed on a short parse."""
    try:
        with open(DRUGSET_PATH) as fh:
            text = fh.read()
    except OSError as e:
        raise SystemExit(f"::error::cannot read the drug table at {DRUGSET_PATH}: {e}")
    rows = re.findall(r'Drug\(\s*"([A-Za-z0-9_-]+)",\s*"[^"]*",\s*[0-9.]+,\s*(\d+),\s*(\d+),\s*'
                      r'"([SML])"\s*\)', text)
    if len(rows) != 24:
        raise SystemExit(f"::error::parsed {len(rows)} rows out of {DRUGSET_PATH}, not the 24 of "
                         f"the ladder; refusing to validate molecule names against it")
    return {name: (int(n_atoms), tier, int(heavy)) for name, heavy, n_atoms, tier in rows}


def atom_counts(mol: str) -> tuple:
    """(n_atoms, n_heavy) of a ladder or LARGE molecule."""
    for name, n, heavy in LARGE:
        if name == mol:
            return n, heavy
    n, _tier, heavy = drug_table()[mol]
    return n, heavy


def n_atoms(mol: str) -> int:
    return atom_counts(mol)[0]


def size_measure(mol: str) -> float:
    """heavy atoms + H/4: the work-estimate size measure (a hydrogen carries few basis functions)."""
    n, heavy = atom_counts(mol)
    return heavy + 0.25 * (n - heavy)


def geometry_file(mol: str) -> str:
    return (os.path.join(LARGE_DIR, f"{mol}.xyz") if mol in LARGE_NAMES
            else os.path.join(GEOM_DIR, f"{mol}.xyz"))


# --------------------------------------------------------------------------------------------- #
# GPU classes
# --------------------------------------------------------------------------------------------- #
# Each class is a SINGLE-model cascade with no fallback: a landing on any other card is no row.
# `pro6000mig` and `pro6000se` are pinned by PREREG-rfcbench-5 Amendment 1 from the mode=gputypes
# listing of run 36951667656: RunPod's only quarter-card PRO 6000 MIG type is on the SERVER Edition,
# so the whole-card comparator for the mig scope is a Server Edition card (`pro6000se`), not the
# Workstation Edition of `pro6000`.
GPU_CASCADES = {
    "pro6000": ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"],
    "5090": ["NVIDIA GeForce RTX 5090"],
    "l40s": ["NVIDIA L40S"],
    "h100": ["NVIDIA H100 80GB HBM3"],
    "a100": ["NVIDIA A100-SXM4-80GB"],
    "pro6000se": ["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
    "pro6000mig": ["NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"],
}
# What the pod's card gate requires `nvidia-smi --query-gpu=name` to print, EXACTLY, per class.
# pro6000mig is gated differently (rfcbench_pod.card_problems): `nvidia-smi --query-gpu=name` prints
# exactly MIG_PARENT_MODEL, `nvidia-smi -L` shows one GPU of that model and exactly one MIG device,
# and the CUDA total memory is in MIG_MEM_GB.
CARD_MODEL = {k: (v[0] if v else None) for k, v in GPU_CASCADES.items()}
MIG_CLASS = "pro6000mig"
MIG_PARENT_MODEL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
CARD_MODEL[MIG_CLASS] = MIG_PARENT_MODEL
MIG_MEM_GB = (20.0, 28.0)
CROSS_CLASSES = ("5090", "l40s", "h100", "a100")

# Which classes each mode may run on. gputypes launches no pod, so it takes no class.
MODE_CLASSES = {
    "commission": ("pro6000",), "ladder": ("pro6000",), "large": ("pro6000",),
    "basis": ("pro6000",), "downstream": ("pro6000",), "cross": CROSS_CLASSES,
    "mig": (MIG_CLASS, "pro6000se"), "migfit": (MIG_CLASS, "pro6000se"), "gputypes": (),
}

# PREREG-rfcbench-6: where one 1g.24gb instance stops holding the treated path. Three LARGE molecules
# that bracket the predicted limits, r2SCAN and B3LYP only (wB97M-V at this size cannot finish R=3
# inside one instance pod's work cap). Mixed and stock arms; the fit is an outcome, not a failure.
MIGFIT_MOLS = ("sildenafil", "imatinib", "atorvastatin")
MIGFIT_XCS = ("r2scan", "b3lyp")

# --------------------------------------------------------------------------------------------- #
# cells
# --------------------------------------------------------------------------------------------- #
# A cell is a dict: {"kind": speed|bridge|downstream, "mol", "xc", "basis", "aux", "arms"}.
# Its key is "kind/mol/xc/basis" and must be unique in a dispatch.

def arms_for(mode: str) -> tuple:
    if mode in CACHE_ARM_MODES:
        return ARM_ORDER
    return tuple(a for a in ARM_ORDER if a != "cache")


def cell_key(c: dict) -> str:
    return f"{c['kind']}/{c['mol']}/{c['xc']}/{c['basis']}"


def _cell(kind, mol, xc, basis="mtzvpp", aux=AUX_DEFAULT, arms=("mixed", "stock")):
    return {"kind": kind, "mol": mol, "xc": xc, "basis": basis, "aux": aux, "arms": list(arms)}


def ladder_mols() -> tuple:
    t = drug_table()
    return tuple(t)       # table order


def mode_cells(mode: str) -> list:
    """The mode's FULL pre-registered cell list, before the dispatch's filters."""
    if mode == "commission":
        cells = [_cell("speed", m, x, arms=ARM_ORDER) for m in TRIO for x in XCS]
        cells.append(_cell("downstream", "paracetamol", "r2scan", arms=("mixed", "stock")))
        cells.append(_cell("speed", "paracetamol", "r2scan", basis="svp", aux=AUX_UNIVERSAL))
        return cells
    if mode == "cross":
        return [_cell("speed", m, x, arms=ARM_ORDER) for m in TRIO for x in XCS]
    if mode == "ladder":
        return [_cell("speed", m, x) for x in XCS for m in ladder_mols()]
    if mode == "large":
        # PREREG-3: r2SCAN and B3LYP on all six (XL1, XL2); wB97M-V on two only (XL3).
        return ([_cell("speed", m, x) for x in ("r2scan", "b3lyp") for m in LARGE_NAMES]
                + [_cell("speed", m, VV10_XC) for m in LARGE_WB97MV])
    if mode == "basis":
        return [_cell("speed", m, x, basis=b, aux=AUX_UNIVERSAL)
                for x in ("r2scan", "b3lyp") for b in ("svp", "mtzvpp", "tzvp") for m in BASIS_MOLS]
    if mode == "downstream":
        return [_cell("downstream", m, x) for x in XCS for m in DOWNSTREAM_MOLS]
    if mode == "mig":
        return ([_cell("speed", m, x) for x in ("r2scan", "b3lyp") for m in TRIO]
                + [_cell("speed", m, VV10_XC) for m in ("paracetamol", "propranolol")])
    if mode == "migfit":
        return [_cell("speed", m, x) for x in MIGFIT_XCS for m in MIGFIT_MOLS]
    return []


def admissible(mode: str) -> dict:
    """What the dispatch's `mols` / `xcs` / `bases` filters may name in this mode."""
    cells = mode_cells(mode)
    mols = sorted({c["mol"] for c in cells})
    if mode == "ladder":
        mols = mols + list(TIERS)            # tier tokens expand to the tier's drugs
    return {"mols": mols, "xcs": sorted({c["xc"] for c in cells}),
            "bases": sorted({c["basis"] for c in cells})}


# Modes whose cell set is fixed by its PREREG: no filter may be given, with one exception per
# FIXED_MODE_RERUNS: PREREG-rfcbench-0 Amendment 3's C0w re-run of C0's wB97M-V trio cells (all
# three arms) after C0 was found CONTENTION-UNKNOWN.
FIXED_MODES = ("commission",)
FIXED_MODE_RERUNS = {("commission", "RFCBENCH_XCS"): ("wb97m-v",)}
# Per-mode default filters (blank input). None = no filter, i.e. every cell of the mode.
DEFAULT_XCS = {"ladder": ("r2scan", "b3lyp"), "large": ("r2scan",), "downstream": ("r2scan", "b3lyp")}


def expand_mols(mode: str, names: list) -> list:
    if mode != "ladder":
        return list(names)
    table = drug_table()
    out = []
    for n in names:
        if n in TIERS:
            out += [m for m, row in table.items() if row[1] == n]
        else:
            out.append(n)
    return out


def select_cells(mode: str, mols=None, xcs=None, bases=None, bridge=False) -> list:
    """Filter the mode's cell list. Order is the mode's own (grouped by xc on the pod)."""
    cells = mode_cells(mode)
    out = [c for c in cells
           if (mols is None or c["mol"] in mols) and (xcs is None or c["xc"] in xcs)
           and (bases is None or c["basis"] in bases)]
    if bridge:
        out += [dict(c, kind="bridge") for c in out if c["kind"] == "speed" and c["xc"] == "r2scan"]
    return out


def groups(cells: list) -> list:
    """Cells grouped into subprocesses: one group per (kind, xc), in first-appearance order, with the
    bridge group last. Each group runs in its own process under its own timeout."""
    order, by = [], {}
    for c in cells:
        g = (c["kind"], c["xc"])
        if g not in by:
            by[g] = []
            order.append(g)
        by[g].append(c)
    order.sort(key=lambda g: g[0] == "bridge")
    return [{"kind": k, "xc": x, "cells": by[(k, x)]} for k, x in order]


# --------------------------------------------------------------------------------------------- #
# work estimates (seconds) -- an OVER-estimate, never an average (defect family 5)
# --------------------------------------------------------------------------------------------- #
# Stock warm wall on the RTX PRO 6000 at def2-mTZVPP / tzvpp-jkfit, anchored on paracetamol and
# celecoxib of run 36908911324 (g4pport VV10 validation, stock arm, einsum backend): r2SCAN
# 4.71 / 15.29 s, B3LYP 2.02 / 12.67 s, wB97M-V 20.81 / 97.60 s, as a per-functional power law in
# `size_measure` (13.25 and 29.5). Through the pad it covers propranolol's measured 10.9 / 6.4 /
# 73.2 s (estimates 13.2 / 9.3 / 76.9 s). Beyond celecoxib (the LARGE set) it is an extrapolation.
ANCHOR_SIZE = (13.25, 29.5)
ANCHOR_WALL_S = {"r2scan": (4.71, 15.29), "b3lyp": (2.02, 12.67), "wb97m-v": (20.81, 97.60)}
# Pod-to-pod drift of the same stock wall (4.71 vs 4.35 s, PORT-PLAN-ao-cache RESULT), and the
# power law's 9 % under-estimate of propranolol wB97M-V.
STOCK_PAD = 1.15
# Basis size relative to def2-mTZVPP (not banked; over-demanding guesses).
BASIS_FACTOR = {"mtzvpp": 1.0, "svp": 0.4, "tzvp": 1.5}
# Stock wall relative to the PRO 6000, padded toward over-demand from what is banked
# (.claude/skills/scfbench-benchmarking): 5090 -- the paracetamol r2SCAN canary 4.71 s sits in the PRO
# 6000 band (run 32027743165), padded 10 %; H100 -- "stock trio ~2.4x faster than Blackwell"
# (32075407566), i.e. 0.42, padded to 0.6; A100 -- celecoxib stock warm 6.93 s (32372635923) against
# ~13-15 s on the PRO 6000, i.e. ~0.5, padded to 0.75; L40S -- nothing banked, 2.0 from its lower FP64
# rate and half the bandwidth. An MIG instance is budgeted 4x slower than the whole card (PREREG-5).
CLASS_FACTOR = {"pro6000": 1.0, "5090": 1.1, "l40s": 2.0, "h100": 0.6, "a100": 0.75,
                "pro6000se": 1.0, "pro6000mig": 4.0}
# The mixed arm budgeted at stock / this. On the 1:64-FP64 cards every value is at least 15 % below
# the PREREG-1/2 prediction for that functional (r2SCAN 1.60-1.80, B3LYP 1.80-2.05, wB97M-V 2.40-2.65)
# and below the banked trio ratios (1.81 / 1.97 / 2.69, run 36942908651). On H100/A100 the treated
# arm can be SLOWER than stock (the prototype's A100 celecoxib e2e ratio was 0.748, run 32372635923),
# so it is budgeted at stock / 0.75. The cache arm is always budgeted at the full stock wall.
FAST_CLASSES = ("pro6000", "5090", "l40s", "pro6000se", "pro6000mig")
MIXED_BUDGET_SPEEDUP_FAST = {"r2scan": 1.5, "b3lyp": 1.5, "wb97m-v": 2.0}
MIXED_BUDGET_SPEEDUP_FP64_STRONG = 0.75


def mixed_budget_speedup(gpu: str, xc: str) -> float:
    return MIXED_BUDGET_SPEEDUP_FAST[xc] if gpu in FAST_CLASSES else MIXED_BUDGET_SPEEDUP_FP64_STRONG
# Downstream (PREREG-4): R at 1e-11 budgeted 1.6x a stock SCF, S' (atom guess) 1.2x; each of the four
# analytic gradients at one stock SCF wall.
DS_REF_FACTOR = 1.6
DS_ATOM_GUESS_FACTOR = 1.2
DS_GRAD_FACTOR = 1.0
# Per subprocess: imports, first-touch NVRTC JIT, cold DF build (102.6 s measured on a KC=MISS pod).
GROUP_COLD_S = 150.0
# Once per pod: apt/pip install and clone (~300 s on a wheelhouse HIT), cuTENSOR pip, fork fetch,
# base-identity + overlay, card gate and the environment probe.
POD_FIXED_S = 420.0


def stock_wall_s(mol: str, xc: str, basis: str = "mtzvpp", gpu: str = "pro6000") -> float:
    (a0, a1), (w0, w1) = ANCHOR_SIZE, ANCHOR_WALL_S[xc]
    p = math.log(w1 / w0) / math.log(a1 / a0)
    return (w0 * (size_measure(mol) / a0) ** p * STOCK_PAD * BASIS_FACTOR[basis]
            * CLASS_FACTOR[gpu])


# --------------------------------------------------------------------------------------------- #
# calibration (optional): measured walls replace the guessed factors of a class
# --------------------------------------------------------------------------------------------- #
# `.github/scripts/rfcbench_calibration.json`, absent by default, written by rfcbench_calibrate.py
# from a banked PASS sentinel (C0 for pro6000). Schema:
#   {"<gpu_class>": {"cells": {"<mol>|<xc>|<basis>|<arm>": seconds, ...},
#                    "powerlaw": {"<xc>|<arm>": {"a": a, "p": p}, ...},
#                    "source": {...}}}
# Both already carry CALIBRATION_MARGIN: a cell is its measured warm median x 1.15, and the power
# law w = a * size_measure**p is scaled so it is >= 1.15 x every measured trio median. For a class
# present here the measured cell wins, then the power law (x BASIS_FACTOR off def2-mTZVPP); the
# guessed STOCK_PAD, CLASS_FACTOR and mixed speed-ups are not used. A malformed file is an error,
# never a silent fall back to the guesses.
CALIBRATION_PATH = os.path.join(HERE, "rfcbench_calibration.json")
CALIBRATION_MARGIN = 1.15
CAL_ARMS = ("mixed", "cache", "stock")
_CAL_CACHE = {}


def _pos(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def validate_calibration(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("calibration: top level is not an object")
    for gpu, cal in data.items():
        if gpu not in GPU_CASCADES or not isinstance(cal, dict):
            raise ValueError(f"calibration: {gpu!r} is not a GPU class with an object")
        cells, power = cal.get("cells"), cal.get("powerlaw")
        if not isinstance(cells, dict) or not isinstance(power, dict):
            raise ValueError(f"calibration: {gpu} lacks 'cells' / 'powerlaw' objects")
        for k, v in cells.items():
            parts = k.split("|")
            if (len(parts) != 4 or parts[1] not in XCS or parts[2] not in BASES
                    or parts[3] not in CAL_ARMS or not _pos(v)):
                raise ValueError(f"calibration: {gpu} cell {k!r}={v!r} is malformed")
        for k, v in power.items():
            parts = k.split("|")
            if (len(parts) != 2 or parts[0] not in XCS or parts[1] not in CAL_ARMS
                    or not isinstance(v, dict) or not _pos(v.get("a"))
                    or not isinstance(v.get("p"), (int, float)) or isinstance(v.get("p"), bool)
                    or not math.isfinite(v["p"])):
                raise ValueError(f"calibration: {gpu} power law {k!r}={v!r} is malformed")
    return data


def calibration() -> dict:
    path = CALIBRATION_PATH
    if not os.path.isfile(path):
        return {}
    key = (path, os.path.getmtime(path))
    if key not in _CAL_CACHE:
        with open(path) as f:
            _CAL_CACHE[key] = validate_calibration(json.load(f))
    return _CAL_CACHE[key]


def arm_wall_s(mol: str, xc: str, basis: str, gpu: str, arm: str) -> float:
    """One run's budgeted wall: calibrated when the class is calibrated, else the guessed model."""
    cal = calibration().get(gpu)
    if cal is not None:
        hit = cal["cells"].get(f"{mol}|{xc}|{basis}|{arm}")
        if hit is not None:
            return float(hit)
        law = cal["powerlaw"].get(f"{xc}|{arm}") or (
            cal["powerlaw"].get(f"{xc}|stock") if arm == "cache" else None)
        if law is None:
            raise ValueError(f"calibration: class {gpu} has no power law for {xc}|{arm}")
        return law["a"] * size_measure(mol) ** law["p"] * BASIS_FACTOR[basis]
    w = stock_wall_s(mol, xc, basis, gpu)
    return w / mixed_budget_speedup(gpu, xc) if arm == "mixed" else w


def cell_estimate_s(cell: dict, repeats: int, gpu: str) -> float:
    w = arm_wall_s(cell["mol"], cell["xc"], cell["basis"], gpu, "stock")
    mixed = arm_wall_s(cell["mol"], cell["xc"], cell["basis"], gpu, "mixed")
    cache = (arm_wall_s(cell["mol"], cell["xc"], cell["basis"], gpu, "cache")
             if "cache" in cell["arms"] else w)
    if cell["kind"] == "downstream":
        return (w * DS_REF_FACTOR + w + w * DS_ATOM_GUESS_FACTOR + mixed
                + 4 * w * DS_GRAD_FACTOR)
    per_pair = sum({"mixed": mixed, "cache": cache, "stock": w}[a] for a in cell["arms"])
    pairs = 1 if cell["kind"] == "bridge" else 1 + int(repeats)
    return pairs * per_pair


def group_estimate_s(group: dict, repeats: int, gpu: str) -> float:
    return GROUP_COLD_S + sum(cell_estimate_s(c, repeats, gpu) for c in group["cells"])


def work_estimate_s(cells: list, repeats: int, gpu: str) -> float:
    return POD_FIXED_S + sum(group_estimate_s(g, repeats, gpu) for g in groups(cells))


# --------------------------------------------------------------------------------------------- #
# the dispatch plan after C0 (PREREG-rfcbench-0 Amendment 2)
# --------------------------------------------------------------------------------------------- #
# Every remaining pod as (name, mode, gpu, mols, xcs, bridge). Under the committed C0 calibration each
# is admitted by the free runner, and per scope the pods' cells partition the pre-registered cells
# exactly (tests/test_rfcbench.py pins both). M2a/M2b repeat M1a/M1b on another pod.
_W_S = ("metformin", "paracetamol", "nicotine", "gabapentin", "aspirin", "theophylline",
        "caffeine", "levodopa")
POD_PLAN = (
    ("L12", "ladder", "pro6000", (), ("r2scan", "b3lyp"), True),
    ("W1", "ladder", "pro6000", _W_S + ("celecoxib",), ("wb97m-v",), False),
    ("W2", "ladder", "pro6000", ("fluconazole", "warfarin", "omeprazole"), ("wb97m-v",), False),
    ("W3", "ladder", "pro6000", ("ibuprofen", "naproxen", "lidocaine", "salbutamol", "phenytoin"),
     ("wb97m-v",), False),
    ("W4", "ladder", "pro6000", ("ketoprofen", "diphenhydramine", "propranolol", "atenolol"),
     ("wb97m-v",), False),
    ("W5", "ladder", "pro6000", ("metoprolol", "trimethoprim", "diclofenac"), ("wb97m-v",), False),
    ("D1", "downstream", "pro6000", (), ("r2scan", "b3lyp"), False),
    ("D2a", "downstream", "pro6000", ("paracetamol", "propranolol", "celecoxib"), ("wb97m-v",),
     False),
    ("D2b", "downstream", "pro6000", ("fluconazole", "warfarin"), ("wb97m-v",), False),
    ("D2c", "downstream", "pro6000", ("omeprazole",), ("wb97m-v",), False),
    ("XL1", "large", "pro6000", (), ("r2scan",), False),
    ("XL2", "large", "pro6000", (), ("b3lyp",), False),
    ("XL3a", "large", "pro6000", ("sildenafil",), ("wb97m-v",), False),
    ("XL3b", "large", "pro6000", ("atorvastatin",), ("wb97m-v",), False),
    ("B1", "basis", "pro6000", (), (), False),
    ("X1", "cross", "5090", (), (), False),
    ("X2a", "cross", "l40s", ("paracetamol", "propranolol"), (), False),
    ("X2b", "cross", "l40s", ("celecoxib",), (), False),
    ("X3", "cross", "h100", (), (), False),
    ("X4", "cross", "a100", (), (), False),
    ("M0", "mig", "pro6000se", (), (), False),
    ("M1a", "mig", "pro6000mig", (), ("r2scan", "b3lyp"), False),
    ("M1b", "mig", "pro6000mig", (), ("wb97m-v",), False),
    ("M2a", "mig", "pro6000mig", (), ("r2scan", "b3lyp"), False),
    ("M2b", "mig", "pro6000mig", (), ("wb97m-v",), False),
)

# PREREG-rfcbench-6, kept apart from POD_PLAN (the 25-pod campaign it closes): F0 is the whole-card
# comparator on a Server Edition card; F1a / F1b split the instance by functional so that each fits
# the work cap at R=3.
MIGFIT_POD_PLAN = (
    ("F0", "migfit", "pro6000se", (), (), False),
    ("F1a", "migfit", "pro6000mig", (), ("r2scan",), False),
    ("F1b", "migfit", "pro6000mig", (), ("b3lyp",), False),
)
