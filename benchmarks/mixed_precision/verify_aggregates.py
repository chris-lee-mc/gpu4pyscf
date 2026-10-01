#!/usr/bin/env python3
"""Re-derive every banked aggregate from the per-molecule CSVs in data/ and check it.

Stdlib only. Each check recomputes a value from the CSV rows and compares it with the
number as printed in the source doc, at that number's printed precision (decimal places
for fixed notation, significant figures for scientific notation). Bounds printed as
"<= X" in the docs are checked as bounds. Exit status is non-zero on any mismatch, and
also if a CSV is missing or a check finds no rows (fail closed: an empty selection is
never a pass).

The CSVs were extracted from SCFBENCH_JSON / ABA_JSON_* / GRADMP_JSON / VV10COMM_JSON /
VV10DF64COMM_JSON lines of the sentinels listed in data/SOURCES.md.
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

results = []  # (source, check, recomputed_str, banked_str, status)

# Banked values that genuinely disagree with the sentinel at their printed precision.
# They are NOT adjusted to pass: each is printed as DISCREPANCY with its reason, and the
# script fails if one of them ever starts matching (so the list cannot go stale silently).
KNOWN_DISCREPANCIES = {
    ("5 energy trio", "a100 celecoxib df_build delta spec-g4p"):
        "SPEC-v2 7.3.3 subtracted the 4-dp rounded phases (2.7379 - 0.3996); "
        "the unrounded sentinel values give 2.33825 -> 2.3382",
}


def load(name):
    path = os.path.join(DATA, name)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(x):
    return float(x)


def geomean(xs):
    xs = list(xs)
    if not xs:
        raise ValueError("geomean of empty selection")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def fmt_like(value, banked):
    """Format value at the printed precision of the banked string."""
    b = banked.strip()
    if "e" in b.lower():
        mant = b.lower().split("e")[0]
        digits = len(mant.replace("-", "").replace(".", "").lstrip("0")) or 1
        return "%.*e" % (digits - 1, value)
    dec = len(b.split(".")[1]) if "." in b else 0
    return "%.*f" % (dec, value)


def same_float_str(a, b):
    return float(a) == float(b)


def record(source, name, got, want, ok):
    reason = KNOWN_DISCREPANCIES.get((source, name))
    if reason is None:
        status = "ok" if ok else "FAIL"
    else:
        # a listed discrepancy that now matches is itself a failure: the list is stale
        status = "FAIL(stale-known)" if ok else "DISCREPANCY"
        want = "%s  [%s]" % (want, reason)
    results.append((source, name, got, want, status))


def check_eq(source, name, value, banked):
    got = fmt_like(value, banked)
    record(source, name, got, banked, same_float_str(got, banked))


def check_le(source, name, value, bound):
    """A bound printed as '<= X' is met if the value, at X's printed precision, is <= X."""
    got = fmt_like(value, bound)
    record(source, name, "%.3e (%s)" % (value, got), "<= " + bound, float(got) <= float(bound))


def check_exact(source, name, got, want):
    record(source, name, str(got), str(want), got == want)


def sel(rows, **kw):
    out = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    if not out:
        raise ValueError("empty selection %r" % (kw,))
    return out


def by_mol(rows):
    return {r["molecule"]: r for r in rows}


# ---------------------------------------------------------------- 1. r2SCAN ladder
def ladder():
    src = "1 r2SCAN ladder"
    rows = load("r2scan_ladder_pro6000.csv")
    tier_rows = [r for r in rows if r["role"] == "tier"]
    banked = {
        "S": ("35550452536", 8, "1.8437", "1.744", "1.985"),
        "M": ("35550452536", 12, "1.7827", "1.666", "1.992"),
        "L": ("35551843564", 4, "1.8100", "1.702", "1.869"),
    }
    for tier, (run, n, gm, lo, hi) in banked.items():
        rs = sel(tier_rows, tier=tier, run_id=run)
        check_exact(src, "%s coverage" % tier, len(rs), n)
        ratios = [num(r["ratio_g4p_over_spec"]) for r in rs]
        check_eq(src, "%s geomean g4p/spec warm" % tier, geomean(ratios), gm)
        check_eq(src, "%s range min" % tier, min(ratios), lo)
        check_eq(src, "%s range max" % tier, max(ratios), hi)
    Lm = by_mol(sel(tier_rows, tier="L"))
    for mol, g, s, r in [("fluconazole", "11.010", "6.049", "1.820"),
                         ("warfarin", "14.346", "7.678", "1.869"),
                         ("omeprazole", "16.566", "8.932", "1.855"),
                         ("celecoxib", "15.604", "9.171", "1.702")]:
        check_eq(src, "L %s g4p warm s" % mol, num(Lm[mol]["g4p_wall_warm_s"]), g)
        check_eq(src, "L %s spec warm s" % mol, num(Lm[mol]["spec_wall_warm_s"]), s)
        check_eq(src, "L %s ratio" % mol, num(Lm[mol]["ratio_g4p_over_spec"]), r)
    allr = [num(r["ratio_g4p_over_spec"]) for r in tier_rows]
    check_exact(src, "ladder coverage (all tiers)", len(tier_rows), 24)
    check_eq(src, "lowest single ratio of 24", min(allr), "1.666")
    lowest = min(tier_rows, key=lambda r: num(r["ratio_g4p_over_spec"]))["molecule"]
    check_exact(src, "lowest-ratio molecule", lowest, "trimethoprim")
    for disp in ("A", "B"):
        rs = sel(rows, dispatch=disp)
        check_eq(src, "dispatch %s worst |dE| Ha" % disp,
                 max(num(r["abs_dE_g4p_vs_spec_Ha"]) for r in rs), "9.1e-13")
        drift = sum(1 for r in rs if r["n_cycle_g4p"] != r["n_cycle_spec"])
        check_exact(src, "dispatch %s cycle drift count" % disp, drift, 0)
    check_eq(src, "canary paracetamol g4p warm A",
             num(sel(rows, dispatch="A", molecule="paracetamol")[0]["g4p_wall_warm_s"]), "4.391")
    check_eq(src, "canary paracetamol g4p warm B",
             num(sel(rows, dispatch="B", molecule="paracetamol")[0]["g4p_wall_warm_s"]), "4.392")


# ---------------------------------------------------------------- 2. B3LYP cg re-bank
def cg():
    src = "2 B3LYP cg re-bank"
    rows = load("b3lyp_cg_rebank_pro6000.csv")
    banked = {"S": ("36134049978", 8, "1.868", "1.109"),
              "M": ("36134049978", 12, "1.845", "1.229"),
              "L": ("36152138663", 4, "1.824", "1.315")}
    for tier, (run, n, vs, rk) in banked.items():
        rs = sel(rows, tier=tier, run_id=run)
        check_exact(src, "%s coverage" % tier, len(rs), n)
        check_eq(src, "%s geomean vs stock" % tier, geomean(num(r["vs_stock_g4p_over_xck"]) for r in rs), vs)
        check_eq(src, "%s geomean R_K" % tier, geomean(num(r["k_ratio_xconly_over_xck"]) for r in rs), rk)
    Lm = by_mol(sel(rows, tier="L"))
    for mol, rk in zip(["fluconazole", "warfarin", "omeprazole", "celecoxib"],
                       ["1.307", "1.322", "1.233", "1.405"]):
        check_eq(src, "L %s R_K" % mol, num(Lm[mol]["k_ratio_xconly_over_xck"]), rk)
    # per-arm cycles, L (doc table: XC+K, XC-only, stock fp64 base, g4p)
    want = {"fluconazole": (13, 13, 13, 13), "warfarin": (14, 14, 14, 14),
            "omeprazole": (14, 13, 13, 13), "celecoxib": (13, 13, 13, 13)}
    for mol, w in want.items():
        r = Lm[mol]
        got = (int(r["n_cycle_xc_plus_k"]), int(r["n_cycle_xc_only"]),
               int(r["n_cycle_fp64_base"]), int(r["n_cycle_g4p"]))
        check_exact(src, "L %s cycles XCK/XCo/base/g4p" % mol, got, w)
    # total cycles S/M
    for tier, xck, base in [("S", 97, 97), ("M", 149, 149)]:
        rs = sel(rows, tier=tier)
        check_exact(src, "%s total XC+K cycles" % tier, sum(int(r["n_cycle_xc_plus_k"]) for r in rs), xck)
        check_exact(src, "%s total fp64-base cycles" % tier, sum(int(r["n_cycle_fp64_base"]) for r in rs), base)
    SM = [r for r in rows if r["run_id"] == "36134049978"]
    L = sel(rows, run_id="36152138663")
    check_le(src, "S/M max |dE| any arm vs g4p", max(num(r["max_abs_dE_any_spec_arm_vs_g4p_Ha"]) for r in SM), "1.3e-11")
    check_le(src, "L max |dE| any arm vs g4p", max(num(r["max_abs_dE_any_spec_arm_vs_g4p_Ha"]) for r in L), "1.6e-10")
    floor = sorted(r["molecule"] for r in rows if r["under_1s_timing_floor"] == "yes")
    check_exact(src, "sub-1 s timer FAILs", floor, ["metformin", "paracetamol"])


# ---------------------------------------------------------------- 3. cuTENSOR A-B-A
def aba():
    src = "3 cuTENSOR A-B-A"
    rows = load("b3lyp_cutensor_aba_pro6000.csv")
    mols = ["fluconazole", "warfarin", "omeprazole", "celecoxib"]
    legs = {}
    banked = {"E1": ("cupy", "1.797", "1.7971", "1.271", ["1.814", "1.728", "1.831", "1.817"]),
              "C": ("cutensor", "1.921", "1.9211", "1.288", ["1.891", "1.869", "1.906", "2.022"]),
              "E2": ("cupy", "1.801", "1.8008", "1.284", ["1.770", "1.709", "1.848", "1.882"])}
    for leg, (eng, vs3, vs4, rk, per) in banked.items():
        rs = sel(rows, leg=leg)
        check_exact(src, "%s coverage" % leg, len(rs), 4)
        check_exact(src, "%s engine" % leg, sorted(set(r["contract_engine_selected"] for r in rs)), [eng])
        v = geomean(num(r["vs_stock_g4p_over_xck"]) for r in rs)
        legs[leg] = v
        check_eq(src, "%s geomean vs stock" % leg, v, vs3)
        check_eq(src, "%s geomean vs stock (4 dp)" % leg, v, vs4)
        check_eq(src, "%s geomean R_K" % leg, geomean(num(r["k_ratio_xconly_over_xck"]) for r in rs), rk)
        m = by_mol(rs)
        for mol, b in zip(mols, per):
            check_eq(src, "%s %s vs stock" % (leg, mol), num(m[mol]["vs_stock_g4p_over_xck"]), b)
    check_eq(src, "drift d", abs(legs["E2"] / legs["E1"] - 1), "0.0021")
    check_eq(src, "reading h", legs["C"] / math.sqrt(legs["E1"] * legs["E2"]), "1.068")
    M = {leg: by_mol(sel(rows, leg=leg)) for leg in ("E1", "C", "E2")}
    for col, label, gm, per in [
            ("stock_g4p_wall_warm_s", "stock", "1.081", ["1.052", "1.056", "1.084", "1.134"]),
            ("xc_plus_k_wall_s", "specialized", "1.155", ["1.111", "1.149", "1.124", "1.240"])]:
        gains = []
        for mol, b in zip(mols, per):
            g = math.sqrt(num(M["E1"][mol][col]) * num(M["E2"][mol][col])) / num(M["C"][mol][col])
            gains.append(g)
            check_eq(src, "%s cuTENSOR gain %s" % (label, mol), g, b)
        check_eq(src, "%s cuTENSOR gain geomean" % label, geomean(gains), gm)
    check_le(src, "max |dE| any leg any arm", max(num(r["max_abs_dE_any_spec_arm_vs_g4p_Ha"]) for r in rows), "1.7e-10")
    for leg in ("E1", "C", "E2"):
        m = M[leg]
        check_exact(src, "%s spec XC+K cycles" % leg, [int(m[x]["n_cycle_xc_plus_k"]) for x in mols], [13, 14, 14, 13])
        check_exact(src, "%s g4p cycles" % leg, [int(m[x]["n_cycle_g4p"]) for x in mols], [13, 14, 13, 13])


# ---------------------------------------------------------------- 4. gradient XC
def grad():
    src = "4 gradient XC"
    rows = load("r2scan_gradient_xc.csv")
    banked = {
        "pro6000": dict(run="32322726163", sxc="4.0532", sgrad="1.9443", sxc3="4.053", sgrad3="1.944",
                        maxdg="3.183e-06",
                        per={"paracetamol": ("0.5006", "0.1789", "2.7978", "0.742", "0.421", "1.7644", "1.011e-06"),
                             "propranolol": ("1.4679", "0.3365", "4.3621", "2.219", "1.089", "2.0389", "1.317e-06"),
                             "celecoxib": ("2.0506", "0.3758", "5.4562", "3.274", "1.603", "2.0433", "3.183e-06")}),
        "a100": dict(run="32373234782", sxc="0.6325", sgrad="0.7409", sxc3="0.632", sgrad3="0.741",
                     maxdg="3.298e-06",
                     per={"paracetamol": ("0.3301", "0.5146", "0.6414", "0.539", "0.725", "0.7444", "1.012e-06"),
                          "propranolol": ("0.6469", "1.0461", "0.6184", "0.992", "1.392", "0.7127", "1.317e-06"),
                          "celecoxib": ("0.7263", "1.1388", "0.6378", "1.348", "1.759", "0.7667", "3.298e-06")}),
    }
    for gpu, b in banked.items():
        rs = sel(rows, gpu_class=gpu, run_id=b["run"])
        check_exact(src, "%s coverage" % gpu, len(rs), 3)
        sx = geomean(num(r["S_xc"]) for r in rs)
        sg = geomean(num(r["S_grad"]) for r in rs)
        check_eq(src, "%s S_xc geomean" % gpu, sx, b["sxc"])
        check_eq(src, "%s S_xc geomean (3 dp)" % gpu, sx, b["sxc3"])
        check_eq(src, "%s S_grad geomean" % gpu, sg, b["sgrad"])
        check_eq(src, "%s S_grad geomean (3 dp)" % gpu, sg, b["sgrad3"])
        check_eq(src, "%s max |dg|" % gpu, max(num(r["max_abs_dgrad_mp_vs_stock_Ha_per_Bohr"]) for r in rs), b["maxdg"])
        m = by_mol(rs)
        for mol, (xs, xm, s_xc, ts, tm, s_g, dg) in b["per"].items():
            r = m[mol]
            check_eq(src, "%s %s xc_excl stock" % (gpu, mol), num(r["xc_excl_stock_s"]), xs)
            check_eq(src, "%s %s xc_excl mp" % (gpu, mol), num(r["xc_excl_mp_s"]), xm)
            check_eq(src, "%s %s S_xc (=stock/mp)" % (gpu, mol), num(r["xc_excl_stock_s"]) / num(r["xc_excl_mp_s"]), s_xc)
            check_eq(src, "%s %s t_grad stock" % (gpu, mol), num(r["t_grad_warm_median_stock_s"]), ts)
            check_eq(src, "%s %s t_grad mp" % (gpu, mol), num(r["t_grad_warm_median_mp_s"]), tm)
            check_eq(src, "%s %s S_grad (=stock/mp)" % (gpu, mol),
                     num(r["t_grad_warm_median_stock_s"]) / num(r["t_grad_warm_median_mp_s"]), s_g)
            check_eq(src, "%s %s max |dg|" % (gpu, mol), num(r["max_abs_dgrad_mp_vs_stock_Ha_per_Bohr"]), dg)


# ---------------------------------------------------------------- 5. energy trio by GPU
def trio():
    src = "5 energy trio"
    rows = load("r2scan_energy_trio_by_gpu.csv")
    mols = ["paracetamol", "propranolol", "celecoxib"]
    # H100, SPEC-v2 7.2
    h = by_mol(sel(rows, gpu_class="h100", run_id="32079364413"))
    for mol, xb, xs, sp, net in [("paracetamol", "1.743", "1.334", "1.307", "1.596"),
                                 ("propranolol", "3.184", "2.606", "1.222", "1.490"),
                                 ("celecoxib", "3.691", "2.933", "1.258", "1.504")]:
        r = h[mol]
        check_eq(src, "h100 %s xc_base" % mol, num(r["xc_base_s"]), xb)
        check_eq(src, "h100 %s xc_spec" % mol, num(r["xc_spec_s"]), xs)
        check_eq(src, "h100 %s xcspeedup" % mol, num(r["xc_base_s"]) / num(r["xc_spec_s"]), sp)
        check_eq(src, "h100 %s net of shadows" % mol, num(r["xc_base_s"]) / num(r["xc_spec_net_of_shadows_s"]), net)
    check_exact(src, "h100 cycles g4p/spec", [(int(h[m]["n_cycle_g4p"]), int(h[m]["n_cycle_spec"])) for m in mols],
                [(14, 14), (14, 14), (15, 15)])
    # A100, SPEC-v2 7.3.2 / 7.3.3
    a = by_mol(sel(rows, gpu_class="a100", run_id="32372635923"))
    for mol, xb, xs, sp, net, wb, ws, e2e in [
            ("paracetamol", "2.263", "1.860", "1.216", "1.470", "2.701", "2.401", "1.127"),
            ("propranolol", "4.242", "3.631", "1.168", "1.400", "5.458", "4.824", "1.086"),
            ("celecoxib", "4.928", "4.858", "1.015", "1.337", "6.926", "9.029", "0.748")]:
        r = a[mol]
        check_eq(src, "a100 %s xc_base" % mol, num(r["xc_base_s"]), xb)
        check_eq(src, "a100 %s xc_spec" % mol, num(r["xc_spec_s"]), xs)
        check_eq(src, "a100 %s xcspeedup" % mol, num(r["xc_base_s"]) / num(r["xc_spec_s"]), sp)
        check_eq(src, "a100 %s net of shadows" % mol, num(r["xc_base_s"]) / num(r["xc_spec_net_of_shadows_s"]), net)
        check_eq(src, "a100 %s wall_base (spec fp64 base arm)" % mol, num(r["spec_fp64_base_arm_wall_s"]), wb)
        check_eq(src, "a100 %s wall_spec" % mol, num(r["spec_wall_warm_s"]), ws)
        # e2e is the harness's spec.speedup: g4p warm / spec warm. NOTE the doc table's
        # "wall_base" column is the spec side's fp64 base arm, so e2e != wall_base/wall_spec there.
        check_eq(src, "a100 %s e2e (g4p/spec)" % mol, num(r["e2e_g4p_over_spec"]), e2e)
    check_eq(src, "a100 xcspeedup geomean", geomean(num(a[m]["xcspeedup"]) for m in mols), "1.130")
    check_eq(src, "a100 e2e geomean (g4p/spec)", geomean(num(a[m]["e2e_g4p_over_spec"]) for m in mols), "0.971")
    c = a["celecoxib"]
    for col, b in [("g4p_wall_warm_s", "6.7497"), ("spec_wall_warm_s", "9.0285"),
                   ("g4p_df_build_warm_s", "0.3996"), ("spec_df_build_warm_s", "2.7379"),
                   ("g4p_scf_iter_warm_s", "6.2161"), ("spec_scf_iter_warm_s", "6.1480"),
                   ("spec_base_scf_iter_s", "6.202")]:
        check_eq(src, "a100 celecoxib %s" % col, num(c[col]), b)
    check_eq(src, "a100 celecoxib wall delta spec-g4p",
             num(c["spec_wall_warm_s"]) - num(c["g4p_wall_warm_s"]), "2.2788")
    check_eq(src, "a100 celecoxib df_build delta spec-g4p",
             num(c["spec_df_build_warm_s"]) - num(c["g4p_df_build_warm_s"]), "2.3383")
    check_exact(src, "a100 cycles g4p/spec", [(int(a[m]["n_cycle_g4p"]), int(a[m]["n_cycle_spec"])) for m in mols],
                [(14, 14), (14, 14), (15, 15)])
    # PRO 6000 production trio (mp-prod), ARCH-GRADMP 1 / 12.1
    p = by_mol(sel(rows, gpu_class="pro6000", run_id="31978102688"))
    for mol, sp in zip(mols, ["1.886", "1.771", "1.889"]):
        check_eq(src, "pro6000 %s xcspeedup" % mol, num(p[mol]["xc_base_s"]) / num(p[mol]["xc_spec_s"]), sp)
    check_eq(src, "pro6000 e2e geomean (g4p/spec) = s_energy",
             geomean(num(p[m]["e2e_g4p_over_spec"]) for m in mols), "1.697")


# ---------------------------------------------------------------- 6. VV10 wB97M-V
def vv10():
    src = "6 VV10 wB97M-V"
    rows = load("vv10_wb97mv_pro6000.csv")
    mols = ["paracetamol", "propranolol", "celecoxib"]
    stages = {"36769392379": "fp32_commission", "36781246916": "fp32_stock_tail_tol1e-4",
              "36792813279": "fp32_stock_tail_tol1e-5", "36869739118": "df64_commission",
              "36873230133": "df64_tail_shadowed", "36875554583": "df64_tail_production"}
    for run, stage in stages.items():
        rs = sel(rows, run_id=run, stage=stage)
        check_exact(src, "%s coverage" % stage, sorted(r["molecule"] for r in rs), sorted(mols))
    check_exact(src, "total rows", len(rows), 18)

    def cert_rel(r):
        return max(num(r["cert_rel_E"]), num(r["cert_rel_U"]), num(r["cert_rel_W"]))

    # PREREG-vv10-commission.md RESULT: FP32 matrix (selected f32_t1_hilo) and treated SCF
    C = by_mol(sel(rows, run_id="36769392379"))
    check_exact(src, "fp32 commission matrix variant",
                sorted(set(r["matrix_variant"] for r in C.values())), ["f32_t1_hilo"])
    check_eq(src, "fp32 max rel (worst molecule)", max(num(r["matrix_max_rel"]) for r in C.values()), "1.16e-7")
    check_eq(src, "fp32 max |dE_nlc| Ha", max(num(r["matrix_max_abs_dEnlc_Ha"]) for r in C.values()), "1.17e-10")
    c = C["celecoxib"]
    check_eq(src, "fp32 celecoxib kernel median s", num(c["matrix_kernel_median_s"]), "0.168")
    check_eq(src, "fp32 celecoxib stock/FP32",
             num(c["matrix_stock_uwe_median_s"]) / num(c["matrix_kernel_median_s"]), "30.4")
    for mol, b in zip(mols, ["1.18", "4.51", "5.12"]):
        check_eq(src, "stock UWE per call %s s" % mol, num(C[mol]["matrix_stock_uwe_median_s"]), b)
    for mol, ws, wt, de, cyc, nf, sw in [
            ("paracetamol", "19.20", "13.81", "4.5e-13", (12, 12), 6, 7),
            ("propranolol", "76.52", "59.33", "0.0", (13, 13), 5, 6),
            ("celecoxib", "93.00", "73.52", "4.5e-13", (13, 13), 5, 6)]:
        r = C[mol]
        check_eq(src, "fp32 commission %s wall stock s" % mol, num(r["wall_base_s"]), ws)
        check_eq(src, "fp32 commission %s wall treated s" % mol, num(r["wall_spec_s"]), wt)
        check_eq(src, "fp32 commission %s |dE|" % mol, num(r["abs_dE_spec_vs_base_Ha"]), de)
        check_exact(src, "fp32 commission %s cycles" % mol,
                    (int(r["n_cycle_spec"]), int(r["n_cycle_base"])), cyc)
        check_exact(src, "fp32 commission %s FP32 calls / switch" % mol,
                    (int(r["n_fp32"]), int(r["switch_call"])), (nf, sw))

    # PREREG-vv10-production.md RESULT (R2) and PREREG-vv10-tol5.md RESULT: FP32 + stock tail
    for run, label, tol, gm, per, tab in [
            ("36781246916", "tol 1e-4", "0.0001", "1.433", ["1.550", "1.393", "1.363"],
             [("paracetamol", (6, 7, 7), "16.11", "9.33", "1.727", "18.95", "12.18", "1.555", "1.552", "0.0", (12, 12)),
              ("propranolol", (5, 9, 6), "65.11", "43.80", "1.486", "75.50", "54.19", "1.393", "1.399", "0.0", (13, 13)),
              ("celecoxib", (5, 9, 6), "74.64", "50.17", "1.488", "91.81", "67.35", "1.363", "1.363", "9.1e-13", (13, 13))]),
            ("36792813279", "tol 1e-5", "1e-05", "1.643", ["1.690", "1.647", "1.595"],
             [("paracetamol", (7, 6, 8), "16.17", "8.30", "1.949", "19.17", "11.35", "1.690", "1.698", "2.3e-13", (12, 12)),
              ("propranolol", (7, 7, 8), "65.30", "35.39", "1.845", "75.86", "45.99", "1.649", "1.663", "4.5e-13", (13, 13)),
              ("celecoxib", (7, 7, 8), "74.85", "40.56", "1.845", "92.27", "57.95", "1.592", "1.589", "4.5e-13", (13, 13))])]:
        rs = sel(rows, run_id=run)
        check_exact(src, "%s tol / tail / variant / shadowed" % label,
                    sorted(set((r["tol"], r["tail"], r["fp32_variant"], r["shadowed"]) for r in rs)),
                    [(tol, "stock", "f32_t1_hilo", "no")])
        _vv10_scf_table(src, label, by_mol(rs), tab)
        m = by_mol(rs)
        sp = {x: num(m[x]["g4p_wall_warm_s"]) / num(m[x]["wall_spec_s"]) for x in mols}
        check_eq(src, "%s spec.speedup geomean (g4p/spec)" % label, geomean(sp.values()), gm)
        for mol, b in zip(mols, per):
            check_eq(src, "%s spec.speedup %s" % (label, mol), sp[mol], b)

    # PREREG-vv10-df64-commission.md RESULT: df64 matrix and treated SCF with certificate
    D = by_mol(sel(rows, run_id="36869739118"))
    check_exact(src, "df64 commission matrix variant",
                sorted(set(r["matrix_variant"] for r in D.values())), ["df64_f32_t1"])
    for mol, rel, dn in [("paracetamol", "8.5e-14", "2.1e-15"), ("propranolol", "1.8e-13", "1.1e-14"),
                         ("celecoxib", "2.3e-13", "1.3e-14")]:
        check_eq(src, "df64 %s max rel" % mol, num(D[mol]["matrix_max_rel"]), rel)
        check_eq(src, "df64 %s max |dE_nlc| Ha" % mol, num(D[mol]["matrix_max_abs_dEnlc_Ha"]), dn)
    d = D["celecoxib"]
    check_eq(src, "df64 celecoxib kernel median s", num(d["matrix_kernel_median_s"]), "0.959")
    check_eq(src, "df64 celecoxib kernel median s (Amendment 1 k)", num(d["matrix_kernel_median_s"]), "0.9594")
    check_eq(src, "df64 celecoxib stock/df64 (P2)",
             num(d["matrix_stock_uwe_median_s"]) / num(d["matrix_kernel_median_s"]), "5.32")
    check_eq(src, "df64 celecoxib cert wall s (Amendment 1 c)", num(d["cert_wall_s"]), "5.084")
    for mol, rel, dn, w, nf, nt in [("paracetamol", "7.7e-14", "2.0e-15", "1.17", 7, 6),
                                    ("propranolol", "1.6e-13", "1.1e-14", "4.48", 7, 7),
                                    ("celecoxib", "1.5e-13", "1.2e-14", "5.08", 7, 7)]:
        r = D[mol]
        check_exact(src, "df64 commission %s FP32/df64 calls, switch, tail" % mol,
                    (int(r["n_fp32"]), int(r["n_tail"]), int(r["switch_call"]), r["tail"]), (nf, nt, 8, "df64"))
        check_eq(src, "df64 commission %s |dE|" % mol, num(r["abs_dE_spec_vs_base_Ha"]), "4.5e-13")
        check_exact(src, "df64 commission %s cycles" % mol,
                    int(r["n_cycle_spec"]), int(r["n_cycle_base"]))
        check_eq(src, "df64 commission %s cert rel" % mol, cert_rel(r), rel)
        check_eq(src, "df64 commission %s cert |dE_nlc|" % mol, num(r["cert_denlc_Ha"]), dn)
        check_eq(src, "df64 commission %s cert wall s" % mol, num(r["cert_wall_s"]), w)

    # PREREG-vv10-df64-production.md RESULT: R1 (shadowed, disclosed) and R2 (ADOPT)
    R1 = by_mol(sel(rows, run_id="36873230133"))
    check_exact(src, "df64 R1 shadowed", sorted(set(r["shadowed"] for r in R1.values())), ["yes"])
    for mol, rel in zip(mols, ["7.7e-14", "1.6e-13", "1.5e-13"]):
        check_eq(src, "df64 R1 %s cert rel" % mol, cert_rel(R1[mol]), rel)
    c = R1["celecoxib"]
    check_eq(src, "df64 R1 celecoxib e2e (shadowed, disclosed)",
             num(c["wall_base_s"]) / num(c["wall_spec_s"]), "1.225")
    rs = sel(rows, run_id="36875554583")
    P = by_mol(rs)
    check_exact(src, "df64 R2 tol / tail / variant / shadowed",
                sorted(set((r["tol"], r["tail"], r["fp32_variant"], r["shadowed"]) for r in rs)),
                [("1e-05", "df64", "f32_t1_hilo", "no")])
    _vv10_scf_table(src, "df64 R2", P, [
        ("paracetamol", (7, 6, 8), "16.14", "2.63", "6.14", "19.14", "6.81", "2.809", "2.835", "2.3e-13", (12, 12)),
        ("propranolol", (7, 7, 8), "65.08", "10.37", "6.27", "75.73", "25.54", "2.965", "2.974", "9.1e-13", (13, 13)),
        ("celecoxib", (7, 7, 8), "74.61", "12.41", "6.01", "92.16", "34.98", "2.635", "2.636", "4.5e-13", (13, 13))])
    for mol, rel, w in [("paracetamol", "7.8e-14", "1.17"), ("propranolol", "1.6e-13", "4.47"),
                        ("celecoxib", "1.5e-13", "5.07")]:
        r = P[mol]
        check_eq(src, "df64 R2 %s cert rel" % mol, cert_rel(r), rel)
        check_eq(src, "df64 R2 %s cert wall s" % mol, num(r["cert_wall_s"]), w)
        check_exact(src, "df64 R2 %s cert on the last call" % mol,
                    int(r["cert_call"]), int(r["n_fp32"]) + int(r["n_tail"]))
        check_le(src, "df64 R2 %s cert rel in VVDF_REL band" % mol, cert_rel(r), "1e-10")
        check_le(src, "df64 R2 %s cert |dE_nlc| in VVDF_DENLC band" % mol, num(r["cert_denlc_Ha"]), "1e-11")
    check_le(src, "df64 R2 max |dE|", max(num(r["abs_dE_spec_vs_base_Ha"]) for r in rs), "9.1e-13")
    sp = {m: num(P[m]["g4p_wall_warm_s"]) / num(P[m]["wall_spec_s"]) for m in mols}
    check_eq(src, "df64 R2 spec.speedup geomean (g4p/spec)", geomean(sp.values()), "2.798")
    for mol, b in zip(mols, ["2.801", "2.971", "2.633"]):
        check_eq(src, "df64 R2 spec.speedup %s" % mol, sp[mol], b)
    c = P["celecoxib"]
    check_eq(src, "df64 R2 celecoxib cert share of wall s", num(c["cert_wall_s"]), "5.1")
    check_eq(src, "df64 R2 celecoxib wall net of cert s", num(c["wall_spec_s"]) - num(c["cert_wall_s"]), "29.9")
    check_eq(src, "df64 R2 celecoxib e2e net of cert",
             num(c["wall_base_s"]) / (num(c["wall_spec_s"]) - num(c["cert_wall_s"])), "3.08")
    # the stage table (celecoxib, same inputs)
    for run, w, e in [("36781246916", "67.35", "1.363"), ("36792813279", "57.95", "1.592"),
                      ("36875554583", "34.98", "2.635")]:
        r = sel(rows, run_id=run, molecule="celecoxib")[0]
        check_eq(src, "stage table %s celecoxib treated wall s" % run, num(r["wall_spec_s"]), w)
        check_eq(src, "stage table %s celecoxib e2e" % run, num(r["wall_base_s"]) / num(r["wall_spec_s"]), e)


def _vv10_scf_table(src, label, m, tab):
    """One per-molecule scfbench RESULT table: calls, nlc, mechanism, walls, e2e, |dE|, cycles."""
    for mol, calls, nb, ns, mech, wb, ws, e2e, proj, de, cyc in tab:
        r = m[mol]
        check_exact(src, "%s %s FP32/tail calls, switch call" % (label, mol),
                    (int(r["n_fp32"]), int(r["n_tail"]), int(r["switch_call"])), calls)
        check_exact(src, "%s %s switch kind" % (label, mol), r["switch_kind"], "dexc")
        check_eq(src, "%s %s nlc base s" % (label, mol), num(r["nlc_base_s"]), nb)
        check_eq(src, "%s %s nlc spec s" % (label, mol), num(r["nlc_spec_s"]), ns)
        check_eq(src, "%s %s mechanism (=nlc base/spec)" % (label, mol),
                 num(r["nlc_base_s"]) / num(r["nlc_spec_s"]), mech)
        check_eq(src, "%s %s wall base s" % (label, mol), num(r["wall_base_s"]), wb)
        check_eq(src, "%s %s wall spec s" % (label, mol), num(r["wall_spec_s"]), ws)
        check_eq(src, "%s %s e2e (=wall base/spec)" % (label, mol),
                 num(r["wall_base_s"]) / num(r["wall_spec_s"]), e2e)
        check_eq(src, "%s %s e2e as recorded" % (label, mol), num(r["e2e_base_over_spec"]), e2e)
        check_eq(src, "%s %s projected" % (label, mol), num(r["e2e_projected"]), proj)
        check_eq(src, "%s %s |dE|" % (label, mol), num(r["abs_dE_spec_vs_base_Ha"]), de)
        check_exact(src, "%s %s cycles spec/base" % (label, mol),
                    (int(r["n_cycle_spec"]), int(r["n_cycle_base"])), cyc)


def main():
    for fn in (ladder, cg, aba, grad, trio, vv10):
        try:
            fn()
        except Exception as e:  # a missing file or empty selection fails closed
            results.append((fn.__name__, "EXCEPTION", repr(e), "", "FAIL"))
    w = max(len(r[1]) for r in results)
    last = None
    for src, name, got, want, status in results:
        if src != last:
            print("\n== %s" % src)
            last = src
        print("  %-11s %-*s  recomputed=%-12s banked=%s" % (status, w, name, got, want))
    bad = [r for r in results if r[4].startswith("FAIL")]
    known = [r for r in results if r[4] == "DISCREPANCY"]
    ok = len(results) - len(bad) - len(known)
    print("\n%d checks: %d ok, %d known DISCREPANCY (reported, not adjusted), %d FAIL"
          % (len(results), ok, len(known), len(bad)))
    if not results:
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
