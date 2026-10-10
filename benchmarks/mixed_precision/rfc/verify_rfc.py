#!/usr/bin/env python3
"""Check every number in rfc/RFC_ISSUE.md and rfc/PR_BODY.md against the evidence package.

Stdlib only. Two tiers, both fail closed:

  BOUND     headline figures, each tied by name to one check of native/verify_native.py. The
            figure must appear verbatim in the file, and the verifier's recomputed value (from
            data/, not from CLAIMS.md) must round to it at the figure's printed precision.
  COVERED   every other numeric token in the two files must equal, at its printed precision,
            a number printed in native/CLAIMS.md, a value recomputed by verify_native.py, or an
            entry of ALLOWLIST (code constants, test counts and the GPU validation run, each with
            its source). A token covered by nothing is printed as UNVERIFIED and fails the run.

Also fails on: a BOUND figure missing from its file (a stale binding), a forbidden string (model
names, secrets), a mention of the rejected geometry-safety options without "not proposed" on the
same line, and a verify_native.py that does not itself pass.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
NATIVE = os.path.join(PKG, "native")
FILES = ["RFC_ISSUE.md", "PR_BODY.md"]

# --- tier A: headline figures bound to verify_native.py checks (check name -> figure as printed)
BOUND = {
    "RFC_ISSUE.md": [
        ("L12 r2scan ladder geomean (24)", "1.657"),
        ("L12 b3lyp ladder geomean (24)", "1.783"),
        ("wB97M-V ladder geomean (24, healthy pods)", "2.914"),
        ("XL r2SCAN geomean (6)", "1.931"),
        ("XL r2SCAN geomean without ritonavir (5)", "1.802"),
        ("XL B3LYP geomean (6)", "1.338"),
        ("L40S r2scan mixed/stock", "1.729"),
        ("L40S b3lyp mixed/stock", "1.524"),
        ("L40S wb97m-v mixed/stock", "2.994"),
        ("H100 r2scan mixed/stock", "1.320"),
        ("H100 b3lyp mixed/stock", "0.905"),
        ("H100 wb97m-v mixed/stock", "0.577"),
        ("H100 r2scan mixed/cache (precision effect)", "1.084"),
        ("r2scan: lowest max |g_M - g_S| (Ha/Bohr)", "3.8e-9"),
        ("r2scan: highest max |g_M - g_S| (Ha/Bohr)", "5.3e-8"),
        ("B3LYP but omeprazole: lowest max |g_M - g_S|", "2.1e-8"),
        ("B3LYP but omeprazole: highest max |g_M - g_S|", "2.7e-7"),
        ("wb97m-v: lowest max |g_M - g_S| (Ha/Bohr)", "1.2e-8"),
        ("wb97m-v: highest max |g_M - g_S| (Ha/Bohr)", "3.2e-8"),
        ("r2scan: lowest max |mu_M - mu_S| (D)", "3.0e-7"),
        ("r2scan: highest max |mu_M - mu_S| (D)", "2.4e-6"),
        ("B3LYP but omeprazole: lowest max |mu_M - mu_S| (D)", "2.0e-7"),
        ("B3LYP but omeprazole: highest max |mu_M - mu_S| (D)", "9.3e-6"),
        ("wb97m-v: lowest max |mu_M - mu_S| (D)", "1.2e-12"),
        ("wb97m-v: highest max |mu_M - mu_S| (D)", "5.9e-12"),
        ("omeprazole B3LYP: max |g_M - g_S|", "4.4e-6"),
        ("omeprazole B3LYP: S-R over M-R", "3.4"),
        ("lowest max |g_M - g_R| over 18 cells", "6.6e-7"),
        ("highest max |g_S - g_R| over 18 cells", "3.4e-6"),
        ("r2scan: lowest max |g_Mt - g_St|", "1.5e-9"),
        ("r2scan: highest max |g_Mt - g_St|", "2.3e-8"),
        ("b3lyp: lowest max |g_Mt - g_St|", "6.8e-9"),
        ("b3lyp: highest max |g_Mt - g_St|", "2.2e-7"),
        ("wb97m-v: lowest max |g_Mt - g_St|", "1.1e-8"),
        ("wb97m-v: highest max |g_Mt - g_St|", "2.1e-8"),
        ("G1 paracetamol/r2scan: warm-step SCF wall, S / M0", "1.41"),
        ("G1 paracetamol/b3lyp: warm-step SCF wall, S / M0", "1.42"),
        ("G1 celecoxib/r2scan: warm-step SCF wall, S / M0", "1.44"),
        ("G1 paracetamol/r2scan: whole optimisation (SCF + gradient), S / MW", "1.28"),
        ("G1 paracetamol/b3lyp: whole optimisation (SCF + gradient), S / MW", "1.19"),
        ("G1 celecoxib/r2scan: whole optimisation (SCF + gradient), S / MW", "1.30"),
        ("G1 paracetamol/r2scan: endpoint RMSD M0 vs S, aligned (A)", "9.9e-6"),
        ("G1 paracetamol/b3lyp: endpoint RMSD M0 vs S, aligned (A)", "1.5e-6"),
        ("G1 celecoxib/r2scan: endpoint RMSD M0 vs S, aligned (A)", "4.0e-7"),
        ("G1 paracetamol/r2scan: M0 endpoint stock grms (Ha/Bohr)", "7.2e-5"),
        ("G1 celecoxib/r2scan: M0 endpoint stock gmax (Ha/Bohr)", "2.3e-4"),
        ("G1: largest |E_stock(mixed endpoint) - E_S(final)| (Ha)", "3.2e-9"),
        # PREREG-9, library defaults (CLAIMS C7)
        ("LD trio geomean r2scan mixed-default/stock-default", "1.577"),
        ("LD trio geomean b3lyp mixed-default/stock-default", "1.636"),
        ("LD trio geomean wb97m-v mixed-default/stock-default", "3.094"),
        ("LD trio geomean r2scan mixed-nocache-campaign/stock-campaign", "1.627"),
        ("LD trio geomean b3lyp mixed-nocache-campaign/stock-campaign", "1.954"),
        ("LD trio geomean wb97m-v mixed-nocache-campaign/stock-campaign", "2.884"),
        ("LD trio geomean r2scan mixed-campaign/stock-campaign", "1.678"),
        ("LD trio geomean b3lyp mixed-campaign/stock-campaign", "1.968"),
        ("LD trio geomean wb97m-v mixed-campaign/stock-campaign", "2.944"),
        ("PREREG-9 anchor: banked C1 trio r2scan, recomputed from L12", "1.638"),
        ("PREREG-9 anchor: banked C1 trio b3lyp, recomputed from L12", "1.860"),
        ("PREREG-9 anchor: banked C1 trio wb97m-v, recomputed from W1/W4r", "2.803"),
        ("anchor delta b3lyp: LD mixed-campaign/stock-campaign trio minus the banked 1.860", "0.108"),
        ("anchor delta wb97m-v: LD mixed-campaign/stock-campaign trio minus the banked 2.803", "0.141"),
        ("LD trio r2scan: stock-default/stock-campaign (the grid, on stock)", "1.356"),
        ("LD trio b3lyp: stock-default/stock-campaign (the grid, on stock)", "1.278"),
        ("LD trio wb97m-v: stock-default/stock-campaign (the grid, on stock)", "1.035"),
        ("LD trio b3lyp: mixed-campaign/mixed-nocache-campaign (the cache)", "1.007"),
    ],
    "PR_BODY.md": [
        ("LD trio geomean r2scan mixed-default/stock-default", "1.577"),
        ("LD trio geomean b3lyp mixed-default/stock-default", "1.636"),
        ("LD trio geomean wb97m-v mixed-default/stock-default", "3.094"),
        ("LD trio geomean r2scan mixed-nocache-campaign/stock-campaign", "1.627"),
        ("LD trio geomean b3lyp mixed-nocache-campaign/stock-campaign", "1.954"),
        ("LD trio geomean wb97m-v mixed-nocache-campaign/stock-campaign", "2.884"),
        ("LD trio geomean r2scan mixed-campaign/stock-campaign", "1.678"),
        ("LD trio geomean b3lyp mixed-campaign/stock-campaign", "1.968"),
        ("LD trio geomean wb97m-v mixed-campaign/stock-campaign", "2.944"),
        ("anchor delta b3lyp: LD mixed-campaign/stock-campaign trio minus the banked 1.860", "0.108"),
        ("anchor delta wb97m-v: LD mixed-campaign/stock-campaign trio minus the banked 2.803", "0.141"),
        ("LD trio r2scan: stock-default/stock-campaign (the grid, on stock)", "1.356"),
        ("LD trio b3lyp: stock-default/stock-campaign (the grid, on stock)", "1.278"),
        ("LD trio wb97m-v: stock-default/stock-campaign (the grid, on stock)", "1.035"),
        ("LD trio b3lyp: mixed-campaign/mixed-nocache-campaign (the cache)", "1.007"),
        ("L12 r2scan ladder geomean (24)", "1.657"),
        ("L12 b3lyp ladder geomean (24)", "1.783"),
        ("wB97M-V ladder geomean (24, healthy pods)", "2.914"),
        ("XL r2SCAN geomean (6)", "1.931"),
        ("XL B3LYP geomean (6)", "1.338"),
        ("L40S r2scan mixed/stock", "1.729"),
        ("L40S b3lyp mixed/stock", "1.524"),
        ("L40S wb97m-v mixed/stock", "2.994"),
        ("H100 r2scan mixed/stock", "1.320"),
        ("H100 b3lyp mixed/stock", "0.905"),
        ("H100 wb97m-v mixed/stock", "0.577"),
        ("H100 r2scan mixed/cache (precision effect)", "1.084"),
        ("r2scan: lowest max |g_M - g_S| (Ha/Bohr)", "3.8e-9"),
        ("r2scan: highest max |g_M - g_S| (Ha/Bohr)", "5.3e-8"),
        ("B3LYP but omeprazole: lowest max |g_M - g_S|", "2.1e-8"),
        ("B3LYP but omeprazole: highest max |g_M - g_S|", "2.7e-7"),
        ("wb97m-v: lowest max |g_M - g_S| (Ha/Bohr)", "1.2e-8"),
        ("wb97m-v: highest max |g_M - g_S| (Ha/Bohr)", "3.2e-8"),
        ("B3LYP but omeprazole: highest max |mu_M - mu_S| (D)", "9.3e-6"),
        ("lowest max |g_M - g_R| over 18 cells", "6.6e-7"),
        ("highest max |g_S - g_R| over 18 cells", "3.4e-6"),
        ("r2scan: lowest max |g_Mt - g_St|", "1.5e-9"),
        ("r2scan: highest max |g_Mt - g_St|", "2.3e-8"),
        ("b3lyp: lowest max |g_Mt - g_St|", "6.8e-9"),
        ("b3lyp: highest max |g_Mt - g_St|", "2.2e-7"),
        ("wb97m-v: lowest max |g_Mt - g_St|", "1.1e-8"),
        ("wb97m-v: highest max |g_Mt - g_St|", "2.1e-8"),
        ("G1 paracetamol/r2scan: endpoint RMSD M0 vs S, aligned (A)", "9.9e-6"),
        ("G1 paracetamol/r2scan: warm-step SCF wall, S / M0", "1.41"),
        ("G1 celecoxib/r2scan: warm-step SCF wall, S / M0", "1.44"),
        ("G1 paracetamol/b3lyp: whole optimisation (SCF + gradient), S / MW", "1.19"),
        ("G1 celecoxib/r2scan: whole optimisation (SCF + gradient), S / MW", "1.30"),
        ("G1: largest |E_stock(mixed endpoint) - E_S(final)| (Ha)", "3.2e-9"),
        ("CC1 stock: G4-MPS", "1.52"),
        ("CC1 mixed: G4-MPS", "1.75"),
        ("CS1 P1_stock_first: r2scan cold extra (s)", "134.0"),
    ],
}

# --- tier B allowlist: numbers that are facts of the code, the tests or the GPU validation run,
# not of the benchmark data. Each with its source. Nothing else is allowed outside CLAIMS.md and
# the verifier's recomputed values.
ALLOWLIST = {
    # gpu4pyscf/dft/mixed_precision.py @ 4d2f3de4: MixedPrecision defaults and module constants
    "1e-3": "mixed_precision.py: XC_SWITCH_TOL = K_SWITCH_TOL = 1e-3",
    "2": "mixed_precision.py: SWITCH_STALL = 2; also 'two consecutive iterations' (fp64_tail); '2×T4'",
    "30": "mixed_precision.py: SWITCH_CALL_CAP = 30",
    "0.7": "mixed_precision.py: AO_CACHE_MEM_FRACTION = 0.7",
    # gpu4pyscf/dft/vv10_mixed.py @ 4d2f3de4
    "1e-5": "vv10_mixed.py: VV10_SWITCH_TOL = 1e-5; upstream tdscf/ris.py warns below conv_tol 1e-5",
    "1e-10": "vv10_mixed.py: VVDF_REL = 1e-10 (certificate band, relative)",
    "1e-11": "vv10_mixed.py: VVDF_DENLC = 1e-11 (certificate band, Ha); tests: SAME = 1e-11",
    "48": "vv10_mixed.py / mixed_precision.py docstring: df64 is ~48-bit",
    # tests @ 4d2f3de4
    "1e-8": "test_mixed_precision.py: ETOL = 1e-8",
    "32": "test_mixed_precision.py: 32 test methods (21 KnownValues + 10 AOCache + 1 MultiGPU)",
    "18": "test_mixed_precision_vv10.py: 18 test methods; also CLAIMS C4 (18 cells); 'under 18 s'",
    "50": "32 + 18 new tests; g4psrc r4 50/50 on all three profiles",
    "22": "git show --numstat 920bf64: 21 KnownValues + 1 MultiGPU tests in the first commit",
    # git diff --numstat c1a6e37 4d2f3de4
    "2931": "git diff --numstat c1a6e37 4d2f3de4: total insertions",
    "39": "git diff --numstat c1a6e37 4d2f3de4: total deletions (39 on code, 0 on tests/README)",
    "2070": "git diff --numstat, gpu4pyscf/ minus tests: insertions",
    "963": "numstat: dft/mixed_precision.py +963",
    "915": "numstat: dft/vv10_mixed.py +915",
    "928": "git show --numstat 920bf64 (the XC + K commit): +928",
    "17": "git show --numstat 920bf64: -17; CLAIMS C4: 17 of 18 cells",
    "77": "numstat: df/df_jk.py +77",
    "3": "numstat: df/df_jk.py -3; CLAIMS: '3 cards', '3 %'",
    "45": "numstat: dft/rks.py +45",
    "4": "numstat: dft/rks.py -4; RFC: 'four commits'",
    "29": "numstat: dft/numint.py +29",
    "20": "numstat: dft/numint.py -20; CLAIMS: '20–44 atoms'",
    "41": "numstat: scf/hf.py +41",
    "12": "numstat: scf/hf.py -12; CLAIMS: '12 readings'",
    "9": "nine files in the diff; CLAIMS: '9 cells'",
    # g4psrc r4 (three profiles on fork 4d2f3de4, master c1a6e37, built from source);
    # validation/VALIDATION-g4psrc.md and g4psrc_r4_*.json carry the records
    "38022477235": "g4psrc r4 pro6000-lock run id (engine repo)",
    "38023769024": "g4psrc r4 pro6000-pyscf28 run id (engine repo)",
    "38024957471": "g4psrc r4 v100-ci run id (engine repo)",
    "44": "g4psrc r4: upstream regression 44 passed (+ 1 skipped) on both trees; CLAIMS: '20–44 atoms'",
    "1": "g4psrc r4: 1 skipped; CLAIMS: 'above 1'",
    "5.9e-12": "g4psrc r4 pro6000-lock: trio |dE| <= 5.9e-12 Ha (g4psrc_r4_pro6000-lock.json)",
    "2.7e-12": "g4psrc r4 pro6000-pyscf28: trio |dE| <= 2.7e-12 Ha (g4psrc_r4_pro6000-pyscf28.json)",
    "133": "g4psrc r4 v100-ci: new_tests wall_s 132.8 (g4psrc_r4_v100-ci.json)",
    "18.3": "g4psrc r4 v100-ci: slowest test 18.29 s, test_vv10_paracetamol (durations in the JSON)",
    "2.8": "upstream unittest.yml multi-gpu job: pip install pyscf==2.8; g4psrc r4 pro6000-pyscf28 ran 2.8.0",
    "0.9": "upstream unittest.yml single-gpu job: gpu4pyscf-libxc-cuda12x 0.9.0; g4psrc r4 v100-ci ran it",
    "1.17": "upstream requirements / multi-gpu job: scipy 1.17, not tested (r4 Amendment 1)",
    # RFC prose
    "6": "six questions; CLAIMS: six molecules",
    "5": "CLAIMS C1: 5 healthy pods",
    "11": "CLAIMS C3: 11 of 12 A100 clean cells",
    "24": "CLAIMS C1: 24 molecules",
}

FORBIDDEN = ["claude", "opus", "anthropic", "gpt-", "openai", "runpod_api", "api_key",
             "api key", "secret"]
GEO_OPTIONS = ["diis_reset_at_switch", "warm_start_gorb", "DIIS reset", "warm-start rule"]

NUM = re.compile(r"(?<![A-Za-z0-9_.#/+])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# a git SHA: 7-40 hex characters with at least one letter (so a long run id is still a number)
SHA = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")


def present(printed, text):
    """The figure appears in the text as a whole number, not as part of a longer one."""
    return re.search(r"(?<![\d.])" + re.escape(printed) + r"(?!\d|\.\d|[eE][-+]?\d)", text) is not None


def precision(s):
    """('sci', significant figures) or ('fixed', decimals) of a printed number."""
    s = s.lstrip("+-")
    if "e" in s.lower():
        mant = s.lower().split("e")[0]
        return "sci", len(mant.replace(".", "").lstrip("0") or "0")
    return "fixed", (len(s.split(".")[1]) if "." in s else 0)


def same(printed, value):
    kind, p = precision(printed)
    try:
        v = float(value)
        x = float(printed)
    except ValueError:
        return False
    if kind == "sci":
        return f"{v:.{max(p - 1, 0)}e}" == f"{x:.{max(p - 1, 0)}e}"
    return f"{v:.{p}f}" == f"{x:.{p}f}"


def tokens(text):
    out = []
    for ln, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("<!--") or line.strip().startswith("-->"):
            continue
        scrub = DATE.sub(" ", line)
        scrub = SHA.sub(" ", scrub)
        scrub = re.sub(r"^\s*\d+\.\s", " ", scrub)          # ordered-list markers
        scrub = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", scrub)
        for m in NUM.finditer(scrub):
            tok = m.group(0).lstrip("+")
            out.append((ln, tok, line.strip()))
    return out


def main():
    sys.path.insert(0, NATIVE)
    cwd = os.getcwd()
    os.chdir(NATIVE)
    try:
        import verify_native as vn          # runs every check at import
        integrity = vn.data_integrity()
    finally:
        os.chdir(cwd)
    if integrity:
        for p in integrity:
            print(f"FAIL data  {p}")
        return 1
    failures = 0
    by_name = {}
    for claim, what, got, banked, ok in vn.results:
        by_name[what] = (got, ok, claim)
        if not ok and (claim, what) not in vn.KNOWN_DISCREPANCIES:
            print(f"FAIL native  {claim} {what}: {got} (banked {banked})")
            failures += 1
    if len(vn.results) != vn.EXPECTED_CHECKS:
        print(f"FAIL native  ran {len(vn.results)} checks, expected {vn.EXPECTED_CHECKS}")
        failures += 1

    # numbers CLAIMS.md prints, and numbers the verifier recomputed (as printed strings)
    with open(os.path.join(NATIVE, "CLAIMS.md")) as fh:
        claims_tokens = {t for _, t, _ in tokens(fh.read())}
    verifier_values = [got for got, ok, claim in by_name.values() if got not in ("True", "False")]

    for name in FILES:
        path = os.path.join(HERE, name)
        with open(path) as fh:
            text = fh.read()
        low = text.lower()
        print(f"\n== {name}")
        for word in FORBIDDEN:
            if word in low:
                print(f"FAIL forbidden string {word!r} in {name}")
                failures += 1
        # the rejected geometry-safety options may be mentioned only as "not proposed", in the
        # same paragraph (a blank-line-delimited block) as the mention
        for para in re.split(r"\n\s*\n", text):
            p = " ".join(para.split()).lower()
            if any(g.lower() in p for g in GEO_OPTIONS) and "not proposed" not in p:
                print(f"FAIL {name}: geometry-safety option mentioned without 'not proposed': "
                      f"{p[:90]!r}")
                failures += 1

        # tier A
        n_bound = 0
        for check, printed in BOUND.get(name, []):
            if check not in by_name:
                print(f"FAIL bound  no verify_native check named {check!r}")
                failures += 1
                continue
            got, ok, claim = by_name[check]
            if not present(printed, text):
                print(f"FAIL bound  {printed!r} ({check}) is not in {name}: stale binding")
                failures += 1
                continue
            if not same(printed, got):
                print(f"FAIL bound  {name}: {printed} vs verifier {got} ({claim} {check})")
                failures += 1
                continue
            n_bound += 1
        print(f"bound: {n_bound}/{len(BOUND.get(name, []))} headline figures match "
              f"verify_native.py recomputed values")

        # tier B
        bound_printed = {p for _, p in BOUND.get(name, [])}
        covered = unverified = 0
        seen = set()
        for ln, tok, line in tokens(text):
            if tok in bound_printed or tok in ALLOWLIST:
                covered += 1
                continue
            # the same number at the same printed precision: 1.66 does not pass on 1.657
            if any(same(tok, c) and precision(tok) == precision(c) for c in claims_tokens):
                covered += 1
                continue
            if any(same(tok, v) and precision(tok) == precision(v) for v in verifier_values):
                covered += 1
                continue
            unverified += 1
            if (ln, tok) not in seen:
                seen.add((ln, tok))
                print(f"UNVERIFIED {name}:{ln}: {tok}    | {line[:100]}")
        print(f"covered: {covered} numeric tokens (CLAIMS.md, verifier values or ALLOWLIST); "
              f"unverified: {unverified}")
        failures += unverified

    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
