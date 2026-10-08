# Sources of the native-port data

Every CSV in `data/` was extracted from one pod's result sentinel, on the `ci-bus` branch of
`chris-lee-mc/gpu-conformer-engine`. The extractor was `.github/scripts/rfcbench_extract.py`
(`--run-id <RUN_ID> --csv`), at the commit that merged each RESULT (for PREREG-7, whose RESULT
is not yet merged, at `d05888b8`; the extractor is byte-identical on the RESULT's branch). A sentinel's line 2 is
`RUN_ID=<RUN_ID>`, and each CSV row carries the same `run_id`; `verify_native.py` checks that
against `data/pods.csv`.

- `data/downstream.csv`: the extractor's downstream readings (`--unquotable` for D1, whose pod
  FAILed one gate), one row per downstream cell.
- `data/tiers.csv`: the ladder's tier table and the six LARGE molecules, from
  `rfcbench_sets.drug_table()` and `rfcbench_sets.LARGE`.
- Geometries: `../geometries/`, all 30 checked by `../geometries/SHA256SUMS`. They are byte-identical
  to the ones the pods read.

`prereg/` mirrors `docs/upstream/PREREG-rfcbench-*.md` of `chris-lee-mc/gpu-conformer-engine`, including the errata (PR #229)
PREREG-7's RESULT and errata from branch `claude/rfcbench-prereg7-results` (PR #231), PREREG-4's
erratum 2 from branch `claude/rfcbench-c4-erratum`, and PREREG-8's RESULT from branch
`claude/prereg8-results` (PR #236). All three are ahead of that repository's `main` until merged.

Fork under test: `chris-lee-mc/gpu4pyscf` at `63af0568d4fd19935bef51b7fd71f161a9cee56f`, overlaid
on `gpu4pyscf-cuda12x==1.8.1` with the lock in `../requirements.lock` and `cutensor-cu12==2.3.1`.

| CSV | scope | GitHub run | repo commit | card | MIG | idle W | contended | cells | NOISY | DEGRADED | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `a1` | PREREG-8 A1 | `37734904781` | `5013b101` | RTX PRO 6000 Blackwell Workstation Edition | — | 65.06 | False | — (96 runs) | — | — | PASS |
| `a2` | PREREG-8 A2 | `37736924342` | `5013b101` | RTX PRO 6000 Blackwell Workstation Edition | — | 53.15 | False | — (16 runs) | — | — | PASS |
| `b1` | PREREG-3 B1 | `37061160282` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 39.42 | False | 24 | 2 | False | PASS |
| `c0` | PREREG-2 C0 (commissioning; CONTENTION-UNKNOWN) | `36951733967` | `0ee605b6` | RTX PRO 6000 Blackwell Workstation Edition | — | unknown | unknown | 10 | 0 | False | PASS |
| `c0w` | PREREG-2 C0w | `37044790110` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 25.5 | False | 3 | 0 | False | PASS |
| `cc1` | PREREG-7 CC1 | `37552079534` | `d05888b8` | RTX PRO 6000 Blackwell Server Edition | — | 86.52 | False | — (120 SCFs) | — | — | PASS |
| `cs1` | PREREG-7 CS1 | `37535025280` | `d05888b8` | RTX PRO 6000 Blackwell Workstation Edition | — | 75.12 | False | — (30 SCFs) | — | — | PASS |
| `d1` | PREREG-4 D1 (STATUS=FAIL, one dipole gate) | `37038641497` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 77.29 | False | 0 | 0 | False | FAIL |
| `d2a` | PREREG-4 D2a | `37040169998` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 55.26 | False | 0 | 0 | False | PASS |
| `d2b` | PREREG-4 D2b | `37042063662` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 25.97 | False | 0 | 0 | False | PASS |
| `d2c` | PREREG-4 D2c | `37043596852` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 27.29 | False | 0 | 0 | False | PASS |
| `f0` | PREREG-6 F0 | `37379456085` | `e9b75f37` | RTX PRO 6000 Blackwell Server Edition | — | 65.62 | False | 6 | 0 | False | PASS |
| `f1a` | PREREG-6 F1a | `37382762854` | `e9b75f37` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 92.84 | False | 3 | 0 | False | PASS |
| `f1b` | PREREG-6 F1b | `37386887345` | `e9b75f37` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 93.96 | False | 3 | 0 | False | PASS |
| `g1` | PREREG-8 G1 | `37738477503` | `5013b101` | RTX PRO 6000 Blackwell Workstation Edition | — | 42.35 | False | — (9 runs) | — | — | PASS |
| `l12` | PREREG-1 L12 | `36968109711` | `78a3412e` | RTX PRO 6000 Blackwell Workstation Edition | — | 78.11 | False | 48 | 0 | False | PASS |
| `l12r` | PREREG-7 L12r (DEGRADED) | `37540378856` | `d05888b8` | RTX PRO 6000 Blackwell Workstation Edition | — | 53.77 | False | 48 | 19 | True | PASS |
| `l12r2` | PREREG-7 L12r2 (DEGRADED) | `37544474854` | `d05888b8` | RTX PRO 6000 Blackwell Workstation Edition | — | 63.24 | False | 48 | 21 | True | PASS |
| `m0` | PREREG-5 M0 | `37073279025` | `3890e545` | RTX PRO 6000 Blackwell Server Edition | — | 72.27 | False | 8 | 1 | False | PASS |
| `m1a` | PREREG-5 M1a | `37261720977` | `bda4cf51` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 153.35 | False | 6 | 0 | False | PASS |
| `m1b` | PREREG-5 M1b | `37263100639` | `bda4cf51` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 159.32 | False | 2 | 0 | False | PASS |
| `m2a` | PREREG-5 M2a (CONTENDED, reported separately) | `37265768276` | `bda4cf51` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 276.13 | True | 6 | 0 | False | PASS |
| `m2ar` | PREREG-5 M2ar | `37310614922` | `bda4cf51` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 95.07 | False | 6 | 0 | False | PASS |
| `m2b` | PREREG-5 M2b (CONTENDED, reported separately) | `37267116454` | `bda4cf51` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 371.86 | True | 2 | 0 | False | PASS |
| `m2br` | PREREG-5 M2br | `37312821303` | `bda4cf51` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 97.93 | False | 2 | 0 | False | PASS |
| `st1` | PREREG-7 ST1 | `37549313181` | `d05888b8` | RTX PRO 6000 Blackwell Workstation Edition | — | 57.18 | False | — (36 SCFs) | — | — | PASS |
| `st2` | PREREG-7 ST2 (CONTENDED, reported separately) | `37550550221` | `d05888b8` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 241.03 | True | — (12 SCFs) | — | — | PASS |
| `st2r` | PREREG-7 ST2r | `37621705258` | `d05888b8` | RTX PRO 6000 Blackwell Server Edition | 1g.24gb | 85.2 | False | — (12 SCFs) | — | — | PASS |
| `w1` | PREREG-1 W1 | `36971083681` | `78a3412e` | RTX PRO 6000 Blackwell Workstation Edition | — | 57.1 | False | 9 | 0 | False | PASS |
| `w2` | PREREG-1 W2 | `36973444064` | `78a3412e` | RTX PRO 6000 Blackwell Workstation Edition | — | 90.63 | False | 3 | 0 | False | PASS |
| `w3` | PREREG-1 W3 | `36975697910` | `78a3412e` | RTX PRO 6000 Blackwell Workstation Edition | — | 92.25 | False | 5 | 0 | False | PASS |
| `w4` | PREREG-1 W4 (CONTENDED, reported separately) | `36977987637` | `78a3412e` | RTX PRO 6000 Blackwell Workstation Edition | — | 575.91 | True | 4 | 0 | False | PASS |
| `w4r` | PREREG-1 W4r | `37010453926` | `78a3412e` | RTX PRO 6000 Blackwell Workstation Edition | — | 75.4 | False | 4 | 0 | False | PASS |
| `w5` | PREREG-1 W5 | `37008076314` | `78a3412e` | RTX PRO 6000 Blackwell Workstation Edition | — | 75.11 | False | 3 | 0 | False | PASS |
| `x2a` | PREREG-2 X2a | `37065229751` | `3890e545` | L40S | — | 105.11 | False | 6 | 1 | False | PASS |
| `x2b` | PREREG-2 X2b | `37067826972` | `3890e545` | L40S | — | 101.33 | False | 3 | 0 | False | PASS |
| `x3` | PREREG-2 X3 | `37070570095` | `3890e545` | H100 80GB HBM3 | — | 114.45 | False | 9 | 0 | False | PASS |
| `x3r` | PREREG-7 X3r | `37548043266` | `d05888b8` | H100 80GB HBM3 | — | 122.61 | False | 9 | 0 | False | PASS |
| `x4` | PREREG-2 X4 (DEGRADED) | `37071560009` | `3890e545` | A100-SXM4-80GB | — | 73.73 | False | 9 | 2 | True | PASS |
| `x4r` | PREREG-2 X4r (DEGRADED) | `37391778846` | `3890e545` | A100-SXM4-80GB | — | 87.08 | False | 9 | 4 | True | PASS |
| `xl1` | PREREG-3 XL1 | `37048299066` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 60.87 | False | 6 | 0 | False | PASS |
| `xl2` | PREREG-3 XL2 | `37051413598` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 28.94 | False | 6 | 0 | False | PASS |
| `xl3a` | PREREG-3 XL3a | `37055501906` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 24.71 | False | 1 | 0 | False | PASS |
| `xl3b` | PREREG-3 XL3b | `37057939484` | `3890e545` | RTX PRO 6000 Blackwell Workstation Edition | — | 37.93 | False | 1 | 0 | False | PASS |

Speed cells only are counted under "cells". Downstream pods (D1, D2a–c) and C0's downstream cell
are in `data/downstream.csv`. PREREG-7's extra-mode pods (CS1, ST1, ST2, ST2r, CC1) have no speed
cells; their CSVs carry one row per SCF, with phase, process, span and exclusive stage timers, and
are written by the same extractor (`extra_rows`).
PREREG-8's pods (A1, A2, G1) ran fork `09271907` (`mixed-precision-geo`, validated by g4pport run
`37725279648`). Their CSVs carry one row per arm. The full gradient and dipole vectors (A1, A2) and
the per-step records (G1) are in the sentinels, and `verify_native.py` N10 reads them from there.

**Not billed, so with no sentinel:**
- the 5090 capacity refusals: runs `37064884792` and `37073071979`;
- two M1a create refusals: a REST schema refusal, and run `37261679254` refused by the serialisation
  gate;
- F0's two runner failures: runs `37369605244` and `37371256270`.
- PREREG-8 A1's first dispatch: run `37728120899`. A pod was launched, but no sentinel was ever
  written; the runner gave up at 4815 s.
