# PREREG rfcbench-0: measurand, harness gates and rules for the RFC evidence campaign

**Status 2026-10-02: PRE-REGISTERED, not yet run.** Written before any `rfcbench` harness code and
before any dispatch. Freezes at the first `rfcbench` sentinel (the commissioning pod C0). After
that, changes go only in dated amendments above RESULT, written before the dispatch they govern.

**What is measured:** the native mixed-precision port on the owner's GPU4PySCF fork, branch
`mixed-precision-aocache-v1.8.1`, whose `gpu4pyscf/` tree is the one validated at
`VALIDATED_FORK_SHA = 63af0568d4fd19935bef51b7fd71f161a9cee56f` (run `36942908651`,
`PORT-PLAN-ao-cache.md` RESULT). Base v1.8.1, overlaid onto the `gpu4pyscf-cuda12x==1.8.1` wheel.

**Owner decisions of 2026-10-01/02:** all four scopes (native ladder, cross-GPU do-no-harm, larger
molecules and bases, downstream accuracy), 15 pods; then a MIG scope, 2–3 pods. No upstream issue
or PR.

The scoped documents are:
- `PREREG-rfcbench-1-native-ladder.md`
- `PREREG-rfcbench-2-cross-gpu.md`
- `PREREG-rfcbench-3-large-basis.md`
- `PREREG-rfcbench-4-downstream.md`
- `PREREG-rfcbench-5-mig.md`

Each carries its own predictions, readings and RESULT. Everything below applies to all of them.

## 1. Measurand

**Whole-SCF warm wall, arm against arm, in one process on one pod.** A *cell* is (molecule, xc,
basis).

1. **Cold pair** (pair 0): every arm once, in arm order. Disclosed only; it absorbs JIT and DF-build
   cold cost.
2. **R warm pairs** (default R = 3, closed set 2..5). Each pair runs the arms in a fixed order with
   **stock last**, so stock gets the warm-pool advantage and the bias runs against the claim. The
   pairs interleave, so drift hits every arm.
3. **Each run is built fresh.** Every run builds a new `pyscf.M` and a new
   `rks.RKS(...).density_fit(auxbasis=...)`, and times `kernel()` between two
   `cupy.cuda.runtime.deviceSynchronize()` calls with `time.perf_counter()`. The DF build is inside
   the wall.
4. **Ratios.**
   - Per-cell ratio of arm A over arm B: median(B walls) / median(A walls) over the R warm pairs.
   - Per-pair ratios and per-arm spread, (max − min) / median, are disclosed.
   - Aggregates are geomeans of per-cell ratios over a stated set.

**Arms.**

| arm | policy |
|---|---|
| `stock` | `mf.mixed_precision = None` |
| `mixed` | the validated settings, **with `ao_cache_fp64=True`**: r2SCAN `MixedPrecision(xc=True, ao_cache_fp64=True)`; B3LYP `MixedPrecision(xc=True, k=True, xc_switch_tol=3e-4, k_switch_tol=1e-3, ao_cache_fp64=True)`; wB97M-V `MixedPrecision(vv10=True, ao_cache_fp64=True)` |
| `cache` | `MixedPrecision(ao_cache_fp64=True)`, every precision component off. Arithmetic is stock FP64, AOs are cached. It separates the cache's gain from the precision's gain. Used where a scope says so |

Order within a pair: `mixed`, then `cache` (if present), then `stock`. There is no per-card or
per-molecule tuning, ever. `rec['policy']` is disclosed verbatim.

**Settings, read back off the built objects, never echoed:**
- `grids.level = 3` and `grids.prune = None`; `ngrids` is recorded;
- `conv_tol = 1e-9` and `max_cycle = 60`, re-pinned after `density_fit()`;
- `conv_tol_grad` at the library default, except in a pre-registered bridge leg;
- basis `def2-mtzvpp` with `auxbasis = def2-tzvpp-jkfit`, unless a scope says otherwise; the
  auxbasis is read back from `mf.with_df.auxbasis`, and `naux` is recorded;
- **cuTENSOR on, and observed.** The pod checks `gpu4pyscf.lib.cutensor.cutensor is not None`, that
  `contract_engine` is not `'cupy'`, and a one-contraction self-test against `cupy.einsum`. A
  failure makes the whole pod INVALID;
- the 14-pin `scfbench_requirements.lock`, unchanged. A pin change requires re-commissioning.

## 2. Gates (GATED; any miss is a cell FAIL, and a cell FAIL is a run FAIL)

**Harness.**
- The observed card (`nvidia-smi`) is the requested class's single model; a landing on any other
  card means no row.
- The fork SHA resolves.
- `git diff --quiet VALIDATED_FORK_SHA FORK_SHA -- gpu4pyscf/`, so only a validated tree can be
  measured.
- The base-identity check passes for every overlaid file.
- cuTENSOR is observed.
- The sentinel's RUN_ID matches the run.
- Every requested cell is present exactly once.

**Accuracy, per cell, against stock.**
- Every run converged.
- max |E_arm − E_stock| ≤ 1e-8 Ha over all pairs, for both `mixed` and `cache`.
- |cycles_arm − cycles_stock| ≤ 1 on every pair.
- `fp64_tail` is True.
- The `cache` arm ran stock arithmetic only: no `fp32` or `df64` in `xc`, `k` or `vv10`.

**The treatment really ran (`mixed` arm).**
- r2SCAN and B3LYP: at least one FP32 XC call, and the last two XC calls FP64.
- B3LYP also: at least one FP32 K call, the last two K calls FP64, and `k_full_rebuild_call` set.
- wB97M-V: the VV10 gates of `g4pport_trio.vv10_problems`, verbatim, including the certificate
  (every rel ≤ 1e-10, denlc ≤ 1e-11 Ha).

**The cache really ran** (`mixed` and `cache` arms), as gates (g)–(h) of
`PORT-PLAN-ao-cache.md`:
- `xc_fp64_stock == 0`;
- the tier is `'fp64+fp32'` (mixed r2SCAN/B3LYP) or `'fp64'` (mixed wB97M-V, and every `cache`
  arm);
- for the `fp64+fp32` tier, the mirror is released on the XC switch call.

**Fallback tiers.** A cell whose tier is not the expected one is classified, never read as 1.0:
- `CACHE_FALLBACK`: tier `'fp32'`, or tier `None` with XC still FP32;
- `NOT_TREATED`: the XC FP32 copy did not fit, so XC is FP64 throughout.

In modes `commission`, `ladder`, `cross` and `downstream` either one is a **FAIL**. In modes
`large`, `basis` and `mig` it is a **pre-registered outcome**: the cell is kept, excluded from the
treated speed aggregates, and listed with its tier and memory figures in a "fit" table.

## 3. Disclosed, never gated

- **Speed.** All speed readings are computed from banked sentinels by the extractor, never on the
  pod.
- **Determinism floor.** The stock-vs-stock |ΔE| across repeats.
- **Health.** `forced` per run; the certificate wall share; cache GiB; `cderi_prebuilt`;
  `nvidia-smi` free and total memory before each group.
- **Canary.** Paracetamol r2SCAN stock warm wall. Its band on the PRO 6000 is 3.8–5.5 s; ≥ 6.5 s
  reads as a degraded pod.
- **Noise flags.**
  - A cell whose spread on either arm exceeds 10 % is flagged `NOISY`, and **retained**.
  - A pod with more than 20 % `NOISY` cells is DEGRADED. Its aggregates are not quotable, and the
    same SHA is re-dispatched.
  - Walls below 1 s are flagged `under_1s`, and retained.

## 4. Readings vocabulary

- **PAYS / NEUTRAL / HARMS** on a ratio r: PAYS r ≥ 1.15, NEUTRAL 0.95 < r < 1.15, HARMS r ≤ 0.95.
- **MODEL-MISS**: a prediction missed by more than its pre-registered band. It is disclosed, and a
  miss is never "fixed" by retuning.
- **Fit outcomes**: `TREATED`, `CACHE_FALLBACK` or `NOT_TREATED`, as above.

## 5. Harness and sentinel

The harness is new and dispatch-only. It never touches a kernel-validation watched path, and nothing
in `g4pport*`, `runpod_scfbench.py` or `bench.py` is modified. The files are:
- `.github/workflows/rfcbench.yml` (`workflow_dispatch` only, no `paths:`);
- `.github/scripts/runpod_rfcbench.py`, a copy of `runpod_g4pport.py`;
- `.github/scripts/rfcbench_pod.py`, which copies the fetch, base-identity and overlay steps of
  `g4pport_pod.py`;
- `.github/scripts/rfcbench_cells.py`, which uses only `MixedPrecision` and
  `mf.mixed_precision_record` and imports nothing from `scf_specialized` or `scf_headtohead.bench`;
- `.github/scripts/rfcbench_sets.py` (stdlib only);
- `.github/scripts/rfcbench_extract.py`;
- `tests/test_rfcbench.py`.

**Sentinel.** Banked immutably at `ci-results/runs/rfcbench-<sha12>-<RUN_ID>.txt`, with
`BANK_REFUSED` on a collision, plus a pointer at `ci-results/rfcbench-<sha12>.txt`. The header
carries `STATUS, RUN_ID, MODE=rfcbench, SHA, FORK_SHA, GPU_CLASS, RFC_MODE, XCS, MOLS, BASIS,
REPEATS, WORK_RC`. One `RFCBENCH_CELL {json}` line is streamed per pair, then `RFCBENCH_JSON`, then a
whole-line `RFCBENCH_PASS` or `RFCBENCH_FAIL`.

**Degrade gracefully.** Every cell runs in its own try/except, and its `cell_status` is one of
`OK, GATE_FAIL, CACHE_FALLBACK, NOT_TREATED, ERROR, CAP_KILLED`. Each xc group runs in its own
subprocess under a timeout. The sentinel is written whatever happens. A `remainder` dispatch (same
inputs, only the missing molecules) recovers cap-killed cells.

**Budgets.**
- `WORK_CAP < FAILSAFE < TIMEOUT_S`, as in g4pport (4440 / 4740 / 4800 s).
- The free runner refuses a dispatch whose over-estimate exceeds `WORK_CAP − 300` s.
- The free runner refuses to launch while any other workflow run is `queued` or `in_progress`.

**GPU-type probe** (`mode=gputypes`). No pod. The runner lists RunPod GPU type ids and memory,
filtered to the PRO 6000 family. It costs nothing on RunPod and exists to read the MIG type id
before `PREREG-rfcbench-5-mig.md` is dispatched.

## 6. Rules

- **Serialisation.** One pod at a time; `queued` and `in_progress` are both checked before every
  dispatch.
- **Failed runs.** A FAIL is diagnosed before re-dispatch. An infra FAIL (`no result sentinel`,
  `ModuleNotFoundError`, NVML, empty output, capacity) is re-dispatched unchanged. A code FAIL is
  fixed under a dated amendment written before the re-dispatch.
- **`WORK_RC = 2`.** `bash -n` on the generated work script locally before any re-dispatch.
- **Capacity.** Per card, up to three attempts over two days, then `n/a-by-capacity` with the
  attempt log. A cascade is never widened to get a number.
- **Repeats.** Two consecutive infra-starved repeats of the same pod means accepting the clean data
  that exists.

## Amendment rule

Amendments are dated, placed above RESULT, and written before the dispatch they govern. None changes
a gate after the sentinel it would judge has landed.

## Amendments

None yet.

## Amendment 1 (2026-10-02, before any dispatch)

Written after the harness was built (commits `74b57701`..`6641a75e`), independently reviewed
adversarially, and fixed. Nothing has been dispatched. This amendment adopts the rules the code
enforces where §1–§6 left them open, and the owner's budget decision.

**Fit outcomes are read from the fork's own note, never inferred** (tightens §2):
- `CACHE_FALLBACK` requires the `ao_cache` note to give a fit-based reason:
  - mixed r2SCAN/B3LYP: tier `'fp32'` with `no FP64 copy: … do not fit` or `… empty grid blocks`;
  - mixed wB97M-V and every `cache` arm: tier `None` with `not built: FP64 copy does not fit` or
    `not built: … empty grid blocks`.
- `NOT_TREATED` requires `did not fit: …; XC stays FP64`.
- `TREATED` requires the expected tier in a clean `built:` note, with nothing appended. A
  `dropped at call` note is not clean.
- **Any other combination is `GATE_FAIL`**, which includes a `predicted` mismatch and tier `None`
  with XC still FP32. That last case replaces the §2 wording "tier `None` with XC still FP32 →
  CACHE_FALLBACK".
- **Gates on fallback cells:**
  - the accuracy and `fp64_tail` gates always apply;
  - the treatment-ran gates apply unless `NOT_TREATED`;
  - the cache-ran gates apply only to `TREATED`;
  - a fallback cell that fails accuracy is `GATE_FAIL` in every mode.

**Observed, not labelled** (adds to §2):
- every non-stock run's `rec['policy']` equals `repr(MixedPrecision(**kwargs))`, built on the pod
  from the real class;
- a stock run has `mf.mixed_precision is None` and no record;
- `ngrids`, `naux` and `nao` are equal across every run of a cell;
- every run except S' (PREREG-4) shares one `init_guess`, which is not `'atom'`, and S' reads back
  `'atom'`;
- the settings of §1 are read back off the built objects, including `conv_tol_grad`, basis and
  auxbasis;
- the ΔE and cycle gates cover the cold pair 0 as well as the warm pairs.

**Stack** (makes §1's "the lock, unchanged" a gate):
- every pin of `scfbench_requirements.lock` and `cutensor-cu12` is read with
  `importlib.metadata` on the pod;
- any mismatch fails the pod, so an unpinned fallback install can never PASS;
- a failed cuTENSOR probe makes the pod FAIL, while its measurement groups still run for
  diagnosis.

**Sentinel and extraction:**
- `STATUS=PASS` requires `WORK_RC=0` and the whole-line `RFCBENCH_PASS`;
- the extractor quotes numbers only from a sentinel with `STATUS=PASS`, `WORK_RC=0`, no pod
  problems and the requested RUN_ID; anything else is printed only under `--unquotable`, with
  every number labelled UNQUOTABLE;
- a cell enters an aggregate only with exactly R warm walls per arm;
- the canary band applies to `pro6000` only.

**Modes and cards:**
- `commission`, `ladder`, `large`, `basis` and `downstream` run on `pro6000` only;
- `cross` runs on `5090`, `l40s`, `h100` and `a100` only;
- `mig` runs on `pro6000mig` only;
- whole-card classes need an exact `nvidia-smi` name and zero MIG devices;
- `commission` takes no filters;
- `mode=large` is refused on the free runner until PREREG-3 carries the six geometry SHA256 values
  in an amendment;
- `mode=mig` is refused until the `pro6000mig` cascade is pinned and PREREG-5 names that exact type
  id in an amendment.

**Budget, the owner's decision of 2026-10-02: calibrate on C0 first.**
- The runner's guessed estimator refuses L12, W1, W2, D2 and X2 today, against
  `WORK_CAP − 300 = 4140 s`.
- C0 is admitted (3876 s est.) and runs first.
- After C0 is banked PASS, `rfcbench_calibrate.py` builds `.github/scripts/rfcbench_calibration.json`
  from its measured walls. That file is committed under a dated amendment written before the next
  dispatch. It refits the power law through the C0 trio, and keeps every estimate at or above the
  measured median × 1.15.
- A pod still refused after calibration is split by `mols` or `xcs` filters into pods whose cells
  are a partition of the pre-registered ones. No cell, arm, R or setting changes.
- Any split that takes the campaign beyond the 18 approved pods (15 + 3 MIG) is put to the owner
  first.
- The calibration file only ever changes admission. It never changes a gate.

**Disclosed limitation.**
- The runner's deadline starts at pod launch, not at container start, as in g4pport, so
  provisioning latency is charged to the run. A slow pod reads as `no result sentinel`.
- `TIMEOUT_S` is 4800 s against the job's 5700 s.

## Amendment 2 (2026-10-02, after C0 and before any further dispatch)

C0 is banked PASS (PREREG-2 RESULT). This amendment governs every later dispatch. It changes
admission, pod count and disclosure only. No gate, band, arm, R or setting changes.

**Calibration** (Amendment 1's rule).
- `.github/scripts/rfcbench_calibration.json` (sha256 `775f4bb015bfec55…`) is `rfcbench_calibrate.py`'s
  output for C0, run `36951733967` attempt 1, `RUN_ID=36951733967-1-0ee605b6`.
- It calibrates `pro6000` only, with margin 1.15. Every other class keeps the guessed estimator.
- Under it, the plan of Amendment 1 still overruns `WORK_CAP − 300 = 4140 s` for every wB97M-V
  pod, for X2 and for M1/M2. wB97M-V costs far more than budgeted (C0's wB97M-V group alone took
  2763 s).

**Owner decision of 2026-10-02: split everything, at R = 3.** The remaining campaign is the 25 pods
of `rfcbench_sets.POD_PLAN`, 26 with C0. Every pod is admitted under the calibration, and per scope
their cells partition the pre-registered cells exactly once; `tests/test_rfcbench.py` pins both.

| scope | pods (mode, gpu: filters) |
|---|---|
| 1 ladder | L12 (r2scan,b3lyp + bridge); W1 wB97M-V: S tier + celecoxib; W2: fluconazole, warfarin, omeprazole; W3: ibuprofen, naproxen, lidocaine, salbutamol, phenytoin; W4: ketoprofen, diphenhydramine, propranolol, atenolol; W5: metoprolol, trimethoprim, diclofenac |
| 4 downstream | D1 (r2scan,b3lyp); D2a wB97M-V: paracetamol, propranolol, celecoxib; D2b: fluconazole, warfarin; D2c: omeprazole |
| 3 large/basis | XL1 (r2scan); XL2 (b3lyp); XL3a sildenafil wB97M-V; XL3b atorvastatin wB97M-V; B1 |
| 2 cross | X1 5090; X2a L40S paracetamol, propranolol; X2b L40S celecoxib; X3 H100; X4 A100 |
| 5 mig | M0 pro6000se; M1a, M2a pro6000mig (r2scan,b3lyp); M1b, M2b pro6000mig (wB97M-V) |

Readings for a split pod are computed over the union of its parts. Parts that ran on different pods
are a cross-pod union, and that is disclosed.

**Extraction.**
- Aggregates are now per (xc, basis). C0's extraction had pooled its def2-SVP cell into the r2SCAN
  mTZVPP geomean; that was a reporting bug, fixed before any reading was banked.

**Disclosure added.**
- Each group records `nvidia-smi` SM/memory clocks, power draw/limit, temperature, pstate and clock
  event reasons, before and after the group. It is disclosed only.
- On C0, gpu4pyscf's stock FP64 VV10 kernel took 8.0 s on the celecoxib certificate inputs, against
  5.1 s on the g4pport pod (run `36942908651`), on the same card model with the same inputs. Nothing
  then recorded why.

## Amendment 3 (2026-10-02, after W4 and before any further dispatch)

**Finding.** Pods W1–W3 idled at 57–92 W before their first group, with SM clocks of 2610–2617 MHz
rising to 2842–2857 MHz under load. Pod W4 (run `36977987637`) already drew **575.91 W** before
our first group, with 97 GB of VRAM free in our view. It ended its group at the 600 W cap and
2197 MHz, and the driver reported no clock-event reason.

W4's wB97M-V mixed/stock ratios are 1.92–2.13, against 2.59–3.25 on W1–W3. Every per-cell spread
stayed small, so §3's NOISY/DEGRADED rule did not fire. C0 shows the same signature: its
celecoxib ratio is 1.95 against W1's 2.59, and its stock VV10 certificate kernel took 8.0 s against
5.1 s. C0 predates the telemetry, so its state is unknown. The cause lies outside our container
(another load on the card, or a faulty power state) and is not named here.

**Owner decision of 2026-10-02: pre-register a rule, and re-run.**

- **CONTENDED.** A pod is CONTENDED when the GPU's `power.draw` before the first group exceeds
  **200 W**. That reading is the first group's `gpu_state_before`, sampled idle after the probe.
  `rfcbench_extract.contention()` implements it.
  - A pod with no readable pre-work power is CONTENTION-UNKNOWN. That covers C0, which predates
    the telemetry.
- **Consequence.** The gates are unchanged, so a CONTENDED pod still PASSes or FAILs on them. Its
  speed readings and aggregates are reported **separately** and never pooled with healthy pods.
  CONTENTION-UNKNOWN is handled the same way.
- **Re-runs.**
  - Each CONTENDED pod is re-dispatched once with the same inputs. A re-run that is CONTENDED
    again is banked as such, and the pair goes to the owner.
  - Re-runs needed now:
    - **W4r**, with W4's inputs;
    - **C0w**, C0's three wB97M-V trio cells with all three arms. The free runner must first let
      `commission` take an `xcs` filter; that is a harness change, merged before C0w, with no
      gate change.
  - Future CONTENDED pods are re-run within a cap of **3**, the default set here because the owner
    named none. A further re-run goes to the owner.
- **Pod count.** 26 + 2 now, so 28, plus at most 3.
- **Not retroactive to gates.** No banked PASS changes. W4's and C0's numbers stay in their
  sentinels and CSVs, labelled.

## RESULT
