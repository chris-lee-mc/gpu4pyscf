# PREREG rfcbench-5: throughput on RTX PRO 6000 MIG instances

**Status 2026-10-02: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`.
Freezes at the first `mode=mig` sentinel. Changes go only in dated amendments above RESULT, written
before the dispatch they govern.

**Owner decision of 2026-10-02:** 2–3 extra pods for MIG. The motivation is that an RTX PRO 6000
split into four MIG instances (~24 GB each) may give more molecules per hour on systems that fit.

## The question

For a stream of independent SCFs of this size, how does throughput compare between:
- (a) one whole PRO 6000 running one SCF at a time, and
- (b) the same card as four MIG instances, each running its own SCF?

And how do the mixed-precision mode and its cache behave inside one ~24 GB instance?

## Method

MIG partitions SMs, L2 and memory bandwidth in hardware, so one instance measured alone stands for
each of the four. **Assumption, disclosed:** neighbouring instances on the same card, which may
belong to other tenants on RunPod, do not perturb it.

**MIG gain** of an arm, per cell: 4 × median wall on the whole card (C0, PREREG-2) / median wall on
one instance (this scope). A value above 1 means four instances beat one whole card for that cell.
The comparison crosses pods (C0 against M1/M2), so it is a cross-experiment reading.

**Pod.** `mode=mig`, `gpu=pro6000mig`. Its cascade is the single RunPod MIG type id for the PRO 6000
at one quarter. That id is read by the `mode=gputypes` probe (PREREG-0 §5) and **pinned in a dated
amendment here before the first `mode=mig` dispatch**. If no such type exists, this scope ends with
`n/a-by-availability`, and no pod is spent.

**Card gate:**
- `nvidia-smi -L` shows exactly one MIG device on the PRO 6000;
- the device total memory, as CUDA reports it, is between 20 and 28 GB.

Both are recorded. Anything else means no row.

**Cells.** Arms `mixed`, `stock`; R = 3, mTZVPP / tzvpp-jkfit:
- trio × {r2SCAN, B3LYP};
- {paracetamol, propranolol} × wB97M-V.

The runner's budget check assumes an instance is 4× slower than the whole card.

**Fit.** Tiers are pre-registered outcomes in this mode (PREREG-0 §2). Prediction: paracetamol and
propranolol are `TREATED`, with tier `fp64+fp32` (or `fp64` for wB97M-V). Celecoxib may fall to
`CACHE_FALLBACK` (a 10.3 GiB cache plus the DF tensor in ~24 GB), and stays `mixed`-treated.

## Pods

| pod | purpose |
|---|---|
| M1 | the measurement |
| M2 | the same inputs on a different pod, for pod-to-pod spread |
| (M3) | a reserve, only for an infra failure of M1 or M2 |

## Predictions (MODEL-MISS band ±0.3 absolute on MIG gain)

| | paracetamol | propranolol | celecoxib |
|---|---|---|---|
| MIG gain, stock | 1.4 | 1.2 | 1.1 |
| MIG gain, mixed | 1.3 | 1.2 | 1.0 |

Mixed/stock inside one instance: within ±0.2 of the whole-card ratio for the same cell. Small
molecules underuse a whole card, so the MIG gain is largest for them.

## Readings

- Per cell and arm: MIG gain, read with PAYS / NEUTRAL / HARMS on the gain.
- Per cell: mixed/stock inside the instance, and the fit outcome.
- "Mixed on MIG versus stock on the whole card": 4 × whole-card stock wall / instance mixed wall.
  This is the operator's figure. It is disclosed, as a cross-experiment reading.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## Amendments

None yet.

## Amendment 1 (2026-10-02, before any mig dispatch)

**Type ids.** Read from the `mode=gputypes` sentinel of run `36951667656`
(`ci-results/runs/rfcbench-gputypes-0ee605b64285-36951667656-1-0ee605b6.txt`, `STATUS=PASS`, 49 types
listed). RunPod's only quarter-card PRO 6000 MIG type is on the **Server Edition**:

| class | RunPod type id | memory | cloud |
|---|---|---|---|
| `pro6000mig` | `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb` | 24 GB | secure only |
| `pro6000se` (new) | `NVIDIA RTX PRO 6000 Blackwell Server Edition` | 96 GB | secure, community |

A `2g.48gb` type (two instances per card) also exists. It is not measured in this scope.

**The comparator changes.** §"Method" compared one instance with C0, which runs on the
**Workstation Edition** (`pro6000`). That comparison crosses card models, so it is withdrawn as a
reading. The whole-card comparator is now a **Server Edition** card running the same mig cells:

- **M0**: `mode=mig, gpu=pro6000se`. The same cells, arms and R as M1. The card gate requires exactly
  `NVIDIA RTX PRO 6000 Blackwell Server Edition` and zero MIG devices.
- **MIG gain** of an arm, per cell = 4 × median wall on M0 / median wall on M1 (and on M2, read
  separately; their spread is the pod-to-pod reading).
- The C0 (Workstation) comparison is disclosed only, labelled "cross-model".

**Card gate for `pro6000mig`.** `nvidia-smi --query-gpu=name` prints exactly
`NVIDIA RTX PRO 6000 Blackwell Server Edition`. `nvidia-smi -L` shows exactly one GPU of that exact
model and exactly one MIG device. The CUDA total memory is 20–28 GB. A substring match is no longer
accepted, so the Max-Q and Workstation editions are refused.

**Pods.** M0, M1 and M2. That is three pods, which uses the reserve the owner approved. Any infra
re-dispatch, or any split needed after the C0 calibration, takes the scope past three, and goes to
the owner first. The 4× budget factor for an instance is unchanged until the calibration amendment.

**Prediction for M0 (MODEL-MISS ±0.15).** Mixed/stock on the Server Edition is within ±0.15 of C0's
ratio for the same cell. Same GB202 die; its clocks and power limit may differ, and that difference is
what this prediction tests.

**Harness.** `rfcbench_sets.GPU_CASCADES` pins both ids above, and `MIG_PARENT_MODEL` is the exact
Server Edition string. The free runner refuses `mode=mig` unless this amendment names **both** ids, in
backticks, above RESULT.

## Amendment 2 (2026-10-03, before the next `pro6000mig` dispatch; no MIG pod has existed)

**What happened.** M1a was refused at create time twice, before any pod existed. The second refusal
(run `37094752419`, after #225 logged the whole error) is a request-schema error from RunPod's REST v1
`POST /pods`: its `gpuTypeIds` enum does not list `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG
1g.24gb`. RunPod's GraphQL `gpuTypes` does list it (Amendment 1). So the REST route cannot create
this type at all, and no amount of retrying would change that. No pod was billed.

**Launch route for `pro6000mig` only.** The launcher creates `pro6000mig` pods through GraphQL
`podFindAndDeployOnDemand` (`GQL_DEPLOY_CLASSES` in `runpod_rfcbench.py`). Every other class,
including M0's `pro6000se`, keeps the REST route unchanged, so no banked pod's launch path moves.

- Same image, disk, cloud order (SECURE then COMMUNITY), pod name and env as REST. The bootstrap
  script, which REST takes as `dockerStartCmd`, travels base64 in the env as `BOOT_B64` (as
  `WORK_B64` already does) and a fixed `dockerArgs` command decodes and runs it. The pod therefore
  executes byte-identical work.
- A create is accepted only on HTTP 200, no GraphQL `errors`, and a non-empty pod id. Anything else is
  a refusal and is logged whole on one line, never with the key.
- Delete is verified: a REST `DELETE` must answer 2xx, otherwise GraphQL `podTerminate` must answer
  without errors; after four attempts the job logs `could NOT delete pod`. This applies to every
  class, since a GraphQL-created pod must never be left running because a REST delete silently did
  not apply to it.

**What does not change.** Cells, arms, R, card gate, predictions, readings, budget and the pod count
(M1, M2 with their a/b splits, plus M0 already banked). The measurand is the work script's output;
only the create call differs.

**Closing rule, written now.** If GraphQL also refuses the type on both clouds for M1a, PREREG-5
closes as **n/a-by-availability** for `pro6000mig`: M0 is banked as a Server-Edition data point, and
no MIG gain is claimed or estimated. A capacity refusal ("no instances currently available") is
retried at most twice more, each at least 6 h apart, inside 48 h of this amendment; then the same
closing rule applies. A schema refusal from GraphQL closes it at once.

## RESULT


### RESULT (2026-10-06): **PASS** on every pod. Four MIG instances beat one whole card for r2SCAN and stock B3LYP; mixed B3LYP on celecoxib HARMS; wB97M-V pays only with mixed precision (1.18)

**Pods.** Every sentinel's line 2 matches its run. Every pod passed its card gate:
`NVIDIA RTX PRO 6000 Blackwell Server Edition` with exactly one `MIG 1g.24gb` device, and CUDA total
memory 25.37 GB. The max SM clock is 2430 MHz on every card, and the M0 card is the same model.

| pod | run | xcs | card / slice | idle W | pool |
|---|---|---|---|---|---|
| M0 | `37073279025` | all | `GPU-aa1e28bd`, whole card | 72 | healthy |
| M1a | `37261720977` | r2SCAN, B3LYP | `GPU-f7b038d0` / `MIG-64d10219` | 153 | healthy |
| M1b | `37263100639` | wB97M-V | `GPU-f7b038d0` / `MIG-64d10219` | 159 | healthy |
| M2a | `37265768276` | r2SCAN, B3LYP | `GPU-59992864` / `MIG-0bd56c7f` | 276 | **CONTENDED** |
| M2b | `37267116454` | wB97M-V | `GPU-59992864` / `MIG-0bd56c7f` | 372 | **CONTENDED** |
| M2ar | `37310614922` | r2SCAN, B3LYP | `GPU-59992864` / `MIG-eb448616` | 95 | healthy |
| M2br | `37312821303` | wB97M-V | `GPU-59992864` / `MIG-0bd56c7f` | 98 | healthy |

Sentinels are at `ci-results/runs/rfcbench-<sha12>-<RUN_ID>.txt` on `ci-bus`. CSVs are in
`rfcbench-data/` as `m0`, `m1a`, `m1b`, `m2a`, `m2b`, `m2ar` and `m2br`.

**Launch history.** M1a was refused at create time twice. Neither refusal billed a pod:
- once by the REST schema, which is what Amendment 2 records;
- once by the serialisation gate (run `37261679254`), because the merge's own CPU test run was
  still in progress.

All MIG pods were created through GraphQL and were deleted afterwards.

**Contention.** M2a and M2b are CONTENDED under PREREG-0 Amendment 3, so each was re-run once at the
owner's go-ahead. M2ar and M2br are healthy, and the **M2 readings below are M2ar and M2br**. M2a
and M2b are reported separately and never pooled. They were 1.3–5.1 % *faster* than their healthy
re-runs on the same card. On a MIG slice `nvidia-smi` reports power for the whole card, which
plausibly includes other tenants' slices; this is disclosed, not tested. The rule is unchanged.

**MIG gain**, 4 × M0 median / instance median. Read PAYS ≥ 1.15 and HARMS ≤ 0.95. A `*` marks a
MODEL-MISS, more than 0.3 from the prediction. The prediction table has one column per molecule
and no functional axis, so it is applied to every functional.

| xc | molecule | stock M1 | stock M2 | pred | mixed M1 | mixed M2 | pred |
|---|---|---|---|---|---|---|---|
| r2SCAN | paracetamol | 1.51 PAYS | 1.55 PAYS | 1.4 | 2.00 PAYS* | 1.99 PAYS* | 1.3 |
| r2SCAN | propranolol | 1.42 PAYS | 1.45 PAYS | 1.2 | 1.69 PAYS* | 1.75 PAYS* | 1.2 |
| r2SCAN | celecoxib | 1.35 PAYS | 1.38 PAYS | 1.1 | 1.59 PAYS* | 1.62 PAYS* | 1.0 |
| B3LYP | paracetamol | 1.72 PAYS* | 1.73 PAYS* | 1.4 | 1.98 PAYS* | 1.95 PAYS* | 1.3 |
| B3LYP | propranolol | 1.35 PAYS | 1.38 PAYS | 1.2 | 1.08 NEUTRAL | 1.09 NEUTRAL | 1.2 |
| B3LYP | celecoxib | 1.16 PAYS | 1.19 PAYS | 1.1 | **0.92 HARMS** | **0.93 HARMS** | 1.0 |
| wB97M-V | paracetamol | 1.00 NEUTRAL* | 1.02 NEUTRAL* | 1.4 | 1.19 PAYS | 1.20 PAYS | 1.3 |
| wB97M-V | propranolol | 1.07 NEUTRAL | 1.05 NEUTRAL | 1.2 | 1.18 PAYS | 1.16 PAYS | 1.2 |

Geometric means over the cells, M1 / M2:

| | stock | mixed |
|---|---|---|
| r2SCAN | 1.43 / 1.46 | 1.75 / 1.78 |
| B3LYP | 1.39 / 1.42 | 1.25 / 1.26 |
| wB97M-V | 1.03 / 1.03 | 1.19 / 1.18 |

**Pod-to-pod.** M1 and M2 ran on different physical cards. Instance walls agree to within 3 %
(M2/M1 from 0.971 to 1.023), and every reading has the same label on both pods.

**Mixed/stock inside one instance against M0** (prediction: within ±0.2). It is **MODEL-MISS in all
8 cells, on both pods**, in both directions:

| cell | M0 | M1 | M2 |
|---|---|---|---|
| r2SCAN paracetamol / propranolol / celecoxib | 1.670 / 1.654 / 1.749 | 2.211 / 1.968 / 2.059 | 2.152 / 1.986 / 2.053 |
| B3LYP paracetamol / propranolol / celecoxib | 2.038 / 1.996 / 2.025 | 2.336 / 1.597 / 1.598 | 2.293 / 1.575 / 1.587 |
| wB97M-V paracetamol / propranolol | 3.081 / 3.200 | 3.664 / 3.536 | 3.630 / 3.525 |

On a quarter card, mixed precision helps r2SCAN and wB97M-V *more* than on the whole card. It helps
B3LYP on propranolol and celecoxib *less*: 1.6 against 2.0. This scope does not time the stages,
so it does not say which stage limits B3LYP on the slice.

**Fit.** Every cell is `TREATED` on every pod. r2SCAN and B3LYP use tier `fp64+fp32`, including
celecoxib, which did not fall to `CACHE_FALLBACK`. wB97M-V uses `fp64`. The fit prediction is met.
The cycle counts per cell match M0's (B3LYP K split `fp32*8` then `fp64*5`/`*6`).

**Operator's figure** (disclosed, cross-experiment): 4 × whole-card stock / instance mixed.

| | M1 | M2 |
|---|---|---|
| r2SCAN | 2.78–3.34 | 2.83–3.33 |
| B3LYP | 1.85–4.03 | 1.88–3.98 |
| wB97M-V | 3.66–3.78 | 3.70–3.70 |

**M0 prediction** (mixed/stock on the Server Edition within ±0.15 of the same cell on C0):
- **r2SCAN: in band**, +0.07 to +0.11.
- **wB97M-V: in band**, +0.06 and +0.07, against C0w.
- **B3LYP: MODEL-MISS**, +0.25, +0.34 and +0.40.
  - The B3LYP comparator is C0, which is CONTENTION-UNKNOWN (PREREG-0 Amendment 3).
  - L12's healthy Workstation B3LYP trio (1.859) is closer to M0's 2.019, but still 0.16 below it.
  - So the difference between the cards is not separated from C0's unknown state.

**What this supports, and what it does not.**
- The MIG gain is a projection. It assumes four instances run at once without perturbing each
  other, which is §"Method"'s assumption. Only one instance was ever measured, so the
  assumption is untested here.
- Neighbouring slices did belong to other tenants: the card-wide idle power was 95–372 W against
  M0's 72 W. Yet the instance timings were stable to within 3 % across two cards and three slices.
- For r2SCAN on molecules of this size, four slices give **1.4–1.6× (stock)** and **1.6–2.0×
  (mixed)** the throughput of one whole card.
- For wB97M-V the gain is **≈1.0 (stock, NEUTRAL)** and **1.16–1.20 (mixed, PAYS on both cells and
  both pods)**. Stock wB97M-V already fills most of the card, so four slices only pay once mixed
  precision is on. A sustained ~18 % throughput gain at scale is material.
- For mixed B3LYP the gain **falls with size and reaches HARMS on celecoxib**. That matches the
  large-molecule finding in PREREG-3.
- None of this is quotable for other MIG profiles (`2g.48gb`), other card models, or other basis
  sets.

**Pods spent.** Seven billed pods: M0, M1a, M1b, M2a, M2b, M2ar and M2br. M2ar and M2br are two of
PREREG-0 Amendment 3's three CONTENDED re-runs, and the owner approved them per Amendment 1 here.
The reserve pod M3 was not used.
