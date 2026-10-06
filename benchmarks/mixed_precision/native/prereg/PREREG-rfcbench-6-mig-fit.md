# PREREG rfcbench-6: where one RTX PRO 6000 MIG instance stops holding the treated path

**Status 2026-10-06: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`.
Freezes at the first `mode=migfit` sentinel. Changes go only in dated amendments above RESULT, written
before the dispatch they govern.

**Owner decision of 2026-10-06:** test the slice fit limits predicted after PREREG-5, with one slice
pod and one whole Server Edition card on sildenafil, imatinib and atorvastatin. Section "Pods" explains
why the slice side is split into two pods.

## The question

PREREG-5 measured one `MIG 1g.24gb` instance up to celecoxib (381 Da, 612 basis functions). Every cell
was `TREATED`. Above that size, two memory rules decide what an instance can still hold:

1. **The fork's AO cache** (`mixed_precision._build_ao_cache`, fork `63af0568`). The tier is chosen
   from the predicted cache size, against a budget of `0.7 ×` free device memory:
   - `fp64+fp32` needs `1.5 × bytes64`;
   - `fp32` needs `0.5 × bytes64`;
   - otherwise the mixed arm gets no cache at all.

   For a hybrid (B3LYP), the DF tensor is built first (`cderi_prebuilt`), so the budget is taken
   after it. For r2SCAN, the cache is built first.
2. **gpu4pyscf's DF tensor placement** (`df/df.py`, v1.8.1). The tensor stays on the device only if
   `naux × npairs × 8 < 0.4 ×` available memory. Otherwise it is kept in host memory and streamed
   to the GPU at every J/K build.

Does one instance hold the treated path for 470–560 Da molecules? And does an instance still beat a
quarter of the whole card once it cannot?

## Method

**Cells.** `mode=migfit`: {sildenafil, imatinib, atorvastatin} × {r2SCAN, B3LYP} at mTZVPP /
tzvpp-jkfit. Arms are `mixed` then `stock` (stock last). R = 3, plus the cold pair, as in every
rfcbench mode. The fit is a pre-registered outcome here: `CACHE_FALLBACK` and `NOT_TREATED` are
results, not failures (`FALLBACK_OUTCOME_MODES`).

wB97M-V is out of scope. Sildenafil's stock wall alone is about 200 s on the whole card, so roughly
800 s on an instance, and R = 3 plus the cold pair cannot finish inside one instance pod's work cap.

**Geometries.** These are the PREREG-3 LARGE geometries, checked by the pod before any SCF against
`rfcbench_sets.LARGE_SHA256`:

| molecule | MW (Da) | basis functions | naux | SHA256 |
|---|---|---|---|---|
| sildenafil | 474.6 | 835 | 3088 | `13e1f2a31ab3ed2a19b28407f378995c20bbd7dfa98391e277c16a2d07e92160` |
| imatinib | 493.6 | 898 | 3349 | `8954647d0f2a6b10c622519b326ee7a714be8cae0413bd4d88116027483155ac` |
| atorvastatin | 558.6 | 994 | 3721 | `3de1abd52b70e63f904dc9990b0e6db2caa0a7d0e864f96a2ea9a039750e2368` |

**Cards.**
- `pro6000mig` = `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb`. It is created through
  GraphQL (PREREG-5 Amendment 2) and gated as in PREREG-5 Amendment 1.
- `pro6000se` = `NVIDIA RTX PRO 6000 Blackwell Server Edition`, the whole-card comparator. The
  Workstation XL1/XL2 runs (PREREG-3) are disclosed only, because the card models differ.

**New recording, not gated.** For every run, the pod records where the DF tensor ended up:
- `settings.cderi.types`, the type of every stored block;
- `settings.cderi.nbytes`, the exact size after pair screening.

The extractor writes them as `cderi_types` and `cderi_bytes`. **The tensor counts as on the device
only if every block is a `cupy` ndarray; anything else counts as host.**

**Readings.**
- **MIG gain** per cell and arm: 4 × median wall on F0 / median wall on the instance. Read PAYS ≥ 1.15
  and HARMS ≤ 0.95.
- **Fit outcome** of the mixed arm on each pod, read against the prediction below.
- **DF placement** of every arm on each pod, with its measured bytes, set against the 0.4 rule.
- **Mixed/stock inside the instance**, disclosed.
- **Operator's figure** (4 × F0 stock / instance mixed), disclosed.

## Pods

| pod | class | cells | over-estimate (C0 calibration) |
|---|---|---|---|
| F0 | `pro6000se` | all 6 | 2152 s |
| F1a | `pro6000mig` | r2SCAN × 3 | 3195 s |
| F1b | `pro6000mig` | B3LYP × 3 | 3673 s |

The limit is `WORK_CAP − 300 = 4140 s`. Both functionals on one instance come to about 5600 s, so the
instance side is split by functional. **That makes 3 pods, one more than the owner's "one slice pod
plus one whole card".** The extra pod is the cost of fitting the time cap, and it adds no cells.

There is no repeat instance pod. PREREG-5 measured the pod-to-pod spread of instance walls at under 3 %
across two cards and three slices; that is disclosed, not re-measured.

**Over-run rule.** Host-resident DF streaming is not in the estimator. If F1a or F1b hits the work
cap, it is banked as a FAIL and its finished cells are reported, labelled. That outcome is itself a
reading: "does not complete R = 3 within one instance pod's budget". Any re-dispatch, including a
split by molecule, goes to the owner first.

**Contention.** PREREG-0 Amendment 3 applies. A CONTENDED pod is re-run once, and only after the
owner says so (as in PREREG-5 Amendment 1).

## Predictions

### Fit and placement

Predicted from the two rules, using the measured `bytes64` (PREREG-3 runs) and an upper bound on the
DF tensor of `naux × nao(nao+1)/2 × 8`. The instance has 24.98 GB free.

| cell | DF tensor (upper bound) | DF on the instance | mixed fit on the instance | on F0 |
|---|---|---|---|---|
| r2SCAN sildenafil | 8.62 GB | host | `CACHE_FALLBACK` (`fp32`) | `TREATED`, DF on device |
| r2SCAN imatinib | 10.81 GB | host | `CACHE_FALLBACK` (`fp32`) | `TREATED`, DF on device |
| r2SCAN atorvastatin | 14.72 GB | host | `CACHE_FALLBACK` (`fp32`) | `TREATED`, DF on device |
| B3LYP sildenafil | 8.62 GB | device | `CACHE_FALLBACK` (`fp32`) | `TREATED`, DF on device |
| B3LYP imatinib | 10.81 GB | **no prediction** | `CACHE_FALLBACK` (`fp32`) | `TREATED`, DF on device |
| B3LYP atorvastatin | 14.72 GB | host | `CACHE_FALLBACK` (`fp32`) | `TREATED`, DF on device |

Low-confidence entries, stated now:
- **r2SCAN sildenafil's placement.** The cache takes about 6.5 GB first, which leaves a device
  threshold near 7.4 GB against the 8.62 GB upper bound. Pair screening may bring the tensor under it.
- **r2SCAN imatinib's tier.** `1.5 × bytes64` misses the budget by about 1 %.
- **B3LYP imatinib's placement.** The 10.81 GB upper bound against a ~10.0 GB threshold is decided by
  screening, so there is no prediction.
- **B3LYP atorvastatin's tier depends on its placement.** On the device it would be `NOT_TREATED`; in
  host memory it is `fp32`.

**Falsifiers.**
- **Any of the six mixed instance cells reading `TREATED` falsifies the limit estimate** given after
  PREREG-5 (full fast path to ~400 Da for B3LYP and ~450 Da for r2SCAN), as too low.
- Any F0 cell not `TREATED` falsifies the whole-card side.

### MIG gain (MODEL-MISS band ±0.3 absolute)

| | sildenafil | imatinib | atorvastatin |
|---|---|---|---|
| r2SCAN, stock | 1.2 | 1.2 | 1.1 |
| r2SCAN, mixed | 1.3 | 1.3 | 1.1 |
| B3LYP, stock | 1.0 | 1.0 | 0.9 |
| B3LYP, mixed | 0.8 | 0.8 | 0.7 |

These extrapolate PREREG-5's trends with size and allow for host-resident DF and the `fp32`-only cache.

**Falsifier of the claim that "the whole card wins past ~500 Da".** If atorvastatin's stock MIG gain
reads PAYS (≥ 1.15) for both functionals, the claim is falsified.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## Amendments

None yet.

## Amendment 1 (2026-10-05, after F0 and before the F1a dispatch)

**A wrong premise in the r2SCAN placement predictions.** gpu4pyscf v1.8.1 never builds the DF tensor for a
non-hybrid functional. `DF.get_jk` with `with_k=False` and `_cderi is None` calls `df_jk.get_j`, which
computes J with the direct three-centre engine (`j_engine_3c2e`) and builds no `_cderi`. F0 (run
`37379456085`) recorded exactly that: every r2SCAN run has `cderi.types = []` and `nbytes = 0`, on
both arms. I missed this when reading the source before pre-registering.

- **Withdrawn:** the "DF on the instance" entries for the three r2SCAN cells (all "host") and the
  low-confidence note on r2SCAN sildenafil's placement. They predicted the placement of a tensor
  that is never built. They are counted as **wrong predictions of my model**, not as untested, and
  the RESULT will say so.
- **Not changed:**
  - every r2SCAN fit prediction (`CACHE_FALLBACK`, `fp32`). Its rule (`1.5 × bytes64` against
    `0.7 ×` free memory) never involved the DF tensor;
  - every B3LYP prediction, including placement. F0 recorded the B3LYP tensors' screened sizes at
    5.55 / 5.23 / 8.62 GB, against the upper bounds of 8.62 / 10.81 / 14.72 GB. That is disclosed
    here and does **not** revise any B3LYP prediction;
  - the MIG-gain table, the falsifiers, the cells and the pods.
- **Reading for r2SCAN placement:** "no DF tensor", from the recorder. For r2SCAN, the instance's
  J cost is the direct engine's on both cards.

## RESULT


### RESULT (2026-10-06): **PASS** on all three pods. One instance falls back to the FP32-only cache for every cell from 475 Da up, and four instances still match or beat one whole card

**Pods.** Every sentinel's line 2 matches its run, and no pod is CONTENDED.

| pod | run | card / slice | idle W | cells |
|---|---|---|---|---|
| F0 | `37379456085` | `GPU-5c1fcceb`, whole Server Edition card | 66 | 6/6 OK, all `TREATED` |
| F1a | `37382762854` | `GPU-f7b038d0` / `MIG-64d10219`, 25.37 GB | 93 | 3/3, all `CACHE_FALLBACK` |
| F1b | `37386887345` | `GPU-59992864` / `MIG-661b6682`, 25.37 GB | 94 | 3/3, all `CACHE_FALLBACK` |

- F0's first two dispatches (runs `37369605244`, `37371256270`) never got a GitHub-hosted runner, so no pod was created and nothing was billed. The third attempt ran.
- No cell is NOISY; the largest spread is 3.0 %. Every arm converged, in the same cycle counts on both cards (14–17).
- CSVs: `rfcbench-data/f0.csv`, `f1a.csv`, `f1b.csv`. They include `cderi_types` and `cderi_bytes`.

**Fit: the prediction is met in all 12 cells.**
- All six instance cells are `CACHE_FALLBACK`, at tier `fp32`, with the fork's note
  `no FP64 copy: FP64 copy and FP32 mirror do not fit`.
- All six F0 cells are `TREATED`, at `fp64+fp32`.
- **Neither falsifier fired.** Together with PREREG-5, where celecoxib (381 Da, 612 basis functions)
  was `TREATED`, one instance keeps the full fast path somewhere **between 381 and 475 Da** at mTZVPP,
  for both functionals. From 475 Da up it runs with the FP32 cache only.
- r2SCAN imatinib's tier was the low-confidence call, about 1 % from the budget, and it landed as
  predicted.

**DF placement.**
- **r2SCAN:** no DF tensor exists on either card (Amendment 1). The three withdrawn placement
  predictions are counted as **wrong**.
- **B3LYP:** the tensor stays on the device in every run, on both cards. Its screened size is
  identical on both cards, and well under the upper bounds that drove the predictions:

  | | screened size | upper bound | predicted placement | outcome |
  |---|---|---|---|---|
  | sildenafil | 5.55 GB | 8.62 GB | device | device, **met** |
  | imatinib | 5.23 GB | 10.81 GB | no prediction | device |
  | atorvastatin | 8.62 GB | 14.72 GB | host | device, **missed** |

- The conditional note that atorvastatin "on the device would be `NOT_TREATED`" was also wrong.
  With the true 8.62 GB, the rule gives `0.7 × (24.98 − 8.62) = 11.45 GB ≥ 8.66 GB`, which is
  tier `fp32`. That is the tier observed. The rule itself held; my size bound was the error.
- So none of the six instance cells streamed a DF tensor from host memory. The "past ~500 Da the
  DF tensor moves to host RAM" estimate given after PREREG-5 was **wrong** for these molecules.

**MIG gain**, 4 × F0 median / instance median. A `*` marks a MODEL-MISS (±0.3).

| | sildenafil (475 Da) | imatinib (494 Da) | atorvastatin (559 Da) |
|---|---|---|---|
| r2SCAN, stock | 1.28 PAYS (pred 1.2) | 1.28 PAYS (pred 1.2) | 1.19 PAYS (pred 1.1) |
| r2SCAN, mixed | 1.42 PAYS (pred 1.3) | 1.44 PAYS (pred 1.3) | 1.29 PAYS (pred 1.1) |
| B3LYP, stock | 1.16 PAYS (pred 1.0) | 1.09 NEUTRAL (pred 1.0) | 1.09 NEUTRAL (pred 0.9) |
| B3LYP, mixed | 1.11 NEUTRAL* (pred 0.8) | 1.08 NEUTRAL (pred 0.8) | 0.98 NEUTRAL (pred 0.7) |

11 of 12 are in band. Every miss and every near-miss is on the high side: the instance did better
than predicted.

**The falsifier for "the whole card wins past ~500 Da"** required atorvastatin's stock gain to read
PAYS for both functionals. r2SCAN reads 1.19 PAYS and B3LYP 1.09 NEUTRAL, so it **did not fire**. But
the claim is not supported either:
- no cell reads HARMS;
- the lowest reading is 0.98 (B3LYP mixed, atorvastatin), i.e. break-even;
- the whole card never won.

**Mixed/stock inside the instance** (disclosed):
- r2SCAN: 2.20 / 2.00 / 2.08, against F0's 1.99 / 1.78 / 1.93. Mixed precision still pays more on
  an instance, even with the FP32-only cache.
- B3LYP: 1.32 / 1.42 / 1.34, against F0's 1.38 / 1.44 / 1.50.

**Operator's figure**, 4 × F0 stock / instance mixed (disclosed): r2SCAN 2.83 / 2.55 / 2.48, B3LYP
1.54 / 1.55 / 1.47.

**Disclosed, not explained.**
- B3LYP's mixed MIG gain is 1.08–1.11 here, against 0.92 for celecoxib in PREREG-5.
- On the whole card, B3LYP mixed/stock is 1.38–1.50 at these sizes, against 2.0 for the trio on M0.
  So the whole card's mixed advantage itself shrinks with size.
- No stage was timed, so this scope does not say why.

**What this supports.**
- At mTZVPP, one 1g.24gb instance runs drug-like molecules up to at least 559 Da with the DF tensor
  on the device.
- It loses the FP64 cache tier somewhere between 381 and 475 Da.
- Four instances deliver:
  - r2SCAN: 1.2–1.3× the throughput of one whole Server Edition card (stock), and 1.3–1.4× (mixed);
  - B3LYP: about break-even to 1.16×.
- The four-instances-at-once assumption of PREREG-5 §"Method" still applies and is still untested.
- Not quotable for wB97M-V, which was out of scope, nor for other bases or MIG profiles.

**Pods spent.** Three billed pods: F0, F1a and F1b. That is one more than "one slice pod plus one
whole card", as §"Pods" stated. The two runner failures cost nothing.
