# Native-port evidence: the mode as ported into this fork, measured on GPUs

The parent directory's bundle backs the RFC with the **prototype's** measurements (a specialised
engine run side by side with stock GPU4PySCF). This directory backs it with the **fork's own port**,
switched on with `mf.mixed_precision = MixedPrecision(...)`.

- 33 pods across five GPU models: RTX PRO 6000 Workstation, RTX PRO 6000 Server, L40S, H100 and
  A100, plus RTX PRO 6000 MIG 1g.24gb instances.
- Under written protocols with pre-stated predictions.
- **At settings that differ from the library defaults:** the FP64 AO cache on, an unpruned grid,
  B3LYP `xc_switch_tol=3e-4`, and cuTENSOR required. `CLAIMS.md` gives the table. The defaults
  themselves were not measured.

```
python verify_native.py      # stdlib only; exit 0 = every quoted number reproduced from data/
```

| file | what it is |
|---|---|
| `CLAIMS.md` | What the RFC may claim, and what it must not. Includes the measured configuration, negative results, cold start, what can and cannot be verified, and a ledger of every miss, correction and lapse. |
| `verify_native.py` | Recomputes 498 quoted numbers and facts from `data/` at their printed precision, and re-extracts every CSV from its pod's sentinel. It checks `data/SHA256SUMS` first, fails closed, and pins its own check count. A listed known discrepancy must still recompute to its corrected value. |
| `reproduce_native.py` | Re-measures one speed cell on your GPU with the campaign's settings, cuTENSOR preload and cuTENSOR gate. Transcribed from the harness and dry-run-tested only: see its header. |
| `SOURCES.md` | Pod, GitHub run, commit, card, contention and noise flags behind every CSV. |
| `data/*.csv` | Per-run walls, energies, cycles, convergence and cache tiers (one CSV per pod), plus `pods.csv`, `downstream.csv` and `tiers.csv`. |
| `provenance/` | The 33 pod sentinels, the extractor, a harness snapshot and a protocol/run timeline. See its README. |
| `prereg/` | The seven protocols, PREREG-rfcbench-0 to -6, with predictions, gates, dated amendments and RESULTs. |

**How to read the protocols.**
- **Predictions** were written before the runs they predict, with bands. A miss is reported as
  MODEL-MISS and not retuned.
- **Gates** are pass/fail on accuracy and treatment. One pod failed a gate: D1, on one dipole. It
  stands as a FAIL.
- **CONTENDED and DEGRADED pods** are reported separately and never pooled.
  - CONTENDED means another load was on the GPU; DEGRADED means too many NOISY cells.
  - The CONTENDED rule (PREREG-0 Amendment 3) was written **after** its first case (W4) had been
    seen. On whole cards it replaced only low readings (C0, W4); on MIG it replaced two pods whose
    MIG gains were higher than their re-runs'.

**Provenance.**
- `provenance/` ships every pod's result file (sentinel), the extractor, a snapshot of the harness,
  and a timeline of protocol commits against run times.
- `verify_native.py` re-extracts every CSV from its sentinel and checks it, so a reader can follow
  run → CSV → claim without access to the private repository where the runs were orchestrated.
- What remains self-attested (the sentinels' origin, and the order of protocols and runs) is listed
  in `provenance/README.md` and `CLAIMS.md`.
- Corrections found while building and reviewing the package are in the `CLAIMS.md` ledger.
