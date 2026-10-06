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
| `verify_native.py` | Recomputes 385 quoted numbers and facts from `data/` at their printed precision. It fails closed, and a listed known discrepancy must still recompute to its corrected value. |
| `reproduce_native.py` | Re-measures one speed cell on your GPU with the campaign's settings, cuTENSOR preload and cuTENSOR gate. Transcribed from the harness and dry-run-tested only: see its header. |
| `SOURCES.md` | Pod, GitHub run, commit, card, contention and noise flags behind every CSV. |
| `data/*.csv` | Per-run walls, energies, cycles, convergence and cache tiers (one CSV per pod), plus `pods.csv`, `downstream.csv` and `tiers.csv`. |
| `prereg/` | The seven protocols, PREREG-rfcbench-0 to -6, with predictions, gates, dated amendments and RESULTs. |

**How to read the protocols.**
- **Predictions** were written before the runs they predict, with bands. A miss is reported as
  MODEL-MISS and not retuned.
- **Gates** are pass/fail on accuracy and treatment. One pod failed a gate: D1, on one dipole. It
  stands as a FAIL.
- **CONTENDED and DEGRADED pods** are reported separately and never pooled.
  - CONTENDED means another load was on the GPU; DEGRADED means too many NOISY cells.
  - The CONTENDED rule (PREREG-0 Amendment 3) was written **after** its first case (W4) had been
    seen, and it has so far replaced only low readings.

**Limits on verification.**
- The sentinels, the measuring harness and the protocol history live in a private repository.
- From this package alone, a reader can check claims against CSVs, but not CSVs against sentinels,
  nor the order of protocols and runs. `CLAIMS.md` spells this out.
- Corrections found while building and reviewing the package are listed in its ledger. Those
  include miscounts and dates in the RESULT texts, and two numbers rounded one unit off.
