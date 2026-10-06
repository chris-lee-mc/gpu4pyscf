# Native-port evidence: the mode as ported into this fork, measured on GPUs

The parent directory's bundle backs the RFC with the **prototype's** measurements (a specialised
engine run side by side with stock GPU4PySCF). This directory backs it with the **fork's own port**,
measured as an ordinary user would run it:

```python
mf.mixed_precision = MixedPrecision(...)
```

It covers 33 pods across four GPU models and a MIG profile, under pre-registered protocols.

```
python verify_native.py      # stdlib only; exit 0 = every quoted number reproduced from data/
```

| file | what it is |
|---|---|
| `CLAIMS.md` | What the RFC may claim: positive results, negative results, gaps. Each claim is tagged with the `verify_native.py` checks that reproduce it. |
| `verify_native.py` | Recomputes 326 numbers and facts from `data/` at their printed precision; fails closed. |
| `reproduce_native.py` | Re-measures one speed cell on your GPU with the campaign's exact settings. Transcribed from the harness and dry-run-tested only: see its header. |
| `SOURCES.md` | Pod, GitHub run, commit, card, contention and noise flags behind every CSV. |
| `data/*.csv` | Per-run walls, energies, cycles and cache tiers (one CSV per pod), plus `pods.csv`, `downstream.csv` and `tiers.csv`. |
| `prereg/` | The seven protocols, PREREG-rfcbench-0 to -6. Each has its predictions, gates, dated amendments (all written before the dispatch they govern) and RESULT. |

**How to read the protocols.**
- Predictions were written before any data, with bands. A miss is reported as MODEL-MISS, never
  retuned.
- **Gates** are pass/fail on accuracy and treatment. One pod failed a gate: D1, on one dipole. It
  stands as a FAIL.
- **CONTENDED pods** (another load on the GPU) and **DEGRADED pods** (too many NOISY cells) are
  reported separately, never pooled, under rules written before they occurred.

**Corrections found while building this package.** Two RESULT texts round a number one unit off in
its last digit. `verify_native.py` lists both as known discrepancies, and `CLAIMS.md` quotes the
recomputed value.
