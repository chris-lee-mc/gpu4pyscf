# Provenance: from runs to CSVs to claims

**What's here:**
- `sentinels/<csv>.txt`: the result file each pod wrote, one per CSV in `../data/`. Line 2 is the
  pod's `RUN_ID`, and its SHA256 is in `../data/pods.csv` (`sentinel_sha256`).
- `rfcbench_extract.py`: the extractor, stdlib-only, as on `chris-lee-mc/gpu-conformer-engine`
  `main` at `cab1dcc7`.
- `harness/`: a read-only snapshot of the harness at that same commit:
  - the pod measurement code (`rfcbench_cells.py`, `rfcbench_pod.py`);
  - the cell sets and estimates (`rfcbench_sets.py`);
  - the launcher (`runpod_rfcbench.py`), the dispatch workflow, the calibration, and the pinned
    Python lock.
- `TIMELINE.csv`: every protocol commit (PREREG-rfcbench-0 to -6), interleaved with every pod's run
  creation and completion times.

## What `../verify_native.py` checks from these (group P2)

For every pod it checks three things:
- the shipped sentinel's SHA256 matches `pods.csv`, and the sentinel carries the pod's `RUN_ID`;
- re-extracting the per-run CSV from that sentinel with the shipped extractor gives, column for
  column, the CSV in `../data/`;
- the pod's status, contention, idle power, NOISY count, DEGRADED flag and cell count, re-derived
  from the sentinel, match `pods.csv`.

It also re-derives `../data/downstream.csv` from the D1, D2 and C0 sentinels. So a reader can
check the whole chain sentinel → CSV → claim without access to the private repository.

## What is still self-attested

- **The sentinels' origin.** The pods wrote them to a branch of a private repository. Nothing here
  proves that a given file came from a GPU run rather than being written by hand. The shipped files
  are what the RESULTs were computed from, and the internal consistency checks above are what can
  be offered.
- **`TIMELINE.csv`.**
  - Protocol times are git committer times, which the author controls.
  - Run times come from the GitHub Actions API of a private repository.
  - So the claim that every protocol and amendment preceded the runs it governs cannot be verified
    independently. Two amendments preceded their dispatches by only 17 s and 87 s.
- **The harness snapshot** is the code as of `cab1dcc7`. Each pod ran at the earlier commit listed
  in `pods.csv` (`repo_sha`). Between those commits the measurement path changed only by adding the
  DF-placement recorder and the extractor's per-basis grouping. The snapshot shows the method; it
  is not the exact bytes every pod ran.
