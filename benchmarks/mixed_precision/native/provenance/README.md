# Provenance: from runs to CSVs to claims

**What's here:**
- `sentinels/<csv>.txt`: the result file each pod wrote, one per CSV in `../data/`. Line 2 is the
  pod's `RUN_ID`, and its SHA256 is in `../data/pods.csv` (`sentinel_sha256`).
- `rfcbench_extract.py`: the extractor, stdlib-only, as on `chris-lee-mc/gpu-conformer-engine`
  at `e1cc2bb5`, the commit the PREREG-9 pods ran. It writes the extra-mode, PREREG-8 and PREREG-9
  (`defaults`) CSVs too, and re-extracts every earlier CSV unchanged (group P2 proves this on all
  46). `harness/rfcbench_extract.py` is the older copy at `5013b101`, kept with its snapshot.
- `harness/`: a read-only snapshot of the harness at `5013b101`, the commit the PREREG-8 pods ran:
  - the pod measurement code (`rfcbench_cells.py`, `rfcbench_pod.py`, and `rfcbench_extra.py` for
    PREREG-7's cold-start, stage-timing and concurrency modes);
  - the cell sets and estimates (`rfcbench_sets.py`);
  - the launcher (`runpod_rfcbench.py`), the dispatch workflow, the calibration, and the pinned
    Python lock.
- `TIMELINE.csv`: every protocol commit (PREREG-rfcbench-0 to -9), interleaved with every pod's run
  creation and completion times. It was missing from the package's first version (an ignore rule
  kept it out of git), and has been regenerated from the same sources; the PREREG-9 update
  regenerated it again (same generator, same sources) and reproduced every earlier row.

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
- **The harness snapshot** is the code as of `5013b101`, which every PREREG-8 pod ran (PREREG-7's
  pods ran `d05888b8`). Earlier pods
  ran at the commits listed in `pods.csv` (`repo_sha`). Between those commits the speed-cell path
  changed only by adding the DF-placement recorder, the extractor's per-basis grouping, and PREREG-7's
  optional stage hook, which is off for speed cells. The snapshot shows the method; it is not the
  exact bytes every earlier pod ran. The PREREG-9 pods (LD1, LD2) ran a later commit, `e1cc2bb5`,
  which adds the `defaults` mode; that harness is not snapshotted here (only its extractor is), so
  the five-arm, two-grid measurement path is described by `../prereg/PREREG-rfcbench-9-library-defaults.md`
  and attested by what the sentinels record (settings read back off `mf` on every arm).
