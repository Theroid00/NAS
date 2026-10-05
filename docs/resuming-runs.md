# Resume interrupted searches

New experiment references use record-relative paths (`path_format: 1`). Copy the
complete tree and resume from its new metadata/comparison location; see
[portable delivery](portable-delivery.md). Path relocation preserves the source,
dependency, dataset, and replay checks and does not bypass recovery restrictions.

Available on branch `resume/resume-trial`. New GA, aging evolution, and random
search runs save resumable metadata automatically. The metadata path is printed
at startup; each comparison also saves `comparison.json` in its output directory.

From the repository root, resume an individual search:

```powershell
.\.venv\Scripts\python.exe main.py --resume "experiments/tabular/generation_logs/metadata_<run-id>.json"
```

Resume a whole comparison suite:

```powershell
.\.venv\Scripts\python.exe run_comparison.py --resume "experiments/tabular/comparisons/<comparison-id>/comparison.json"
```

Replace the placeholders with the paths printed by the original run. Resume
loads its original method, dataset, device, seeds, budget, and training settings.
Other CLI settings are not applied when `--resume` is supplied. Keep the same
checkout location, dataset cache, and dependency environment.

## What is preserved

- Completed trials are read from the durable journal, without training again.
- The private search RNG restarts at its original seed. Replaying archived
  scores reconstructs GA populations, adaptive mutation, aging order, and random
  candidate batches before evaluating missing trials.
- The original run ID, historical winner, trial seeds, scores, and artifacts
  remain associated with the same run. Completed generation timings are reused
  when rebuilding the generation log.
- Comparison resume skips searches and final-training seeds already recorded.
  It also recovers a completed search or final-training result if the process
  stopped before updating the comparison manifest.
- Failed attempts remain in the journal. They can repeat a trial ID; completed
  trial IDs remain sequential and unique. Failure attempts do not consume a
  completed-candidate budget, but their recorded duration is included in
  evaluation time.

## Failure handling

Every complete trial is flushed and synced before metadata is atomically
replaced. The trial journal is the source of truth if a process dies between
those writes. Metadata, winners, comparison manifests, and final-training result
JSON use atomic replacement.

An incomplete final journal write is saved as a `.partial.<id>` file before the
complete prefix is recovered. Corruption elsewhere is rejected. OS run locks
prevent simultaneous writers and release automatically when a process dies;
the small `.lock` files intentionally remain on disk.

Resume checks format compatibility, configuration, dependency versions, dataset
provenance, subset fingerprints, and replayed candidate decisions. A divergent
replay stops rather than combining incompatible results. Resume records the
current Git revision and dependency versions alongside original provenance.
Changes to controller semantics must bump `RESUME_VERSION`.

## Boundaries

This is **trial-level resume**, not epoch-level resume. A search trial interrupted
during training restarts with the same seed. Interrupted final retraining also
restarts; completed final retraining is reused. Optimizer/scheduler state and
partially trained weights are not restored.

Old experiments created before this feature are kept intact but cannot resume.
Current resumable searches use metadata version 3, including startup source fingerprints.
Comparison recovery uses manifest version 2. Earlier search versions 1/2 and
comparison version 1 remain readable but require their original implementation
for recovery; completed old comparisons can still be packaged with `--comparison`.
New recovery checks source hashes before writing records or reusing trials, so
changed training/controller code cannot be mixed into the same experiment.
No automatic retry loop or background service is introduced: fix a fatal error
if needed, then invoke `--resume`.

Reported elapsed time excludes the gap between invocations. A hard process kill
can lose the unrecorded duration of its in-flight trial. GPU reproducibility is
verified in the current environment; different hardware need not yield identical
floating-point results.

## Verification

Tests compare resumed and uninterrupted candidate sequences, scores, seeds, and
winners for all three methods after interruptions inside initial populations,
later generations, and random-search batches. Additional coverage includes
repeated failures, completed-run replay, malformed journal tails, corruption,
configuration changes, dataset changes, active-run locking, abrupt process death,
comparison recovery, and the final-result/manifest write gap. A CUDA test trains
real Breast Cancer MLPs and checks that only unfinished trials run after resume.

At the initial resume milestone on the RTX 4060 environment, all 39 tests passed, including the CUDA
test. Both `main.py --resume` and `run_comparison.py --resume` also passed
end-to-end CLI checks using explicitly marked smoke artifacts.

The later cleanup passes 55 tests, including recovery and CUDA artifact parity;
see [validation and results](validation-and-results.md).
