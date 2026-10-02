# Current state, cleanup, and maintenance

Updated: 2026-10-03. Branch: `resume/resume-trial`.

Code cleanup commit: `d103990` (`Remove obsolete image paths and consolidate winner retraining`).

## User's current scope

The priority is a clean, understandable codebase with detailed documentation.
The dashboard was previously added beyond the user's desired core emphasis;
the user has now deferred dashboard and report-generation work. Those components
are kept intact, but no new presentation/report features are added in this cleanup.
The main project is genetic architecture search with aging/random comparisons,
not dashboard development. The industry goal remains; research is parked.

## Removed redundancy

| Removed path / interface | Why it was removed | Current replacement |
| --- | --- | --- |
| `data/cifar.py` | Historical image loader unused by the tabular pipeline | `data/tabular.py` |
| `models/builder.py` | Historical CNN builder unused by maintained training | `models/mlp.py` |
| `models/baselines/resnet.py` | Historical image baseline unused by the current suite | Optional tabular MLP helper stays separate |
| `requirements-cifar.txt` | Only supported the removed image path | Core/GPU dependencies for tabular training |
| `train_random_best.py` | Identical forwarding alias for the generic winner trainer | `evaluate_best.py` works for every searched method |
| `evaluate_architecture` | Unused scalar-only wrapper discarded trial diagnostics | `evaluate_trial` returns the complete outcome |
| `random_search` tuple wrapper | Unused legacy wrapper around the persisted-result API | `run_random_search` returns a consistent winner record |
| `seed_worker` | Only used by historical CIFAR workers | Tabular loaders use zero workers |
| Stale CIFAR/data/proxy constants in `training/config.py` | No maintained imports; duplicated/conflicting defaults | `data/specs.py` and explicit CLI settings |

Deleted code is recoverable in Git history before this cleanup. Historical image
artifacts and earlier benchmark records are not deleted. Restore the appropriate
earlier checkout to study the original CIFAR training pipeline; mixing its schema
with current schema 3 is deliberately rejected.

## Simplified maintained code

- Added `training.trainer.train_winner` so the two winner-retraining CLIs share
  validation, saved-dataset selection, split handling, and full-training delegation.
- Made `main.py`, `run_comparison.py`, `evaluate_best.py`, calibration, ablation,
  and sweep CLIs default to CUDA. Explicit CPU mode remains available. Portfolio
  and serving already defaulted to CUDA.
- Centralized library-default dimensions in lightweight `data/specs.py`.
  `models/mlp.py` no longer imports the dataset loader just to obtain dimensions.
- Removed redundant aliases of the same model variable around checkpoint save/load.
- Corrected obsolete CPU-only export wording and reorganized reading guidance.

Search-space schema, candidate ordering, search RNG behavior, proxy schedule,
fitness, tie policies, optimizer settings, checkpoint keys, and measured benchmark
results are preserved. No candidate-score cache or duplicate-architecture removal
was introduced: that would change budgets, randomness, and experiment meaning.

## Distinct functionality intentionally retained

The fixed tabular MLP is a small tested optional control, not a participant in the
main suite. Calibration, ablations, hyperparameter sweeps, independent score audit,
and optional NATS lookup answer different questions and therefore are retained.
Plotting and replay utilities inspect saved data. Inference export and API form
an optional delivery path. Empty package initializers are import boundaries.
This cleanup avoids deleting distinct working tools merely because they are not
part of the shortest run command.

## Known gaps and deferred work

| Area | Current behavior / gap | Sensible next change if needed |
| --- | --- | --- |
| Export recovery | Destination is created before export finishes; interruption can block retry or leave an example marked available but missing | Validate a staged directory and publish it atomically; check promised files on startup |
| Report regeneration | Plain report output can replace portfolio enrichment at the same default path | Separate plain/enriched outputs or preserve explicitly compatible enrichment; deferred |
| Dashboard | No selected-model test/inference summary or uncertainty display; only first full-training repeat is inspected | Presentation polish if the user chooses to resume dashboard work; deferred |
| Fresh-machine delivery | Local environment is verified; no clean-install rehearsal or downloadable weights release | Lock GPU dependencies and verify a clean setup before distributing |
| Docker | Existing optional image is CPU-only and unbuilt locally | Build/verify only if container delivery becomes a requirement |
| Source provenance | Packaging hashes are historical; startup records Git revision but not a complete dirty-worktree source snapshot | Capture source fingerprints at the beginning of future searches |
| Resume | Completed trials recover, partial epochs do not; code changes can affect later evaluations | Keep the training implementation stable during a run; epoch recovery is a separate feature |
| Paths | Comparison/result files contain original local paths | Add relocation support if distributing raw experiments becomes necessary |
| Objective | Only validation accuracy drives search | Declare and implement a new shared objective before rerunning all methods |
| Dataset ingestion | Two named classification datasets; no generic upload/CSV schema | Add ingestion only when a specific dataset/user flow requires it |
| Broader claims | Five seeds, one main dataset, one retraining seed per winner | More datasets/seeds/controls only if a broader claim is needed |
| Production serving | Local API, no auth or production monitoring; sequential warm latency only | Treat deployment hardening as a separate scoped task |

The export and plain-report failure cases were reproduced in temporary files during
the preceding audit. They do not alter the measured search results. Existing
`analysis/` content is untracked user/local material and is left untouched.

## How to make future changes safely

1. Identify whether a change affects search decisions, fitness, training, or only
   packaging. Numerical changes require a new experiment, not edited old results.
2. Preserve schema/version boundaries when changing chromosome meaning or journal
   requirements. Do not silently reinterpret old gene arrays or force resume.
3. Keep scoring in the shared evaluator and objective selection explicit.
4. Use focused tests for behavior changes, then the full offline suite. Check CUDA
   with the optional exported-model test where relevant.
5. Keep new experiment files ignored; commit selected documented result snapshots
   with their real protocol and limitations.
6. Update these guides, the command reference, and the cleanup log when removing or
   renaming a public entry point. Use Git commits to preserve reviewable milestones.

## Cleanup verification

Verification covers the existing tests plus shared-retraining/default-device tests,
optional GPU artifact parity, an offline smoke comparison, documented command help,
documentation links, obsolete-reference searches, and Git whitespace checks.
After cleanup, **all 55 tests passed**, including CUDA checkpoint/API parity against
the existing exported Covertype model. Dependency checking found no broken installed
requirements. No full benchmark or final test-set evaluation was rerun for cleanup.
