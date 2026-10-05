# Project documentation

Updated: 2026-10-05. Start here when returning to the codebase.

The project searches for the best observed MLP architecture for a tabular dataset
within a declared candidate-evaluation budget. It compares generational genetic
search, aging evolution, and random search. Validation accuracy is the implemented
search objective. The current goal is a readable AI/ML engineering portfolio.

## Reading order

| Guide | What it explains |
| --- | --- |
| [Project overview](overview.md) | Problem, terminology, full workflow, scope, and design decisions |
| [Search algorithms](search-algorithms.md) | Chromosome, model construction, GA, aging, random search, fairness, ties, and costs |
| [Data and training](data-and-training.md) | Datasets, splits, preprocessing, proxy evaluation, retraining, seeds, and metric formulas |
| [Codebase map](codebase-map.md) | Responsibilities and important functions of every maintained module and script |
| [Command reference](commands.md) | Installation, GPU defaults, all entry points, important flags, examples, and troubleshooting |
| [Pinned GPU setup](gpu-setup.md) | Fresh environment installation, exact dependency checks, CUDA training/inference verification, and saved evidence |
| [Artifacts and provenance](artifacts.md) | JSON/JSONL/CSV/checkpoint formats, file relationships, exports, and reproducibility boundaries |
| [Optional inference API](inference-api.md) | Saved preprocessing, request/response contract, routes, errors, timing, and local deployment limits |
| [Interrupted-run recovery](resuming-runs.md) | Durable trial recovery, deterministic replay, locks, failure handling, and version boundaries |
| [Validation and results](validation-and-results.md) | Tests, measured suite, score interpretation, audit evidence, and verification limits |
| [Current state and cleanup](maintenance.md) | Removed redundancy, retained optional tools, deferred issues, and future maintenance |
| [Complete project review](project-audit.md) | Latest source review, independent results checks, GPU integration, recovery fix, and remaining limits |
| [Portable delivery](portable-delivery.md) | Implemented relative records, short GPU example, model bundles, and Docker/CI usage and verification limits |

## Existing records and optional components

- [Project direction](project-direction.md): industry goal and user decisions.
- [Improvement roadmap](improvement-roadmap.md): prioritized portability/examples, Docker verification, optional hosting, and later pipeline improvements.
- [Research direction](research-direction.md): preserved ideas and literature from the earlier review; parked.
- [Industry results](industry-results.json): machine-readable snapshot of the measured Covertype suite.
- [Industry demo](industry-demo.md): previously implemented optional packaging/API/dashboard instructions.
- [Scoring audit](scoring-audit.md) and [audit data](scoring-audit.json): independent replay of the earlier six-search pilot.
- `covertype-gpu-pilot.json`: short Covertype pilot measurements.
- `tabular-pilot.json` and `tabular-gpu-pilot.json`: earlier Breast Cancer measurements.

The dashboard and report-generation tools are optional, existing components.
The current cleanup leaves them in place and adds no dashboard or reporting features.
Use `main.py`, `run_comparison.py`, and `evaluate_best.py` for the core workflow.

## Documentation conventions

Paths are relative to the repository root unless stated otherwise. Commands assume
an activated Python 3.12 virtual environment and execution from the repository root.
`PATH_TO_...` values are placeholders; replace them with paths printed by a run.
Percentages in prose multiply stored fractional values by 100. Validation and test
results are labelled separately. A documented command is not evidence that a new
experiment was run. Existing benchmark JSON is a historical snapshot and is not
rewritten merely because code or documentation changes.
