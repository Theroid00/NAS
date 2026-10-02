# Installation and command reference

## Environment and GPU

Use Python 3.12 from the repository root. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python prepare_dataset.py --dataset covertype
```

The measured environment used torch 2.14.1+cu130 on an RTX 4060 Laptop GPU with
a CUDA-compatible NVIDIA driver. `requirements-gpu.txt` now pins the complete
tested Windows/Python 3.12 core GPU package set, including transitive dependencies.
Optional service dependencies are pinned separately. Other OS/Python combinations
are not covered by this installation check. The measured benchmark versions remain
preserved in `industry-results.json`. Installing an
unavailable CUDA build or an incompatible driver cannot be fixed by adding a
device flag alone. No command silently falls back to CPU after CUDA is requested.

For the explicit offline CPU environment, install `requirements-tested-cpu.txt`
in a separate environment and pass `--device cpu`. The optional API and its tests
need `python -m pip install -r requirements-service.txt`. The core searches do not
need FastAPI. `python -m pip check` checks installed dependency compatibility;
it does not prove model or driver correctness.

All maintained real-training CLIs now default to CUDA: `main.py`,
`run_comparison.py`, `evaluate_best.py`, `calibrate_proxy.py`, `run_ablations.py`,
`run_hyper_sweep.py`, `run_portfolio.py`, and `serve.py`. Library functions retain
explicit CPU-compatible defaults. Smoke runs use synthetic scores and can run
without CUDA even if the device setting says cuda; their records are marked smoke.

## Recommended core commands

```powershell
# One GA run: 50 candidate evaluations, 10 proxy epochs.
python main.py --mode nas --pop 10 --gen 5 --proxy-epochs 10 --device cuda

# Matched aging or random run.
python main.py --mode aging --pop 10 --n-eval 50 --proxy-epochs 10 --device cuda
python main.py --mode random-search --n-eval 50 --proxy-epochs 10 --device cuda

# Measured suite protocol: 750 candidates and 15 full retrainings; no test scoring.
python run_comparison.py --dataset covertype --device cuda --budget 50 --population 10 --seeds 42 43 44 45 46 --proxy-epochs 10 --full-epochs 20 --full-seeds 101

# Retrain a saved winner, keeping its original dataset/split and withholding test.
python evaluate_best.py --json PATH_TO_WINNER_JSON --epochs 20 --seed 101 --device cuda --validation-only

# After freezing selection and protocol, test a saved full-training checkpoint.
python evaluate_best.py --test-checkpoint PATH_TO_FULL_TRAIN_JSON --device cuda
```

Every method trains candidates sequentially. The 50-candidate suite budget is
per method per search seed, not the total across all methods/seeds. Full epochs
add a separate retraining cost. Avoid unintentionally running the much larger
standalone defaults when a short check is intended.

## `main.py` options

| Option | Default / behavior |
| --- | --- |
| `--dataset` | `covertype`; alternative `breast_cancer_wisconsin` |
| `--mode` | `nas`; also `aging`, `random-search`, `train-best`, `train-mlp`, `plot` |
| `--device` | `cuda`; CPU requires explicit `cpu` |
| `--pop`, `--gen` | 20, 15; GA budget is their product; population also applies to aging |
| `--n-eval` | 300; aging/random candidate budget |
| `--proxy-epochs` | 20 |
| `--proxy-size`, `--validation-size` | Dataset defaults; zero uses the entire corresponding split |
| `--crossover-p`, `--mutation-p` | 0.8, 0.1; GA operators |
| `--tournament-k`, `--elites` | 5, 2; GA settings |
| `--max-params` | No limit; positive value rejects oversized candidates |
| `--seed` | 42; search seed or full-training seed according to mode |
| `--split-seed` | 42 for a new search; saved split for winner retraining unless overridden |
| `--best-json` | Explicit saved winner; otherwise newest compatible winner in `--save-dir` |
| `--full-epochs` | 100; full-training modes |
| `--validation-only` | Withholds test in full-training modes |
| `--resume` | Metadata JSON; restores original search method/settings, ignoring other search flags |
| `--smoke` | Synthetic scoring only |
| `--log-dir` | `experiments/tabular/generation_logs` |
| `--save-dir` | `experiments/tabular/best_architectures` |

`train-best` uses the same `train_winner` implementation as `evaluate_best.py`.
`train-mlp` explicitly trains the optional fixed 64→32 MLP. `plot` reads generation
CSV; normal searches also try to save convergence/architecture figures when
plotting dependencies are available. Random search has no generation CSV.

## `run_comparison.py` options

The defaults are methods `ga aging random`, search seeds `42 43 44`, budget 300,
population 20, proxy epochs 20, split seed 42, dataset Covertype, and device CUDA.
`--proxy-size`, `--validation-size`, and `--max-params` work as in individual search.
`--methods` and `--seeds` require distinct entries. The budget must cover the
initial population. `--full-epochs` is unset by default, so there is no full
retraining unless requested. `--full-seeds` defaults to 101; repeats retrain each
winner independently. Full retraining is incompatible with `--smoke`.

`--out-dir` defaults to `experiments/tabular/comparisons`; a unique comparison
directory is created underneath. `--resume PATH_TO_COMPARISON_JSON` restores its
original configuration, skips recorded completed work, and recovers the active
search. Suite records are updated after searches and retrainings.

## `evaluate_best.py` options

`--json` selects a saved search winner; otherwise the command looks for the latest
compatible winner in `--save-dir`. Defaults are CUDA, 100 epochs, training seed
42, and `experiments/tabular/best_architectures`. `--split-seed` is an explicit
override. Supply `--validation-only` while comparing candidates. With
`--test-checkpoint`, the command loads the full-training result and evaluates its
existing weights rather than retraining; training flags do not change that checkpoint.
An already-tested checkpoint is rejected.

## Recovery and quick orchestration checks

```powershell
python main.py --resume PATH_TO_SEARCH_METADATA_JSON
python run_comparison.py --resume PATH_TO_COMPARISON_JSON
python run_comparison.py --smoke --budget 7 --population 3 --seeds 42 43
```

Recovery requires the original checkout location, dependencies, dataset, and
supported saved format. It does not resume an unfinished epoch. See the dedicated
[recovery guide](resuming-runs.md). Smoke checks verify record/controller behavior,
not learning or real accuracy; smoke winners cannot be retrained or packaged as
measured models.

## Optional analysis tools

| Command | Important settings and behavior |
| --- | --- |
| `calibrate_proxy.py` | Defaults: 12 architectures, training seeds 101/102, sample/split seed 42, proxy epochs 20, long epochs 100, top-k 3, CUDA/Covertype; accepts subset sizes; scores validation only |
| `run_ablations.py` | Defaults: search seeds 42/43/44, budget 200, proxy epochs 20, split 42, CUDA/Covertype; runs five GA configurations; `--smoke` available |
| `run_hyper_sweep.py` | Defaults: seeds 42/43/44, budget 150, populations 10/20/30, proxy epochs 5/10/20, split 42, CUDA/Covertype; unequal epoch costs are explicit |
| `audit_scoring.py PATH_TO_COMPARISON_JSON --output PATH_TO_AUDIT_JSON` | Replays archived proxy winners on CUDA, independently verifies scoring, and does not use test rows; this command performs real training |
| `benchmark_search.py --benchmark PATH_TO_NATS_ARCHIVE` | Requires optional benchmark dependencies/archive; defaults to 30 seeds, budget 300, population 20, stored 12-epoch results; optional `--hp 200`; no live tabular GPU training |
| `generate_comparison.py --results ...` | Explicit full-training JSONs; optional labels; metric defaults to test accuracy or can be `best_val_accuracy`; all requested scores must exist and belong to the same dataset |
| `demo_replay.py --csv PATH_TO_GENERATION_CSV` | Animates an existing log; requires an interactive plotting environment |
| `verify_environment.py --device cuda` | Checks exact GPU package pins and executes one offline Breast Cancer training epoch on CUDA; optional `--service`, `--artifact`, and `--out` |

Ablations, sweeps, calibration, and the NATS lookup save their own JSON summaries;
they do not have the comparison suite's durable recovery interface. Use `--help`
on a script for its exact accepted options. These tools are retained for distinct
analysis purposes rather than required for the normal hiring demo.

## Optional packaging and inference

```powershell
python make_example.py --results PATH_TO_FULL_TRAIN_JSON --out example-request.json
python export_model.py --results PATH_TO_FULL_TRAIN_JSON --out PATH_TO_NEW_ARTIFACT_DIRECTORY --example example-request.json
python serve.py --artifact PATH_TO_ARTIFACT_DIRECTORY --device cuda
```

The example contains a raw training row, not a held-out test row. Export requires
a new destination directory. The API scales features using the saved manifest
and loads no training dataset. `serve.py` defaults to host 127.0.0.1, port 8000,
and one CPU preprocessing thread. `--artifact`, `--report`, and `--device` can also
come from `NAS_ARTIFACT`, `NAS_REPORT`, and `NAS_DEVICE`. Omitting a report leaves
predictions usable but the existing dashboard comparison view has no report.

`run_portfolio.py` optionally chains the measured suite protocol, validation
selection, one test evaluation, export, and a report. It defaults to Covertype,
CUDA, five seeds 42–46, budget 50, population 10, proxy epochs 10, full epochs 20,
and retraining seed 101. `--comparison` packages a completed real suite;
`--resume` recovers and packages one. Those modes restore the suite's device rather
than applying a new `--device`. `--report` defaults to `docs/industry-results.json`,
and `--artifact-root` to `artifacts/industry`. Outputs can replace a previous report.

`report_results.py --comparison ... --out ...` builds a plain report for existing
presentation code. Its default path is also `docs/industry-results.json`; use a
separate output to avoid removing enrichment from a portfolio report. Existing
report/dashboard issues are recorded in [maintenance.md](maintenance.md) and
are deferred at the user's request.

## Common troubleshooting

| Symptom | Check / explanation |
| --- | --- |
| Requested CUDA unavailable | Check torch's CUDA build, driver compatibility, and `torch.cuda.is_available()`; the code will not silently switch to CPU |
| Covertype cache missing | Run `prepare_dataset.py --dataset covertype` once; search does not download |
| Permission denied reading cached data | Check OS account permissions; in a restricted agent sandbox the cached files may need approved access |
| Subset cannot be stratified | Increase subset size enough to represent every class |
| Tied search scores | Fixed initialization candidates may match; discrete accuracy can tie; inspect loss/architecture and trial records |
| No valid candidate | Inspect status/error fields, parameter limits, memory exhaustion, and divergence |
| Resume version/dependency mismatch | Preserve the original environment or start a new run; do not edit metadata to force compatibility |
| Export destination already exists | Complete artifacts are intentionally preserved; choose another directory. Failed new exports do not publish the destination and can retry the same name |
| Model weights checksum differs | Artifact is corrupted or weights/manifest do not belong together |
| Prediction HTTP 422 | Supply 1–1024 numeric finite rows in exact feature order; Covertype's last 44 values must be binary |
