# Tabular Neural Architecture Search

Compare generational genetic search (GA), aging evolution, and random search using small MLPs on Breast Cancer Wisconsin classification. This branch targets one laptop with one CPU or one GPU. Candidates run sequentially; there is no multi-GPU worker pool or DataParallel path.

The dataset ships with scikit-learn: 569 rows, 30 numerical features, two classes, no runtime download. A fixed stratified 60/20/20 split gives 341 training, 114 validation, and 114 test rows. StandardScaler is fitted on training rows only. All methods share those splits and the same training protocol. Dataset fingerprint, split indices, scaler statistics, dependency versions, trial seeds, architecture schema, and Git revision are saved for provenance.

## Install and verify

Use Python 3.12 in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements-tested-cpu.txt` pins the locally tested CPU environment used by CI. For the remote RTX 4060, install a CUDA-capable PyTorch build using the [official installer](https://pytorch.org/get-started/locally/) and use `--device cuda`. The CPU lock file installs CPU-only PyTorch. CPU is the default; these tiny models may run faster on CPU because GPU launch and transfer overhead can dominate. Measure both on your laptop before choosing.

## Run comparisons

Start with a quick pilot (three methods, one seed, 60 total candidate evaluations):

```powershell
python run_comparison.py --device cpu --budget 20 --population 10 --seeds 42 --proxy-epochs 10
```

A larger repeated comparison evaluates 100 candidates per method for three search seeds, then retrains each winner with two fresh seeds. Test data is never scored by this command:

```powershell
python run_comparison.py --device cuda --budget 100 --population 20 --seeds 42 43 44 --proxy-epochs 20 --full-epochs 100 --full-seeds 101 102
```

This costs 900 proxy evaluations and 18 final training runs. Reduce `--budget`, number of `--seeds`, or `--proxy-epochs` for faster iteration. Reduce all methods equally. The program records measured search durations; estimate a larger run by scaling the pilot's total evaluation time by evaluation count and epoch count, then allow for final retraining. This is an estimate, not a measured RTX 4060 runtime.

The default comparison uses 300 evaluations per method, three seeds, and 20 proxy epochs; final retraining is enabled only with `--full-epochs`. `--proxy-size 0` uses all 341 training rows. Positive sizes select a fixed stratified training subset; validation always uses the same 114 rows. Reducing epochs or budget is preferable to discarding rows in this already small dataset.

Use `--smoke` for orchestration checks with generated scores. Smoke records are rejected by training and provide no accuracy evidence.

## Search space and fairness

Schema 3 encodes nine choices: depth (1–4 hidden layers), four widths (16/32/64/128), activation (ReLU/leaky ReLU/ELU), dropout (0/0.1/0.3), LayerNorm, and residual connections. Unused widths are inactive. Residual connections apply only when adjacent widths match. Aging mutation changes one expressed choice. LayerNorm supports even singleton training batches.

Every candidate starts from scratch using Adam (learning rate 0.001, weight decay 0.0001), cosine decay, batch size 32, and deterministic seeds. CPU training uses one thread to avoid overhead for tiny matrices. Search RNG is independent of training RNG. All three methods receive identical candidate evaluation budgets and the same per-trial seed schedule. GA re-evaluates survivors; those retrainings consume budget. Candidate evaluation matching is not identical wall-clock or parameter matching; durations and parameter counts are also logged.

Search fitness is end-of-proxy validation accuracy. Full training selects its checkpoint by best validation accuracy. Historical best search candidates remain archived even after population replacement. Infrastructure failures stop a run and are recorded; memory exhaustion, divergence, and parameter limit failures have explicit statuses.

## Calibrate before making claims

```powershell
python calibrate_proxy.py --device cpu --samples 12 --seeds 101 102 --proxy-epochs 20 --long-epochs 100 --top-k 3
```

Calibration measures Spearman rank agreement and top-k overlap between proxy and longer training, using validation only. If agreement is weak, increase proxy epochs before the main comparison. Scores can tie because validation has only 114 rows; undefined correlation is reported rather than invented.

This small dataset is an inexpensive engineering benchmark, not enough evidence for a general NAS research claim. It may saturate quickly, so random search can tie or win. Run repeated seeds, compare variation and cost, include a fixed MLP, and expand to larger tabular datasets if the methods are indistinguishable. Tabular results must never be compared numerically with historical CIFAR results.

## Individual searches and fixed MLP

```powershell
python main.py --mode nas --pop 10 --gen 5 --proxy-epochs 20 --device cpu
python main.py --mode aging --pop 10 --n-eval 50 --proxy-epochs 20 --device cpu
python main.py --mode random-search --n-eval 50 --proxy-epochs 20 --device cpu
python main.py --mode train-mlp --full-epochs 100 --seed 101 --validation-only
```

The fixed baseline is 30→64→32→2 with ReLU, dropout 0.1, no normalization or skips (4,130 parameters). It uses the same final training and checkpoint protocol.

```powershell
python evaluate_best.py --json PATH_TO_WINNER_JSON --epochs 100 --seed 101 --validation-only
python evaluate_best.py --test-checkpoint PATH_TO_FULL_TRAIN_JSON --device cpu
```

Evaluate test only after freezing the protocol and candidate choices. A checkpoint with a recorded test score cannot be tested again through this command. Use the winner's original `--split-seed` when retraining; training seeds can vary independently.

## Outputs and historical files

New outputs live under `experiments/tabular/`, in unique run directories/files. Search writes generation CSV, trial JSONL, metadata JSON, and a winner JSON. Comparison writes its manifest incrementally. Full training saves model weights, history, training configuration, and paths. Existing files are never silently overwritten.

Historical CIFAR loaders, CNN builder, ResNet code, reports, and experiment artifacts remain for reference. Their schema 2 winners are rejected by the schema 3 trainer; the current CLI trains MLPs. Optional `requirements-cifar.txt` installs torchvision for historical code. Restore an earlier Git revision to reproduce the earlier CIFAR pipeline.

`benchmark_search.py` remains an optional separate NATS-Bench topology lookup experiment (`requirements-benchmark.txt`). It requires the downloaded benchmark archive and evaluates a different image search space; its results do not establish an advantage for this tabular MLP task.

## Additional experiments

`run_ablations.py` tests crossover, mutation, elitism, and population changes using repeated equal evaluation budgets. `run_hyper_sweep.py` explores population sizes and proxy epochs, recording that varying epoch counts changes training cost. Both now use the tabular pipeline. Neither accesses test scores.
