# Tabular Neural Architecture Search

Compare generational genetic search (GA), aging evolution, and random search using compact MLPs. **Covertype is the main benchmark**; Breast Cancer Wisconsin remains a small offline check. Experiments run sequentially on one CPU or one GPU, including an RTX 4060 laptop. There is no multi-GPU worker pool or DataParallel path.

This is an **AI/ML engineering portfolio project** focused on reproducible workflows, reliable evaluation, and practical compute tradeoffs. See [project direction](docs/project-direction.md) for current priorities and the research ideas saved for later.

New searches and comparison suites can resume after interruption without retraining completed trials. See [resuming runs](docs/resuming-runs.md) for `--resume` commands and recovery details.

## Dataset and preprocessing

[UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) contains 581,012 rows, 54 features, and seven forest cover classes. It has 10 numerical columns and 44 binary columns. The fixed stratified 60/20/20 split gives 348,607 training rows, 116,202 validation rows, and 116,203 test rows. StandardScaler is fitted exclusively on the numerical columns of the training split; binary columns are preserved. Class counts, scaler statistics, data fingerprint, split seed, and split index fingerprints are recorded. Full split indices can be reconstructed from the fixed dataset ordering and seed and verified against those fingerprints.

Search defaults to a **fixed stratified 20,000-row training subset and 10,000-row validation subset**, batch size 512. Every method uses the same rows; training shuffle/initialization seeds vary independently. Preprocessing uses the full training split, while candidate training uses the subset. Final winner retraining uses the full training split and selects its checkpoint using the full validation split. Test scores are withheld during comparisons and calibration.

Breast Cancer Wisconsin uses 569 rows, 30 numerical features, and two classes. Its defaults remain all 341 training rows and 114 validation rows, batch size 32. It is useful for checking execution but proved too easy to distinguish the methods in the first pilot.

The CLI defaults to Covertype. Library functions retain Breast Cancer defaults for compatibility with earlier scripts; pass `dataset="covertype"` when using the Python API.

## Installation

Use Python 3.12 and an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For the RTX 4060 with a CUDA 13.0-compatible NVIDIA driver:

```powershell
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements-tested-cpu.txt` pins the tested CPU environment used by CI and installs CPU-only PyTorch. For other GPU drivers/platforms, choose a build with the [official PyTorch installer](https://pytorch.org/get-started/locally/). GPU packages are a substantial one-time download. Device selection is explicit; an unavailable CUDA device causes an error.

Download Covertype once before searching. The [scikit-learn fetcher](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html) verifies the source archive and caches the data under `data/tabular_cache/`, which is ignored by Git. Search never initiates a network download:

```powershell
python prepare_dataset.py --dataset covertype
python -m unittest discover -s tests -v
```

The tests use synthetic Covertype data, so CI needs no dataset download. They cover split/subset separation, training-only scaling, unchanged binary columns, seven-class forward/backward passes, parameter estimates, checkpoint restoration, dataset provenance, and loss differences when accuracy ties.

## Run a GPU pilot

```powershell
python run_comparison.py --dataset covertype --device cuda --budget 20 --population 5 --seeds 42 43 --proxy-epochs 5
```

This evaluates 120 candidates across three methods and two seeds. GA has four generations per seed; aging has 15 replacements after its initial population. The initial candidates and their training seeds match across methods for a controlled starting point. Search trajectories then diverge. A shared initial winner can still produce a legitimate tie.

Measured RTX 4060 pilot using this command: **100.86 seconds** summed across six searches. Mean best proxy validation accuracy was GA **72.58%**, aging **73.95%**, and random **73.925%**. Scores were distinct within each seed; aging/random were essentially tied on average. All 120 trials used `cuda:0`, the same 20,000/10,000 fixed subsets, and valid outcomes. This five-epoch, two-seed pilot checks the new dataset and measures runtime; it does not establish a reliable winner or validate proxy ranking fidelity. No final retraining or test evaluation was performed. Configuration, score/loss variation, and measured results are in `docs/covertype-gpu-pilot.json`.

Use measured pilot durations to size the main comparison. Covertype training costs substantially more than the 569-row dataset; earlier Breast Cancer timing estimates do not apply. Increasing epochs or candidate budget increases cost approximately proportionally, but model size and full retraining also matter.

```powershell
python run_comparison.py --dataset covertype --device cuda --budget 100 --population 20 --seeds 42 43 44 --proxy-epochs 20 --full-epochs 30 --full-seeds 101 102
```

This costs 900 proxy evaluations and 18 final retrainings. Final retrainings use all 348,607 training rows and validate on 116,202 rows, so they require a separate runtime allowance. Test data is not scored. Increase the number of search seeds before making claims about consistent superiority.

`--proxy-size` and `--validation-size` override fixed subset sizes. Zero means all rows in the corresponding training or validation split. Apply identical settings to all methods. Larger fixed validation subsets improve score resolution; they do not guarantee distinct winners. Validation accuracy remains the primary fitness, and earliest observations win exact ties. Validation cross-entropy is also recorded to expose confidence differences between equal-accuracy candidates; it is not silently substituted as the fitness.

`--smoke` generates synthetic scores for orchestration checks. Smoke winners are rejected by training and provide no accuracy evidence.

## Training and search protocol

Architecture schema 3 encodes nine choices: depth (1–4 hidden layers), four widths (16/32/64/128), activation (ReLU/leaky ReLU/ELU), dropout (0/0.1/0.3), LayerNorm, and residual connections. Input width and output classes come from the dataset. Unused hidden widths are inactive. Residual connections apply only when adjacent widths match. Aging mutation changes one expressed choice. LayerNorm supports singleton batches.

Candidates start from scratch with Adam, learning rate 0.001, weight decay 0.0001, cosine decay, and deterministic seeds. CPU training uses one thread. Search randomness is independent of training randomness. All methods receive equal candidate evaluation budgets and the same per-trial seed schedule. GA retrains survivors, which consumes budget. Matching evaluations does not match wall-clock cost; both time and parameter counts are recorded.

Search fitness is end-of-proxy validation accuracy. Full training saves and restores the best-validation checkpoint and records that checkpoint's validation loss. A durable archive preserves the historical best search candidate. Fatal dataset/programming failures stop and record the run; memory exhaustion, divergence, and parameter-limit violations have explicit candidate statuses.

## Calibrate the proxy

```powershell
python calibrate_proxy.py --dataset covertype --device cuda --samples 12 --seeds 101 102 --proxy-epochs 20 --long-epochs 30 --top-k 3
```

Calibration measures Spearman rank agreement and top-k overlap between the proxy and longer full-data training, using validation only. Proxy scores use the fixed validation subset; reference scores use the full validation split. This measures the combined fidelity of the smaller training set, validation subset, and shorter training schedule. If agreement is weak, increase training fidelity before the main comparison. Undefined correlations from constant scores are reported as such.

A larger dataset makes this a more informative experiment, but it does not guarantee evolution beats random search. Compare repeated results, uncertainty, and cost. Include a fixed MLP and eventually additional datasets before claiming a general NAS advantage.

## Individual searches and fixed MLP

```powershell
python main.py --dataset covertype --mode nas --pop 10 --gen 5 --proxy-epochs 10 --device cuda
python main.py --dataset covertype --mode aging --pop 10 --n-eval 50 --proxy-epochs 10 --device cuda
python main.py --dataset covertype --mode random-search --n-eval 50 --proxy-epochs 10 --device cuda
python main.py --dataset covertype --mode train-mlp --full-epochs 30 --seed 101 --device cuda --validation-only
```

The fixed baseline has two hidden layers (64 and 32), ReLU, dropout 0.1, no normalization or skips. Its input/output sizes match the dataset, and it uses the shared final training/checkpoint protocol.

```powershell
python evaluate_best.py --json PATH_TO_WINNER_JSON --epochs 30 --seed 101 --device cuda --validation-only
python evaluate_best.py --test-checkpoint PATH_TO_FULL_TRAIN_JSON --device cuda
```

Retraining reads the dataset and, by default, the split seed from the saved winner. Test evaluation reconstructs the model from its saved dataset and verifies the current dataset fingerprint. Freeze the protocol and candidate choices before evaluating test. A checkpoint with a recorded test score cannot be tested again through this command.

## Historical pilots and outputs

New runs live under `experiments/tabular/`, with unique run directories/files and explicit dataset names in records. Search writes generation CSV, trial JSONL, metadata JSON, and a winner JSON. Comparison manifests are saved incrementally. Full training saves weights, history, protocol, and paths. Results are not silently overwritten. Comparison plots reject files from different datasets.

The earlier **Breast Cancer** pilots are preserved in `docs/tabular-pilot.json` and `docs/tabular-gpu-pilot.json`. On that task, all methods selected the same initial candidate and tied at 98.25% proxy accuracy. GPU retraining with seed 101 tied at 99.12%. The GPU searches took 21.40 seconds and three 30-epoch retrainings took 4.12 seconds. Corresponding CPU sums were 9.42 and 1.18 seconds. These timings exclude package installation and Python startup and do not predict Covertype runtime. To reproduce that task, explicitly pass `--dataset breast_cancer_wisconsin`.

Historical CIFAR loaders, CNN builder, ResNet code, reports, and artifacts remain for reference. Their schema 2 winners are rejected by the tabular trainer. `requirements-cifar.txt` supplies optional torchvision; restore an earlier Git revision to reproduce the CIFAR pipeline.

`benchmark_search.py` is a separate optional NATS-Bench topology lookup experiment (`requirements-benchmark.txt`). It requires its downloaded archive and evaluates a different image search space; its results do not establish an advantage for the tabular MLP task.

`run_ablations.py` and `run_hyper_sweep.py` now accept `--dataset`, defaulting to Covertype. Ablations compare operators under equal candidate evaluation budgets; sweeps record the unequal training costs when proxy epochs vary. Neither evaluates test scores.
