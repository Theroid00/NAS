# Tabular Neural Architecture Search

Find the best observed neural network architecture for a supported tabular dataset
within a fixed evaluation budget. The project compares a genetic algorithm,
aging evolution, and random search using the same MLP space and training protocol.
Validation accuracy drives selection; balanced accuracy, macro F1, class scores,
model size, and time describe the tradeoffs.

**Start reading at [docs/README.md](docs/README.md).** The documentation explains
all maintained code, algorithms, data handling, commands, artifacts, recovery,
validation, results, and current limitations.

## Core GPU workflow

For a short offline integration example, run `python quickstart.py --device cuda`.
For saved-model prediction, use `python predict_example.py --artifact PATH --device cuda`.
See [portable examples and Docker delivery](docs/portable-delivery.md) for artifact
bundles, moving experiments, and serving the inference API in a container.

Use Python 3.12 and an isolated environment. The CUDA requirements target the
RTX 4060 environment used for the measured suite; a compatible driver is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gpu.txt
python prepare_dataset.py --dataset covertype
python main.py --mode nas --pop 10 --gen 5 --proxy-epochs 10 --device cuda
```

Compare all three methods and retrain each winner while withholding test:

```powershell
python run_comparison.py --dataset covertype --device cuda --budget 50 --population 10 --seeds 42 43 44 45 46 --proxy-epochs 10 --full-epochs 20 --full-seeds 101
```

This is 750 candidate evaluations and 15 retrainings. Experiments train candidates
sequentially on one GPU. Training CLIs default to CUDA and fail when unavailable;
`--device cpu` is explicit. Library functions retain small offline CPU defaults.
Completed trials can be recovered without retraining; see [recovery](docs/resuming-runs.md).
The GPU package set is pinned. Use `python verify_environment.py --device cuda`
to check installed versions and execute a tiny real CUDA training check; see
[fresh GPU setup](docs/gpu-setup.md). New runs record startup source hashes and
reject recovery with changed implementation files.

Retrain a specific saved winner with `evaluate_best.py --json ... --validation-only`.
After freezing the choice, use `evaluate_best.py --test-checkpoint ...` to evaluate
its saved checkpoint. See [all commands](docs/commands.md) for paths and options.

## Measured result

On Covertype (581,012 rows, 54 features, seven classes), mean full-validation
accuracy across five search seeds was GA **91.66%**, aging **92.53%**, and random
**90.97%**. The validation-selected aging model achieved **94.09% held-out test
accuracy**. These are results for the documented budget and protocol, not a
universal search advantage. See [validation and results](docs/validation-and-results.md).

## Scope and optional tools

The goal is an AI/ML engineering portfolio: reproducible search, correct scoring,
single-GPU execution, trial recovery, and readable code. Covertype is the main
dataset; Breast Cancer Wisconsin is a small offline check. The existing model
export, API, dashboard, and report tools remain optional; dashboard/report work
is currently deferred. The fixed MLP and NATS-Bench lookup are separate optional
controls, not participants in the main comparison.

The obsolete CIFAR loader/CNN/ResNet path and unused compatibility wrappers were
removed from the current code; Git history preserves them. See [cleanup and known
gaps](docs/maintenance.md), [industry direction](docs/project-direction.md), and
[saved research direction](docs/research-direction.md).
