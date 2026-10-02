# Neural Architecture Search with Genetic Algorithms

A CIFAR-10 research project comparing generational GA, aging evolution, and random search. Candidates are trained from scratch with a common proxy protocol. The aim is to measure whether search history improves architecture selection under a fixed budget.

## Historical results and their limits

The repository retains these original measurements:

| Method | Reported proxy accuracy | Full test accuracy | Parameters |
| --- | ---: | ---: | ---: |
| GA-NAS | 72.15% at final generation | 92.07% | 19,061,642 |
| Random search | 69.90% | 92.15% | 5,605,130 |
| Fixed ResNet | — | 90.52% | 1,227,594 in the current implementation |

These are historical results, not measurements from the revised pipeline. The original GA saved winners using incorrectly aligned population/fitness arrays; the ResNet used a different optimizer and final-epoch checkpoint policy. The main GA log peaked at 73.20%, above its reported final score. A single pair of full-training results does not demonstrate an advantage for either search method. Existing experiment files are preserved for provenance; re-run comparisons before making performance claims about this branch.

## What changed

- Winner selection uses a durable archive of evaluated chromosome/fitness pairs.
- Every trial records architecture, fitness, status, training seed, split seed, device, parameter count when available, and elapsed time in JSONL.
- Metadata records configuration, search-space schema, dependency versions, Git revision, actual devices, and completion/failure status.
- Search randomness uses a private RNG; model training and DataLoader randomness use separate seeds.
- Parallel search uses one spawned process per GPU, preventing shared global RNG races. Unsupported devices fail before search; one-GPU parallel mode is rejected.
- Infrastructure/programming failures stop the run. Divergence, CUDA memory exhaustion, and parameter-limit violations are explicit candidate outcomes rather than unexplained zero scores.
- GA, random winners, and fixed ResNet share Adam, cosine decay, augmentation, dataset splits, and best-validation checkpoint selection for new full-training runs.
- Smoke results and incompatible saved chromosomes are rejected by full-training commands.
- Aging evolution, repeated comparisons, proxy-rank calibration, and an optional offline topology benchmark are available.

## Installation and tests

Use Python 3.10 or newer. Python 3.12 with CPU PyTorch was used for local verification. Install a CUDA-enabled PyTorch build appropriate to your hardware before running GPU experiments; the test environment below is CPU-only.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements-tested-cpu.txt` records the exact locally tested environment. Install it in a separate virtual environment when reproducing CPU checks. DEAP is not required: evolutionary operators are implemented directly.

Dataset downloads go to `data/cifar10_data/`, which is ignored by Git. The `data` Python package is tracked. CIFAR loading is cached within each evaluation process. Downloads or filesystem failures stop a real search rather than silently assigning every candidate zero fitness.

## Search methods

```powershell
# GA plumbing check: random scores, no model training or dataset download
python main.py --smoke --pop 5 --gen 2 --log-dir experiments/comparisons/smoke --save-dir experiments/comparisons/smoke

# Generational GA: 20 x 15 = 300 candidate evaluations
python main.py --pop 20 --gen 15 --device cuda --seed 42 --split-seed 42

# Aging evolution: mutate one active choice, evaluate child, remove oldest member
python main.py --mode aging --pop 20 --n-eval 300 --device cuda --seed 42

# Persisted random-search winner using the same proxy evaluator
python main.py --mode random-search --n-eval 300 --device cuda --seed 42

# Concurrent evaluation on all available GPUs
python main.py --pop 20 --gen 15 --device cuda --parallel
```

GA defaults: tournament size 5, two elites, crossover probability 0.8, per-gene mutation 0.1. After three generations without a new observed maximum, mutation doubles to 0.2 at these defaults (cap 0.3), then resets when improvement resumes. Elites are retrained, and those evaluations consume budget; old weights are not inherited. Best-observation selection can favor lucky training outcomes, so repeat finalist training before drawing conclusions.

Aging evolution uses tournament size 5 and FIFO replacement. Its mutation changes one expressed gene to a different value, avoids unused filter genes, and treats average/mixed pooling as equivalent when choosing a pooling mutation. It retains an independent history even after population members age out. Its initial population can be evaluated in parallel; subsequent children are evaluated one at a time to preserve the sequential algorithm.

`--max-params 6000000` applies a common optional parameter ceiling to each search method. Rejected candidates count as attempted evaluations and are explicitly logged. If no valid candidate is found, the run fails without exporting a winner.

Smoke fitness depends on trial seeds rather than architecture quality. Equal smoke budgets/seeds can therefore produce equal best scores across methods; smoke comparisons test plumbing only.

## Repeated comparisons

```powershell
# Quick check: all three methods, seven evaluations each, two search seeds
python run_comparison.py --smoke --budget 7 --population 3 --seeds 42 43

# Real search pilot with identical candidate-evaluation budgets
python run_comparison.py --device cuda --budget 300 --population 20 --seeds 42 43 44

# Repeat each winner's full training; validation only, without test evaluation
python run_comparison.py --device cuda --budget 300 --population 20 --seeds 42 43 44 --full-epochs 50 --full-seeds 101 102
```

Each invocation creates a unique directory under `experiments/comparisons/`, with a comparison manifest and per-method/per-seed trials and winners. The manifest reports proxy mean/sample standard deviation, full-validation statistics when requested, search wall time, and summed candidate evaluation time. A single-search standard deviation is `null` rather than misleadingly zero.

The primary budget is candidate evaluations, not equal GPU compute. Model sizes differ; parameter rejection has a different cost from training. Use recorded evaluation times and hardware allocation to assess compute efficiency. `total_evaluation_seconds` is summed evaluator wall time, not a measurement of GPU utilization. Do not select a method by repeatedly inspecting test scores. Choose and freeze the comparison using validation, then perform explicit final test evaluations.

## Full training and final test evaluation

```powershell
# Any saved GA/aging/random winner; 45k train, 5k validation
python evaluate_best.py --json path/to/best_run.json --device cuda --epochs 50 --seed 101 --validation-only

# Fixed ResNet with the same new training/checkpoint policy
python main.py --mode train-resnet --device cuda --full-epochs 50 --seed 101 --validation-only

# Evaluate a frozen checkpoint on the official test set once
python evaluate_best.py --test-checkpoint path/to/full_train_run.json --device cuda
```

Omitting `--validation-only` in standalone full-training commands evaluates test accuracy after loading the best validation checkpoint. Comparison and calibration runners always request validation-only training. Checkpoints save unwrapped model weights, allowing a different device count when loading. `--parallel` opts into full-training DataParallel; use the same choice across baseline methods.

The standalone random-winner trainer now accepts the same arguments as `evaluate_best.py`, including `--json`; it no longer embeds a historical chromosome. Newest-file discovery skips marked smoke and incompatible records. Ten-gene legacy results require an explicit migration by architectural field name; they are not padded or silently reinterpreted. Thirteen-gene legacy files can load if chromosome and architecture agree, but their missing experiment provenance still limits reproducibility.

## Validate the proxy before tuning search

```powershell
python calibrate_proxy.py --device cuda --samples 12 --seeds 101 102 --proxy-epochs 5 --long-epochs 20
```

This samples distinct effective architectures, evaluates each with the proxy and longer training under repeated seeds, and reports Spearman rank correlation and top-k shortlist overlap. Average ranks handle ties. Constant scores produce an undefined (`null`) correlation. Candidate failures are retained; only architectures with complete paired repeats contribute to the aggregate ranking. Longer-training accuracy is a reference protocol, not proof of asymptotic convergence. No test evaluation is performed. The command can be expensive; choose samples/epochs according to available compute.

## Cheap offline topology experiments

NAS-Bench-201 is now maintained through [NATS-Bench](https://github.com/D-X-Y/NATS-Bench). The optional adapter uses its topology search space (`tss`) and a local benchmark file or unpacked simple archive:

```powershell
python -m pip install -r requirements-benchmark.txt
python benchmark_search.py --benchmark path/to/NATS-tss-v1_0-3ffb9-simple --budget 300 --population 20 --seeds 0 1 2 3 4
```

Download the benchmark data using the official repository instructions; the adapter does not download multi-gigabyte archives automatically. It applies the search policies to six cell-edge choices rather than the project's 13-gene CNN space. It uses mean stored CIFAR-10 validation accuracy (`cifar10-valid`) and records simulated training time. It does not use stored test scores for selection. Mean stored scores remove much of the evaluator noise, so these experiments do not reproduce noisy live training. Benchmark results must be validated in the real project before claiming transfer. The local adapter tests use a synthetic table; the full benchmark archive was not available during verification.

## Ablations, sweeps, and figures

```powershell
python run_ablations.py --device cuda --budget 200 --seeds 42 43 44
python run_hyper_sweep.py --device cuda --budget 150 --seeds 42 43 44
python demo_replay.py --csv experiments/generation_logs/run_example.csv
python generate_comparison.py --results path/to/ga_full.json path/to/random_full.json --labels GA Random --metric best_val_accuracy --out experiments/comparisons/validation.png
```

Ablations use equal candidate-evaluation budgets even for the smaller population. Sweeps preserve evaluation counts while varying proxy training epochs, so they do not have equal training compute or identical measurement protocols. Both preserve repeated seeds and results in unique directories. Comparison figures read supplied measured JSON files; values are not hardcoded. Newly generated architecture diagrams show all five actual widths, residual block summaries, and an unclipped classifier output. Older figures remain historical artifacts.

## Search space and limitations

The thirteen genes cover depth (2–5 blocks), five filter widths, kernel size, block activation, dropout, normalization, classifier width, pooling, and a global residual toggle. The current space has **839,808 genotypes** and **207,360 effective model configurations** after accounting for unused filter genes and duplicate average/mixed pooling. Genotype sampling remains compatible with the original space: it samples depth uniformly and gives average-style pooling two encoded choices. All search methods share that distribution.

The classifier uses ReLU and produces logits. Networks downsample after every block. Residual locations, arbitrary graph topology, depthwise convolutions, and attention are not searched in the live CNN space. Wider networks can be expensive; there is no default latency objective. Fixed deterministic settings improve repeatability within an environment, but do not guarantee identical results across PyTorch versions, devices, or hardware. CUDA/multi-GPU execution still requires hardware-specific validation.

Trial records survive failures, but interrupted search and training are not resumable in this version. Candidate re-evaluations are charged and retained rather than cached. The historical result archive contains multiple schemas and does not gain correctness merely because the current code is fixed.

## References

- [Random Search and Reproducibility for Neural Architecture Search](https://proceedings.mlr.press/v115/li20c.html)
- [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548)
- [Best Practices for Scientific Research on Neural Architecture Search](https://arxiv.org/abs/1909.02453)
- [NAS-Bench-201](https://arxiv.org/abs/2001.00326) and [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)

Apache License 2.0; see [LICENSE](LICENSE).
