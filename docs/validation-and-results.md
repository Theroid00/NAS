# Validation and measured results

## What was validated

Before cleanup, all 51 tests passed, including optional CUDA inference parity
against the exported Covertype model. The current cleanup adds four focused tests
for shared winner retraining and command-line device defaults. The complete suite
is rerun after the refactor; its result is recorded in [maintenance.md](maintenance.md).

After the subsequent export/provenance/environment changes, all **65 tests
passed** on 2026-10-03 in a fresh isolated GPU environment with the pinned core
and service packages. Real CUDA setup training, saved-model inference, and API
prediction passed independently of the unit suite. See [GPU setup](gpu-setup.md)
and its [recorded evidence](gpu-environment-check.json). These checks do not
replace or change the historical benchmark measurements below.

The current-state audit also checked the original main-suite JSON/JSONL files:

- Exactly 15 searches, 50 evaluations each, and 750 `ok` candidate outcomes.
- Every candidate recorded `cuda:0`, 20,000 training rows, 10,000 validation rows,
  and ten proxy epochs.
- One dataset metadata value, one proxy-subset protocol, and one dependency
  environment across the suite.
- Matching initial candidates and per-position training seed schedules across
  GA, aging, and random search for each search seed.
- Each saved historical winner matched the earliest highest-fitness valid trial.
- Saved accuracies matched diagonal/total confusion counts, and balanced accuracy
  and macro F1 recomputed consistently from those matrices.
- Each of 15 full results used its searched chromosome, seed 101, CUDA, twenty
  epochs, and the same dataset metadata; its best validation score matched history.
- Exactly one full-trained model contained a test score: the validation-selected winner.
- Rebuilding the comparison report matched the saved per-run records, summaries,
  and selected model. The exported weight checksum matched its manifest.

These are checks against persisted records, not a second full 750-candidate
training run. The earlier six-search pilot received independent real GPU replay
and scikit-learn/SciPy score verification in [scoring-audit.md](scoring-audit.md).
The current core metric tests independently compare against scikit-learn on
imbalanced predictions and uneven batches.

## Main Covertype suite

The recorded suite is `20261002_183014_bdbb5671`. It compared GA, aging evolution,
and random search for search seeds 42–46, with 50 evaluations each, population 10,
ten proxy epochs, and full winner retraining for twenty epochs with seed 101.
The test set was withheld during comparisons. Fixed MLP training was not included.

| Method | Mean proxy accuracy | Mean full validation accuracy ± sample standard deviation | Full balanced accuracy | Full macro F1 |
| --- | ---: | ---: | ---: | ---: |
| GA | 77.41% | 91.66% ± 2.03 percentage points | 85.18% | 86.74% |
| Aging evolution | 77.71% | 92.53% ± 0.93 percentage points | 86.62% | 87.90% |
| Random search | 76.75% | 90.97% ± 1.63 percentage points | 83.76% | 85.53% |

Each row averages five search winners; full training has one seed per winner in
this suite. The standard deviation is variation across those five search seeds,
not a confidence interval and not variation across multiple training seeds for
one architecture. Aging's mean full accuracy is 1.57 percentage points higher
than random's here; this is a bounded dataset/protocol result.

Mean search durations were GA 102.3 s, aging 101.5 s, and random 95.3 s per seed.
Summed recorded search time was 1,496.02 s, and summed full retraining time was
1,386.09 s: about 48 minutes combined. These sums exclude setup/downloads and
later packaging/test evaluation, and are not a promise about another machine.

## Selected model

The highest full validation accuracy selected aging search seed 45, retrained
with seed 101. It has four 128-wide hidden layers, ReLU, zero dropout, LayerNorm,
no residual addition, and 58,503 parameters. Its full validation accuracy is
94.10%. It was selected using validation before test evaluation.

| Held-out test metric | Value |
| --- | ---: |
| Accuracy | 94.09% |
| Balanced accuracy | 89.47% |
| Macro F1 | 90.49% |
| Test examples | 116,203 |

GA and random do not have comparable final test means in this suite; only the
one selected model was tested. Do not relabel the method validation means as test
results or imply that all 15 models were evaluated on the held-out test set.

## Optional inference timing

The exported selected model was measured on the RTX 4060. After ten warmups,
100 repeated requests gave median in-process latency about 0.65 ms for one row
and 0.71 ms for 32 repeated rows. It includes preprocessing, device transfer,
forward pass, softmax, and response construction; it excludes HTTP transport.
The repeated input is a training example and establishes no accuracy claim.
Cold-start latency and concurrent service throughput are not measured by this test.
The live endpoint was checked to report `cuda:0`.

## Test coverage map

| Test module | Main guarantees |
| --- | --- |
| `test_search.py` | Valid chromosome/budget, archive correctness, seed independence, fatal failure propagation |
| `test_aging_comparison.py` | Aging FIFO behavior, expressed mutation, matched repeatable comparison budgets |
| `test_resume.py` | Replay equivalence for all methods, interrupted generations/batches, journal tails, corruption, dependencies/config/data checks, process locks, abrupt death, comparison write gaps, real CUDA recovery |
| `test_datasets.py` | Stratified disjoint splits/subsets, numeric-only scaling, unchanged indicator columns, seven-class gradients, saved dataset/dimension checks |
| `test_training.py` | Parameter estimates, forward/backward shapes, optimizer/checkpoint policy, weighted validation, fatal failures vs candidate outcomes, repeatability, test opt-in, saved-winner rejection |
| `test_metrics.py` | Independent scikit-learn metric parity, uneven batch weighting, absent classes and zero predictions |
| `test_benchmark.py` | Optional NATS topology encoding and repeatable method budgets |
| `test_reporting.py` | Synthetic vs measured labeling, historical progress, validation-based selection |
| `test_serving.py` | Saved preprocessing/checkpoint parity without dataset loading, input errors, corrupt weights/scales, binary columns, source labels, bundled example/report compatibility |
| `test_portfolio.py` | End-to-end finalization, repeated finish without repeated test evaluation, smoke rejection |
| `test_gpu_serving.py` | Optional real exported-model API/checkpoint parity on CUDA |
| `test_entrypoints.py` | Shared winner retraining preserves dataset/split, explicit override, invalid-winner rejection, shared CLI delegation, CUDA default and explicit CPU choice |
| `test_export_recovery.py` | Interrupted/staged export retry, real process-kill recovery, preservation of existing artifacts, and promised-example startup checks |
| `test_provenance.py` | Startup hashes, changed/added/removed source detection, source-mismatch recovery rejection without saved-record mutation, and mid-suite guards |
| `test_environment.py` | Exact pinned-version checks and dependency-drift rejection |

Tests use offline Breast Cancer or synthetic Covertype fixtures, not a Covertype
download. Optional service tests are skipped when service dependencies are absent.
GPU tests require CUDA; exported-model parity additionally requires an artifact path.

```powershell
python -m pip install -r requirements-service.txt
$env:NAS_TEST_ARTIFACT = "PATH_TO_EXPORTED_COVERTYPE_ARTIFACT"
python -m unittest discover -s tests -v
python -m pip check
```

CI installs pinned CPU and service dependencies on Windows, runs unittest, then
a smoke comparison. It does not prove that the NVIDIA driver/CUDA path works on
an arbitrary GPU machine. That path was checked locally. The optional Docker
image has not been built because Docker is not installed on this development host.

## Evidence boundaries

There is one main dataset, five search seeds, one full-training seed per winner,
and one held-out test model. No general evolutionary-search superiority, proxy
ranking guarantee, production SLA, or cross-dataset conclusion follows from that.
A dedicated Covertype proxy calibration and broader baselines would be useful
for research, but are not required to understand the current working system.
The benchmark records remain unchanged during code cleanup, including their
original environment/source information and machine-local paths.
