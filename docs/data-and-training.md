# Data, training, and scoring

## Supported datasets and defaults

Dimensions and practical search defaults live in `data/specs.py`, not in the
model builder or optimizer configuration.

| Setting | Covertype | Breast Cancer Wisconsin |
| --- | ---: | ---: |
| Rows | 581,012 | 569 |
| Features | 54 | 30 |
| Classes | 7 | 2 |
| Training rows | 348,607 | 341 |
| Validation rows | 116,202 | 114 |
| Test rows | 116,203 | 114 |
| Default proxy training rows | 20,000 | All training rows |
| Default proxy validation rows | 10,000 | All validation rows |
| Default batch size | 512 | 32 |
| Scaled columns | First 10 numeric columns | All 30 columns |

The command-line experiment default is Covertype. Library calls keep Breast
Cancer as their default for small offline checks and existing Python callers.
Pass dataset and dimensions explicitly in real experiment integrations.

Covertype uses scikit-learn's cached fetcher. `prepare_dataset.py` is the explicit
download step; normal searches call the fetcher with downloading disabled. The
cache lives in `data/tabular_cache/` and is excluded from Git. Breast Cancer is
bundled with scikit-learn and requires no dataset download. Covertype source labels
1–7 are converted to model labels 0–6; exported inference maps them back to 1–7.

## Splitting and scaling

`_split(split_seed, dataset)` partitions integer row indices. The first stratified
split reserves 40% of rows; a second stratified split divides that held-out part
equally into validation and test. Both use the split seed. Dataset ordering is
part of provenance, so reordering otherwise identical input rows changes hashes.

StandardScaler is fitted on numeric columns of **the full training split only**.
It is applied independently to train, validation, and test features, which are
stored as float32 tensors. Covertype's 44 wilderness/soil indicator columns remain
binary. Targets are integer tensors suitable for cross-entropy.

Proxy subsets are stratified samples within their respective already-separated
training and validation partitions, selected deterministically with the split seed.
The scaler is not refitted for each proxy subset or candidate. Zero subset size
means the full corresponding partition, rather than no rows. An impossible
stratified sample, oversized subset, invalid dataset, or missing cache fails clearly.

Loaders use no worker subprocesses. Only training loaders shuffle, with a dedicated
torch generator seeded by the training seed. Row indices are retained on datasets
for subset fingerprints and training-example extraction. Raw arrays, splits, and
proxy datasets are cached in-process with bounded LRU caches.

## Randomness and device handling

The search controller owns a private RNG. `set_training_seed` separately seeds
Python's global RNG, NumPy, and PyTorch; enables deterministic operations; disables
cuDNN benchmarking; and sets the cuBLAS workspace configuration if absent.
CPU thread count is set to one. Candidate training reseeding cannot alter the
controller's private RNG or the fixed data subsets.

Experiment CLIs default to `cuda`. The core runtime resolves it to `cuda:0` and
rejects unavailable or invalid devices. `--device cpu` remains an explicit choice
for offline tests. One candidate runs on one device at a time. Determinism is
tested in the current environment; identical bits across different hardware or
dependency versions are not promised.

## Proxy evaluation

`training/evaluator.py::evaluate_trial` decodes the genes and estimates model size
before allocating weights. A configured parameter limit can reject a candidate
without loading training data. Otherwise it builds the MLP, checks a dummy forward
pass, creates loaders, and trains from scratch with:

- Adam, learning rate 0.001 and weight decay 0.0001.
- Cross-entropy loss on raw logits.
- Cosine learning-rate decay over the configured proxy epochs.
- A fresh initialization and training shuffle for each trial.

The evaluator scores validation **after the final proxy epoch**, rather than
selecting an intermediate proxy checkpoint. Its return dictionary contains fitness,
validation loss, class metrics, sample counts, parameter count, elapsed seconds,
and status. Nonfinite training loss or validation logits are divergence outcomes;
GPU memory exhaustion is a candidate failure. Other exceptions propagate as
fatal infrastructure/programming failures. GPU cached allocations are released
after an evaluation. Candidate weights are not retained for full retraining.

## Full retraining and explicit testing

`full_train` builds a new model from the searched chromosome; `train_model` owns
the shared loop. It uses the same optimizer settings, a cosine schedule over the
full epoch count, and all training rows. Each epoch records sample-weighted train
loss/accuracy and full validation loss/metrics. A strictly higher validation
accuracy saves a checkpoint, so the earliest tied epoch is retained.

At the end, the saved best-validation weights are restored. Validation metrics
in the result describe that selected checkpoint, not necessarily the last epoch.
When test evaluation is requested, it uses those restored weights. Comparison
suites always pass `evaluate_test=False`.

`train_winner` is the shared bridge from a winner JSON to full training. It checks
schema, dataset, smoke status, and chromosome consistency through `load_winner`.
It uses the saved dataset and split seed unless a split override is explicit.
Both `main.py --mode train-best` and `evaluate_best.py` call this bridge.

The standalone training CLIs historically evaluate test unless `--validation-only`
is supplied. **Use `--validation-only` while choosing architecture/protocol.**
After freezing those choices, use `evaluate_best.py --test-checkpoint ...`.
`evaluate_checkpoint` checks schema/dataset and dataset fingerprint, rebuilds the
model, loads its weights, scores test, and atomically stores the result. It refuses
a checkpoint that already has a recorded test accuracy. This guard is a sequential
CLI safeguard; the standalone command does not implement a concurrent evaluation lock.

## Metric definitions

`validate` switches the model to evaluation mode and disables gradients. Accuracy
is total correct predictions divided by total samples; cross-entropy is the sum
of individual losses divided by samples. Neither is an unweighted mean of batch
means, so a short final batch receives its correct weight.

The streaming confusion matrix has true classes as rows and predicted classes
as columns. For class i, let TP be its diagonal count, S its row sum, and P its
column sum:

| Metric | Formula / treatment |
| --- | --- |
| Recall | TP / S; zero when S is zero |
| Precision | TP / P; zero when P is zero |
| Class F1 | 2TP / (S + P); zero when the denominator is zero |
| Balanced accuracy | Mean recall across classes present in ground truth |
| Macro F1 | Mean class F1 across all output classes, including absent classes |
| Overall accuracy | Sum of diagonal counts / total counts |

`classification_metrics` returns balanced accuracy, macro F1, per-class recall,
precision, F1, support, class IDs, and the confusion matrix. Stored scores are
fractions in 0–1. Search and checkpoint selection continue to use overall
validation accuracy. Loss and class-balanced scores are not hidden tie-breakers.

## Interpretation limits

Full training gives the proxy winner another initialization and more rows/epochs,
so scores can change substantially. Proxy ranking need not match full-training
ranking. The calibration tool measures this agreement using validation only;
the measured main Covertype suite has not received a dedicated proxy calibration.
Validation is reused for selection and therefore is not an unbiased final score.
Only the final frozen model's test score serves that role under this split protocol.
