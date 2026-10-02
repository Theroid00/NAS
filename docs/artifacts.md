# Saved files, exports, and provenance

## File relationships

An individual search writes four main files: a trial JSONL journal, metadata JSON,
winner JSON, and (for GA/aging) generation CSV. They use a unique timestamp/UUID
run ID. A comparison creates a unique parent directory, then a subdirectory per
method and search seed. Full-training checkpoints/results live beside that search.

```text
experiments/tabular/comparisons/<comparison-id>/
  comparison.json
  comparison.lock
  ga/42/
    metadata_<search-id>.json
    metadata_<search-id>.lock
    trials_<search-id>.jsonl
    run_<search-id>.csv
    best_<search-id>.json
    ckpt_<search-id>_seed101_best.pt
    full_train_<search-id>_seed101.json
  aging/42/...
  random/42/...
```

The example illustrates naming, not a promise that every run has retraining or
a generation CSV. Default individual-search logs and winners use separate
`generation_logs/` and `best_architectures/` directories. New tabular experiments
and `.pt` weights are excluded from Git. The versioned `docs/*.json` results are
selected historical records, not the only source of experiment data.

## Trial journal: `trials_<id>.jsonl`

Each line is one JSON record flushed and fsynced before its corresponding metadata
update. Successful/completed trial IDs advance sequentially from 1. Fatal attempts
use the next uncompleted ID; retries can therefore leave multiple `error` records
with that same ID before one completed record. Fitness is unset for fatal errors.

| Fields | Meaning |
| --- | --- |
| `trial_id`, `generation` | Evaluation position and optional controller generation/cycle |
| `chromosome`, `arch` | Encoded and decoded architecture; checked for consistency on replay |
| `seed`, `split_seed` | Candidate initialization/shuffle seed and dataset partition seed |
| `dataset_name`, `proxy_size`, `validation_size`, `proxy_epochs` | Data and schedule choices |
| `device` | Resolved execution device, such as `cuda:0` |
| `status`, `error` | Outcome category and optional failure text |
| `fitness`, `validation_loss` | End-of-proxy validation accuracy and sample-weighted cross-entropy |
| `validation_metrics` | Accuracy/loss/sample count plus class-balanced metrics and confusion matrix |
| `num_params`, `training_samples`, `validation_samples` | Model size and observed data sizes for trained candidates |
| `elapsed_s` | Recorded candidate duration, including initialization/training/scoring |

Failed-before-training outcomes do not necessarily contain every trained-candidate
field. Consumers should use status-aware access rather than assume all records
have metrics. This journal, not generation CSV, is the authoritative sequence
used for search recovery and historical-best trajectories.

## Search metadata: `metadata_<id>.json`

Metadata includes the run ID, method, architecture schema, full search space,
configuration, smoke flag, Python/platform/dependency versions, Git revision,
resolved device, status, paths, and resume format version. Real searches additionally
save dataset metadata and proxy protocol hashes. Progress updates include counts,
elapsed time, and accumulated evaluation seconds. Recovery adds a resume history.

The current `resume_version` is 2. A metadata file alone is insufficient to recover:
its journal and data must also be available. Exact dependency versions, dataset
provenance, subset hashes, original configuration, and replay decisions are checked.
Source hashes are not enforced at resume startup; a Git change is recorded in
resume history rather than automatically rejected. Controller divergence during
replay is rejected, but changing the training implementation alone could mix old
and new evaluations without such divergence. Keep one training implementation
throughout a run.

`elapsed_s` excludes intentional gaps between invocations. A hard kill can lose
the unrecorded time of its active candidate. `total_evaluation_seconds` sums saved
attempt durations; controller/record-writing overhead can make wall-clock search
time different. CSV generation durations have their own rounding and replay rules.

## Winner: `best_<id>.json`

The winner contains `best_chromosome`, `best_arch`, `best_fitness`, best trial ID,
best validation loss/metrics, parameter count, evaluation count, accumulated time,
method, dataset, schema/search space, smoke flag, configuration (`hyperparams`),
and paths to metadata/journal/CSV. Its winner policy is best observed validation
accuracy, preserving the earliest exact tie.

`load_winner` rejects smoke/failed/injected-evaluator results, incompatible schemas,
unknown datasets, changed search spaces when present, and inconsistent encoded vs
decoded architecture. `latest_winner` searches one directory, not a recursive
comparison tree. Always pass an explicit winner path when several experiments
exist and you want a specific run.

## Generation CSV

`GenerationLogger` writes `gen`, best/mean/std/min fitness, best chromosome, and
elapsed seconds. Fitness values are rounded to six decimals and time to two
decimals for display. GA rows summarize that evaluated generation; aging rows
summarize its current population. These curves can decrease while the historical
archive remains unchanged. Random search does not create this CSV. A journal can
be more precise than CSV and is required to validate exact ties or resume.

## Comparison manifest: `comparison.json`

The manifest records methods, population, proxy/full epochs, subset sizes, device,
parameter limit, candidate budget, smoke status, split seed, search seeds,
full-training seeds, runs, summaries, and suite status. Its comparison resume format
is version 1, distinct from individual search metadata version 2.

Each run row points to its winner and records proxy fitness/metrics, counts/time,
and completed full-training result paths. Updates occur after each search and
retraining. A completed output that exists before its row was committed can be
recovered. Completed rows are skipped on suite resume; they do not receive the
same full revalidation as individually replaying every search journal.

Method summaries aggregate proxy winners across search seeds. Full-validation
accuracy first averages repeated retraining seeds within one search winner, then
averages those per-search means. Test scores are never included in these method
summaries or used to select a search winner.

## Full-training result and checkpoint

`full_train_<id>.json` contains chromosome/architecture (or an optional baseline
marker), schema, dataset metadata, run/training/split seeds, parameter count,
best validation accuracy/loss/metrics, epoch history, elapsed time, paths, and
protocol. Protocol includes optimizer, learning rate, weight decay, epochs, batch
size, train/validation sample counts, checkpoint policy, and resolved device.

`test_accuracy` and `test_metrics` are null when withheld. Explicit test evaluation
fills them by atomically replacing this result JSON. The `.pt` file stores the
model `state_dict` for the chosen best-validation epoch, not optimizer/scheduler
state. It is a trusted local PyTorch weights file loaded with `weights_only=True`.

Full-training JSON snapshots are atomic; epoch checkpoint writes use `torch.save`
directly. Interrupted training therefore is not an epoch-recovery mechanism.
Existing checkpoint/result names cause a distinct run-ID suffix instead of silent
overwrite. Partially trained weights are not reused after a restart.

## Dataset fingerprints

Dataset metadata includes dimensions, split seed/policy, sample counts, class
counts, scaler columns/means/scales, a raw data/labels SHA-256, and SHA-256 values
for train/validation/test index arrays. Breast Cancer also stores full index lists;
Covertype indices are reproducible from ordering and seed instead of embedded.
Proxy protocol separately hashes its fixed training/validation row indices.

The data hash covers byte representation as well as values and order. Use the same
dataset and environment when reproducing it. Saved source labels differ from model
class IDs for Covertype. Metric confusion matrices use model class IDs.

## Optional portable inference artifact

`export_model.py` creates a new directory containing:

| File | Content |
| --- | --- |
| `model.pt` | Copied best-validation weights |
| `manifest.json` | Artifact version 1, schema 3, dataset/dimensions, architecture, names/labels, preprocessing, checksums, model ID, parameters, and validation/test diagnostics |
| `example.json` | Optional raw training-row request; present only when supplied during export |

The manifest's scaler is enough to preprocess predictions without the original
dataset cache. A request is `{"features": [[...], [...]]}` in exact feature order;
rows contain raw values, not values already standardized by the caller.
`Predictor` checks the manifest dimensions/scales, verifies weights SHA-256,
loads the model on the requested device, applies scaling, and returns probabilities,
model/source labels, model ID, device, and elapsed milliseconds. CUDA output is
transferred back to the CPU before JSON response construction.

These exports can be moved independently from training runs. They are ignored by
Git; the repository currently provides no downloadable model release. Export is
not transactional across the directory: a failed export can leave a partial
directory that blocks retry. See the maintenance guide; this cleanup does not fix it.

## Optional report and delivery record

The existing dashboard report includes config, trajectories, per-run full metrics,
method means/sample standard deviations, and a validation-selected model. Its
JSON is viewable independently once generated, but regenerating it requires
the local winner/journal/full-result files referenced by the comparison.

Portfolio finalization enriches it with environment versions, training-source
hashes captured **at packaging time**, artifact path, example training-row index,
and an in-process inference benchmark. A `delivery.json` beside the suite points
to the report/artifact/selected model and records test completion. Raw experiment
paths are absolute machine paths; this preserves local provenance but does not
make a fresh clone immediately runnable with those paths.

The plain report generator does not preserve enrichment. Existing historical
source hashes describe packaging-time files; cleanup legitimately changes some
files later. Do not overwrite historical hashes to make the current checkout look
identical to the benchmark's recorded source. Use Git history and the cleanup log
to distinguish source revisions from changes to numerical results.
