# Complete project review — 2026-10-03

Branch: `resume/resume-trial`. This review covers maintained source, tests,
installation, CI, documentation, and local historical results. The direction
remains an AI/ML engineering portfolio. Dashboard/report enhancements and new
research experiments remain deferred. Detailed module guides start at
[README.md](README.md); verification is in [project-audit-evidence.json](project-audit-evidence.json).

## Assessment and fix

The project implements a working bounded MLP architecture-search workflow for
tabular classification. Common evaluation, independent search/training randomness,
strict budgets, streaming metrics, durable trial records, deterministic recovery,
explicit GPU selection, and saved inference preprocessing are its strongest
engineering features. No scoring defect, test-selection leakage, budget overrun,
or mismatch between the main result records and published summary was found.
The result supports this specific dataset/protocol, not a universal evolutionary
advantage or a guarantee of finding the global optimum.

One source-consistency gap was fixed. Comparison source checks previously ran
before searches but not before each winner retraining. They now run before every
full-training repeat. Recovered/new full-training results must also match the
comparison's source snapshot before being accepted. Two regression tests cover
an edit after search and a recovered training record with a different snapshot.
The training, scoring, and historical results were not changed.

## Algorithms and model construction

The shared schema encodes depth, four hidden widths, activation, dropout,
LayerNorm, and residual additions. Decoding rejects malformed chromosomes and
non-integer/out-of-range indices. The model builder uses dataset dimensions;
parameter estimates include biases and LayerNorm and are checked against actual
models. Residual addition requires matching adjacent dimensions.

GA uses tournaments, crossover, per-gene mutation, elitism, and a stagnation-based
mutation increase. Its final partial generation respects the remaining budget.
Aging uses FIFO replacement and changes one expressed choice per child. Random
samples the same chromosome space. All use `SearchSession` and the same evaluator.
The archive retains the earliest valid highest-accuracy candidate. Candidate
OOM/divergence/parameter-limit outcomes consume budget but cannot win; fatal
infrastructure errors stop the run. Local search randomness is unaffected by
training-seed resets.

Candidate budgets do not equal wall-clock budgets. Models have different costs,
failed candidates count, and execution is sequential on one GPU. Inactive widths
and ineffective residual flags can produce duplicate expressed models. GA
mutation can retain an existing gene value. These are known protocol properties;
deduplication or changed sampling would require a new experiment.

## Data, training, and scoring

Fixed stratified 60/20/20 splits separate training, validation, and test. The scaler
fits training rows only. Covertype scales ten numeric columns and preserves its
binary indicators. Proxy subsets are fixed across methods; data/split/subset
hashes, class counts, dimensions, and preprocessing statistics are recorded.

Candidates share Adam and cosine learning-rate scheduling. Proxy fitness is final
proxy-epoch validation accuracy. Full training uses all training rows and the
earliest best validation checkpoint. The increase from roughly 77% proxy scores
to roughly 92% full validation scores reflects different data/epoch/checkpoint
protocols, not relabelling validation as test or adjusting arithmetic.

Validation disables gradients and training behavior. Accuracy uses correct/total;
loss is weighted by samples. Confusion matrices produce per-class scores,
balanced accuracy, and macro F1. Balanced accuracy includes target-present classes;
macro F1 includes every configured class, assigning zero where needed. Independent
scikit-learn tests cover class imbalance, absent classes, and uneven batches.

Comparison retraining withholds test. The final model is selected by full
validation before its test evaluation. Standalone retraining permits immediate
test unless `--validation-only` is passed; comparisons should use the documented
validation-only workflow. The checkpoint test command rejects a second recorded
evaluation, but does not prevent users from creating other experiments.

## Recovery and provenance

Trial journals are flushed/synced per candidate. JSON uses temporary files and
replacement. OS locks prevent concurrent cooperating writers and release after
process death. Recovery replays the deterministic controller, checking trial
order, chromosomes, seeds, device, protocol, dataset, and core dependency versions.
An incomplete final journal fragment is backed up before removal; other corruption
is rejected.

Search format 3/comparison format 2 record source-file and aggregate hashes.
Uncommitted changes and source additions/removals are detected. Recovery rejects
changed source before rewriting saved records. Hashes capture disk files, not
runtime monkeypatches or driver state. Byte hashes include line endings; a
logically identical checkout with different bytes can fail recovery.

Completed trials recover; partial epochs/full training restart. A partial
full-training checkpoint can cause a fresh run-ID suffix and leave an unused
checkpoint. Raw paths remain machine-local. Older interrupted formats need their
original implementation. Completed historical records remain readable/exportable.

## Export and optional serving

Exports validate schema, architecture, dimensions, weights, and preprocessing in
a unique sibling staging directory, then publish under a destination lock by
rename. Existing targets are preserved. Normal interruptions clean staging; hard
kills can leave an unused hidden folder while allowing retry at the same target.
Promised examples are required and validated at startup; new examples have a
checksum, while valid legacy examples remain compatible.

Inference verifies the weight checksum, loads with `weights_only=True`, and
applies the saved scaler without loading training data. Input validation checks
shape, row bounds, finite values, and Covertype binary columns. API validation also
rejects strings, booleans, and extra fields. Real CUDA saved-model inference and
TestClient prediction passed. This validates the application path, not external
HTTP networking, concurrent throughput, or production deployment.

The API is a local demonstration. Auth, production monitoring, proxy body limits,
and concurrency testing remain outside current scope. Health confirms startup,
not a prediction per request. The known plain-report overwrite/enrichment issue
remains deferred; this audit rebuilds the report in memory and never overwrites
the historical published results. Existing dashboard/report code was not expanded.

## Installation, CI, and Git delivery

The fresh isolated Windows/Python 3.12 GPU environment passed exact core/service
pins, `pip check`, real training, inference, and API prediction on the RTX 4060
Laptop GPU with CUDA runtime 13.0. Broad `requirements.txt` is a convenience
specification, not the verified environment lock. Driver/OS portability is not
established by one successful host.

CI defines Windows CPU tests and a synthetic smoke comparison. Local GPU checks
do not establish that a remote Actions job passed. The optional CPU Docker image
remains unbuilt; Linux compatibility was not verified. Python syntax and local
documentation links passed. Environments, data cache, weights, new experiments,
and untracked `analysis/` material are excluded from Git delivery. Historical
CIFAR outputs remain intentionally tracked and documented as older-schema results.
A basic credential-pattern check found no matches; this is not a formal security
audit or examination of every historical Git object.

## Independent result verification

The original Covertype suite was checked without retraining its candidates:

- Fifteen searches, fifty successful evaluations each: **750 records**, all CUDA.
- Identical dataset/subset provenance; 20,000 training rows, 10,000 validation
  rows, ten proxy epochs; paired initial candidates and per-position seeds.
- Archive winners agree with earliest best valid trials and decoded architectures.
- Accuracy, balanced accuracy, and macro F1 agree with independently recomputed
  confusion counts for all candidates and full winner/test metric records.
- Fifteen retrainings match chromosomes, dataset, seed 101, twenty epochs, and
  best-validation history; exactly one validation-selected model has test metrics.
- In-memory report reconstruction matches every core historical report field;
  exported weights match their checksum; the historical result file is unchanged.

| Method | Mean full validation accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| GA | 91.66% | 85.18% | 86.74% |
| Aging evolution | 92.53% | 86.62% | 87.90% |
| Random search | 90.97% | 83.76% | 85.53% |

The selected aging model retains 94.10% validation and 94.09% test accuracy. Five
search seeds, one dataset, and one full-training seed per winner limit broader
claims. GA/random validation means are not test means. Persisted-count checks
verify record consistency, not recreated historical predictions. Earlier GPU
replay evidence remains in [scoring-audit.md](scoring-audit.md).

All **67 tests passed** with optional CUDA artifact/API tests enabled. A fresh
real GPU integration comparison ran all three methods on the bundled dataset,
four evaluations each, one seed, and one proxy/full epoch. Its completed recovery
reused results without retraining; test remained withheld. This checks workflow
correctness and is not an additional meaningful algorithm benchmark.

## Remaining priorities

No identified issue blocks code review or presenting this bounded industry
project. Review and merge when satisfied. Further features should follow an actual
requirement: another shared objective, dataset ingestion, artifact relocation,
epoch recovery, or a deployment target. Numerical changes require a new suite.
Dashboard/report polish and research expansion remain optional and deferred.
