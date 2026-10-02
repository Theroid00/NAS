# Industry demo

This project demonstrates an end-to-end ML engineering workflow on one RTX 4060:
reproducible model search, durable trial recovery, honest evaluation, portable
inference artifacts, a validated prediction API, and an interactive results viewer.
The parked research plan is in [research-direction.md](research-direction.md).

## Run on the GPU

Use Python 3.12, install `requirements-gpu.txt` and `requirements-service.txt`,
then prepare Covertype once. From an activated virtual environment:

```powershell
python prepare_dataset.py --dataset covertype
python run_portfolio.py --device cuda
```

Defaults compare GA, aging evolution, and random search, five search seeds each,
50 candidates per search, ten proxy epochs, and twenty full-training epochs.
Candidates run sequentially on one GPU. No fixed MLP participates in this suite.
The workflow selects one model by full validation accuracy, evaluates its test
checkpoint once, exports preprocessing plus weights, and saves the results report.
CUDA being unavailable is an error; the command does not silently switch devices.

Recover an interrupted suite or package a completed one without rerunning it:

```powershell
python run_portfolio.py --resume PATH_TO_COMPARISON_JSON
python run_portfolio.py --comparison PATH_TO_COMPLETED_COMPARISON_JSON
```

The packaging command prints the artifact directory. Start its GPU service:

```powershell
python serve.py --device cuda --artifact PATH_TO_ARTIFACT --report docs/industry-results.json
```

Open `http://127.0.0.1:8000`. Compare the three methods, switch the search chart
between candidate count and recorded evaluation time, inspect winner architectures
and class metrics, then click **Use example row** and **Predict**. The bundled row
comes from training, and prediction takes raw features in the manifest's order.
The service uses saved training-only scaling and needs no dataset cache.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$example = Invoke-RestMethod http://127.0.0.1:8000/example | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $example
```

`/health` and `/predict` identify the actual inference device. `/model` describes
the artifact and feature order; `/docs` exposes interactive API documentation.
Invalid dimensions, nonfinite values, nonnumeric features, and nonbinary Covertype
indicator columns receive 422 responses. Artifact checksums and preprocessing
dimensions are checked at startup. Request logs contain timing and status, not features.

## Measured results

[industry-results.json](industry-results.json) saves all 15 search trajectories,
metrics, configuration, source hashes, and GPU inference measurements. The suite
evaluated 750 candidates and retrained 15 winners on the full training split.
Means below are across five search seeds; each winner used retraining seed 101.

| Method | Proxy accuracy | Full validation accuracy | Mean search time |
| --- | ---: | ---: | ---: |
| GA | 77.41% | 91.66% | 102.3 s |
| Aging evolution | 77.71% | 92.53% | 101.5 s |
| Random search | 76.75% | 90.97% | 95.3 s |

The selected aging winner (search seed 45) has 58,503 parameters, 94.10% validation
accuracy and **94.09% test accuracy**, with test balanced accuracy **89.47%** and
macro F1 **90.49%**. Only this validation-selected model was evaluated on test.
Aging's mean is higher in this bounded run; this does not establish general superiority.

On the RTX 4060, warmed in-process inference measured a median **0.65 ms** for one
row and **0.71 ms** for 32 repeated rows (100 requests each). These include scaling,
device transfer, forward pass, softmax and response construction, and exclude HTTP.
They use a repeated training example and are not production latency guarantees.
The first request includes startup overhead. GPU results are synchronized before
returning probabilities, so timing does not measure only kernel submission.

Balanced accuracy averages recall over classes present in ground truth. Macro F1
averages over all model output classes, using zero for undefined scores. Confusion
matrix rows are true classes and columns are predictions. Search still optimizes
validation accuracy; the extra metrics reveal rare-class performance.

## Engineering limits and validation

The unit suite covers scoring, checkpoint restoration, deterministic recovery,
artifact/API parity, request validation, report aggregation, and idempotent finalization.
The original 51 tests passed locally, including GPU checkpoint/API parity against the
exported Covertype artifact. The live HTTP endpoint and dashboard prediction
were also verified to return `device: cuda:0`.
The subsequent code cleanup passes 55 tests; see [current validation](validation-and-results.md).
To also verify the exported model on the GPU without retraining:

```powershell
$env:NAS_TEST_ARTIFACT = "PATH_TO_ARTIFACT"
python -m unittest discover -s tests -v
```

Trial recovery restarts an interrupted candidate; it does not recover partial
optimizer state. Reports preserve original machine paths for provenance; exported
artifacts can be moved independently. Training code was unchanged during the
measured suite; early runs started before its metrics commit and later runs after
it, so per-search Git revisions differ. Packaging also records training source hashes.

The included Dockerfile is an optional CPU deployment image, explicitly configured
with `--device cpu`. Local model experiments and the running demo use CUDA.
Docker was unavailable on the development machine, so the image has not been built
or validated. The service is a local demo; authentication, production monitoring,
and a public deployment are outside the implemented scope.
