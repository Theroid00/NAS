# Existing optional inference API

This documents the already implemented delivery path. It is optional; core
architecture search and comparison do not require running a web service. The
current cleanup adds no dashboard/report features.

## Startup and model lifecycle

`serve.py` loads `serving.api.create_app` and starts Uvicorn on localhost port 8000.
The CLI defaults to CUDA. The reusable app factory and `Predictor` library default
to CPU for offline fixtures; the CLI passes its device explicitly.

During lifespan startup, `Predictor` reads the manifest, verifies artifact/schema
versions and dimensions, validates preprocessing, checks weights SHA-256, builds
the MLP, loads weights with `weights_only=True`, moves it to the requested device,
and enters evaluation mode. Startup fails for corrupted/incompatible artifacts
or unavailable requested CUDA. If a report is supplied, its version is checked
and its dataset must match the served model. No training dataset is loaded.

The model remains in memory across requests and is released at shutdown. There
is no live model-switch endpoint, training endpoint, or background search worker.
Restart with another artifact to change the served model.

## Routes

| Method and path | Behavior |
| --- | --- |
| `GET /health` | Status, whether a model is loaded, and its actual parameter device; works without a configured model |
| `GET /model` | Artifact manifest with architecture, input/class names, preprocessing, checksums, and stored diagnostics; 503 if no model |
| `POST /predict` | Validates and scores raw feature rows; returns predictions/device/timing; 503 if no model, 422 for invalid input |
| `GET /example` | Optional bundled raw training-row request; 404 if not configured/advertised |
| `GET /report` | Reads the configured JSON report; 404 if none |
| `GET /` | Existing static dashboard HTML |
| `GET /dashboard.js` | Existing dashboard script |
| `GET /docs` | FastAPI-generated interactive API documentation |
| `GET /openapi.json` | FastAPI-generated API schema |

The health endpoint does not execute a forward pass. Startup now checks every
advertised example for existence, checksum when present, and valid request shape.
A partial artifact with a missing promised example fails startup rather than
reporting healthy and later failing `/example`. New exports publish only after
these checks. Removing files externally after startup is outside that guarantee.

## Prediction input

The body has one field, `features`, containing a JSON array of rows. Additional
fields are rejected. Values must be numbers, not strings or booleans, and must be
finite. There must be 1–1024 rows; each must have the model's exact feature width,
30 or 54. The schema permits rows up to width 54, then the predictor enforces the
specific artifact width. Invalid row length therefore still results in HTTP 422.

```json
{"features": [[3131, 287, 15, 95, 33, 2495, 176, 238, 202, 785, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

This is the bundled Covertype training example, not a general template for another
dataset. Read `/model` feature order. Covertype's first ten columns are numerical
attributes, followed by four wilderness indicators and forty soil indicators.
The last 44 values must each be zero or one. The predictor checks binary values,
but does not enforce exactly one active wilderness/soil category per row.

The caller sends unscaled values. Preprocessing uses float64 arrays to apply
saved `(raw - mean) / scale` on the manifest's numeric columns, then float32
tensors. Indicator columns are preserved. Nonfinite input or values overflowing
the supported float32 range are rejected. There is no missing-value imputation,
categorical text encoding, feature-name lookup, or batch feature reordering.

## Response

Each result contains `model_id`, `dataset`, actual `device`, `elapsed_ms`, and a
list of predictions. Each prediction contains zero-based `class_id`, dataset
`source_label`, human-readable `class_name`, and the probability vector in class-ID
order. Softmax is applied at inference only. Covertype IDs 0–6 map to source
labels 1–7. Breast Cancer uses IDs/labels 0 and 1 for malignant and benign.

Example labels for Covertype are Spruce/Fir, Lodgepole Pine, Ponderosa Pine,
Cottonwood/Willow, Aspen, Douglas-fir, and Krummholz. These probabilities are model
outputs, not separately calibrated confidence estimates.

## Errors and timing

Pydantic validation failures are converted to JSON 422 details without echoing
invalid inputs. This matters for NaN/Infinity, whose inclusion in a JSON error
body could itself fail serialization. Predictor `ValueError` also maps to 422.
Startup problems and missing/corrupt files are not converted into successful
predictions. The artifact must be a trusted, complete local export.

Prediction timing begins after synchronizing the selected CUDA device, then
includes preprocessing, device transfer, forward pass, finite-output checking,
softmax, CPU transfer, and response construction. Returning CPU probabilities
waits for GPU computation to finish. The service middleware separately reports
`X-Response-Time-Ms`, including handler work. Logs record method/path/status/time,
not raw features. In-process benchmark timings exclude HTTP and cold startup.

## Local operation and limits

```powershell
python serve.py --artifact PATH_TO_ARTIFACT --device cuda
Invoke-RestMethod http://127.0.0.1:8000/health
$example = Invoke-RestMethod http://127.0.0.1:8000/example | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $example
```

Defaults bind localhost rather than creating a public service. Multiple HTTP
requests can enter synchronous handlers through FastAPI's thread pool; no
dedicated inference queue, concurrency limit, dynamic batching, or production
throughput benchmark is implemented. Authentication, rate limits, input-body
byte limits, production observability, and public deployment are separate work.
The existing Dockerfile installs CPU dependencies and explicitly serves on CPU;
it is optional and unverified locally. GPU model runs in the current host use CUDA.
