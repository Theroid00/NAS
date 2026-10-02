# Pinned GPU setup and verification

The supported installation check targets Windows, Python 3.12, and the RTX 4060
Laptop GPU. `requirements-gpu.txt` pins the complete tested core package set;
`requirements-service.txt` separately pins optional API/test packages. The CUDA
torch wheel is approximately 2 GB, so initial setup can be a substantial download.
The Python requirements do not install an NVIDIA driver. A compatible driver is
still required. Linux and other Python versions are not verified by this check.

## Verified on this host

On 2026-10-03, a newly created `.venv-gpu-check` installed the pinned core and
service packages successfully. `pip check` found no broken requirements. The
verification command passed real CUDA training, exported-model inference, and
an API prediction on the NVIDIA GeForce RTX 4060 Laptop GPU, using Python
3.12.14, PyTorch 2.14.1+cu130, and CUDA runtime 13.0. The complete 65-test suite
passed in that environment, including optional GPU artifact/API tests.

The recorded [environment evidence](gpu-environment-check.json) contains package
versions and the source snapshot used for the check. Its one-epoch training score
is a setup check, not a model-quality benchmark. The historical Covertype suite
and held-out test results were not rerun or changed.

## New isolated environment

From the repository root, using a Python 3.12 installation:

```powershell
python -m venv .venv-gpu-check
.\.venv-gpu-check\Scripts\python.exe -m pip install -r requirements-gpu.txt -r requirements-service.txt
.\.venv-gpu-check\Scripts\python.exe -m pip check
.\.venv-gpu-check\Scripts\python.exe verify_environment.py --device cuda --service
```

The verification script compares installed versions against every pinned core
requirement and (with `--service`) every service requirement. It resolves CUDA,
then trains one small Breast Cancer model for one epoch and validates its output.
This dataset is bundled, so the setup check downloads no training dataset and
does not touch Covertype's held-out test set. A dependency mismatch, unavailable
CUDA, or failed training outcome makes the command fail.

Use a fresh environment rather than installing CPU requirements over an existing
GPU environment. The test environment is excluded from Git; it can be retained
for future checks independently of `.venv`. Successful installation on this host
does not establish compatibility on every driver/OS combination.

## Check an exported model and API

```powershell
.\.venv-gpu-check\Scripts\python.exe verify_environment.py --device cuda --service --artifact PATH_TO_ARTIFACT --out experiments/tabular/gpu-environment-check.json
$env:NAS_TEST_ARTIFACT = "PATH_TO_ARTIFACT"
.\.venv-gpu-check\Scripts\python.exe -m unittest discover -s tests -v
```

The artifact must contain its bundled `example.json`. The script verifies weights
and preprocessing, predicts using CUDA, and, with `--service`, checks the same
example through the API. It records actual device, GPU name, runtime/package
versions, startup source fingerprint, and completed checks. `--out` is optional;
without it the evidence is printed as JSON. This is installation evidence, not
a rerun or replacement of the historical benchmark report.

The unit suite includes CPU fixtures for portable CI and dedicated local CUDA
checks. The real setup-training check and exported-model/API checks use the device
requested explicitly. Omitting `NAS_TEST_ARTIFACT` skips the optional exported-model
GPU integration test; omitting service dependencies skips the optional API tests.

## Source provenance and recovery boundary

New searches record source-file hashes before their first evaluation. New
comparison manifests record the same map and check it before each method;
full-training results record their own startup map. Documentation/UI/output-only
changes are excluded, while edits/additions/removals of covered Python source
files are detected even when they have not been committed.

Resume now requires individual search format 3 and comparison format 2. An older
run's completed results remain readable and can be exported/packaged, but recovery
must use its original implementation. No source snapshot is invented retroactively
for an older run. New source mismatches stop recovery before rewriting its records.
