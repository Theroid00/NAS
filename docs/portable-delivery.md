# Portable experiments, examples, and inference delivery

Updated: 2026-10-05. This guide describes the implemented delivery stages of the
[roadmap](improvement-roadmap.md). Public hosting/model upload and larger optional
pipeline changes remain separate. Training examples default to the single GPU.

## Short real GPU example

Use the [pinned GPU environment](gpu-setup.md), then run:

```powershell
python quickstart.py --device cuda
```

This uses bundled Breast Cancer data, four candidate evaluations per method,
one search seed, one proxy epoch, and two full-training epochs per winner. It
withholds test, selects by validation, and exports a model with a raw example.
It prints comparison and artifact paths. This takes seconds on the verified
RTX 4060 setup and tests integration; its scores are not a new algorithm benchmark.
For a machine without CUDA, choose `--device cpu` explicitly.

Prediction does not train or download a dataset:

```powershell
python predict_example.py --artifact PATH_TO_ARTIFACT --device cuda
```

Add `--input REQUEST.json` for another request with raw numeric `features` in
the manifest's feature order. See [input contract](inference-api.md).

## Portable record contract

New search metadata, winner records, comparison manifests, and full-training
results have `path_format: 1`. Known path fields are stored relative to the JSON
file containing them, using `/` separators. `utils.records.read_record` resolves
them to this record's current location. `write_record` leaves caller dictionaries
unchanged. Python return values retain usable local paths.

Covered references: `trial_path`, `metadata_path`, `save_path`, `log_path`,
`checkpoint_path`, `results_path`, `winner_path`, and `save_dir`, including
references nested in comparison rows. Copy the complete experiment tree, including
journals/checkpoints and any output directories referenced with `..`. Keep linked
outputs on the same filesystem drive; the writer rejects cross-drive references.
Generated presentation reports and delivery summaries are not converted to the
experiment format; historical report paths remain as recorded.

Example using the comparison directory printed by quickstart:

```powershell
python relocate_experiment.py --source ORIGINAL_EXPERIMENT --destination MOVED_EXPERIMENT
python run_comparison.py --resume MOVED_EXPERIMENT/comparison.json
```

Relocation copies into staging, rewrites experiment records, and publishes to a
new destination. It preserves the source and refuses existing destinations,
destinations inside the source, symbolic links, and references outside an explicit
legacy root. New relative records can also be moved directly with normal file
copying. They need no rewrite if the entire referenced tree moves together.

Legacy records without `path_format` still read with their previous absolute or
current-working-directory-relative semantics. For an explicit legacy copy, use
`--legacy-root ORIGINAL_ABSOLUTE_ROOT`; foreign Windows absolute roots can be
mapped on Linux. Legacy relative references need their original working-directory
context so they resolve correctly; do not guess a missing original root. This
tool migrates paths, not source identity. Historical interrupted runs still require
matching original source/dependencies and recovery formats. Relocation does not
override hashes or permit resuming after a code edit. Completed older winners can
be read/exported or retrained with a new explicitly recorded source snapshot.

## Versioned model bundles

Create a release-ready bundle from a validated artifact:

```powershell
python model_bundle.py pack --artifact PATH_TO_ARTIFACT --out artifacts/releases/model-v1.zip
```

The output contains exactly `manifest.json`, `model.pt`, and `example.json`.
A `.release.json` sidecar records model ID, dataset, metrics, size, and archive
SHA256. Identical artifacts produce identical ZIP bytes. Existing output names
are refused. This prepares files locally; it does not publish a release.

Install a local archive or a specific HTTPS release URL:

```powershell
python model_bundle.py install --archive model-v1.zip --sha256 VERIFIED_SHA256 --destination artifacts/model-v1
python model_bundle.py install --url PINNED_HTTPS_RELEASE_URL --sha256 VERIFIED_SHA256 --destination artifacts/model-v1
python predict_example.py --artifact artifacts/model-v1 --device cuda
```

Require a trusted full SHA256 digest, not an unversioned `latest` URL. The installer
checks downloaded bytes, limits archive/expanded size to 50 MiB, rejects extra or
traversing filenames, and validates weights/preprocessing/example before publishing
the destination. Existing targets are preserved. HTTPS redirects must stay HTTPS.

The existing Covertype winner was packaged and installed locally without
retraining. Local bundle: `artifacts/releases/covertype-v1.zip`, 222,657 bytes.
SHA256: `ca5da1d3d572faa6f2ad5ac8e7c13909e7d83183370ed49be717b439ebd20c9a`.
Its source snapshot remains unavailable in the legacy training result; packaging
does not invent a historical training hash. Its model ID and published metrics
remain unchanged. No public download URL exists yet. Public release/upload is
separate from local release preparation; weights remain excluded from Git.

## Docker inference

The image uses an explicit Python 3.12 base version, pinned CPU/service packages,
non-root UID 1000, and a model readiness healthcheck. The artifact is mounted
read-only rather than embedded in the image. It uses CPU for this small model's
inference; search and training remain on the laptop GPU.

```powershell
docker build -t nas-inference .
$artifactPath = (Resolve-Path artifacts/model-v1).Path
docker run --rm --name nas-api -p 8000:8000 --mount "type=bind,source=$artifactPath,target=/models/selected,readonly" nas-inference
```

Open `/docs` at `http://localhost:8000/docs` for the existing API. The health probe
checks `/health` and requires `model_loaded`. The default `NAS_ARTIFACT` is
`/models/selected`; a missing/invalid artifact fails startup. Use `NAS_ARTIFACT`
to select another mounted path. Port 8000 is used by both service and healthcheck;
map a different host port with `-p HOST_PORT:8000`. The image's default device is
CPU and it contains no GPU runtime or trained weights.

On Linux, exported directories are private to their owner by default. Match that
owner's non-root UID/GID when mounting them instead of broadening permissions:

```bash
docker run --rm --user "$(id -u):$(id -g)" -p 8000:8000 -v "$PWD/artifacts/model-v1:/models/selected:ro" nas-inference
```

The CI fixture and API run use the same host UID/GID. This preserves read access
without making model directories world-readable. Windows bind-mount permissions
are managed by Docker Desktop; a permission error must be resolved before startup.

The Docker ignore rules exclude local environments, raw experiments, weights,
datasets, docs/tests, and `analysis/`. The Ubuntu CI job builds the image, creates
a tiny offline CPU fixture inside it, serves the read-only artifact, checks real
HTTP readiness/model metadata/prediction parity/invalid inputs, restarts, and
rejects missing/corrupt artifacts. This CPU fixture is a deployment test, not a
replacement for GPU architecture search.

Local limitation: Docker and usable WSL are unavailable on this host. The fixture
and HTTP verification scripts can be tested here against the local API, but this
does not prove Linux build/runtime compatibility. The CI job provides that check
when run. Its status must be reported separately; configuration alone is not
evidence that a container passed. No Docker installation or public deployment is
performed by these scripts.

## Implementation verification

All 73 tests passed in the isolated GPU environment. The real CUDA quickstart
completed and its moved experiment resumed. Tests also copied a real experiment,
removed the original temporary fixture, then recovered, retrained, and exported.
Selected-model bundle installation preserved CUDA predictions. The HTTP smoke
script passed against the local CPU service. Legacy report reconstruction still
matched the historical snapshot. Evidence: [delivery-validation.json](delivery-validation.json).
