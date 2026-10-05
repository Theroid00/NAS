# Industry improvement and containerization roadmap

Updated: 2026-10-05. Implementation status is recorded below; detailed commands
and verification limits are in [portable delivery](portable-delivery.md).
The latest completed baseline is the [project audit](project-audit.md): 67 tests
passed, GPU integration checks passed, and historical benchmark records agreed.
The core project has no identified blocker requiring a broad refactor.

## Goal and scope

Make the project easier for another engineer or recruiter to run, inspect, and
understand. Prioritize portable experiments, a short runnable example, and a
verified inference container. Keep search/training on the user's single RTX 4060.
Keep dashboard/report expansion and research work deferred. Do not change scoring,
sampling, budgets, or historical benchmark results as part of packaging work.

## Proposed order

| Order | Work | Value | Status |
| --- | --- | --- | --- |
| 1 | Review the code and merge the reviewed branch when satisfied | Understand and explain the implementation; establish the release baseline | User review/merge pending |
| 2 | Portable experiment paths | Copy saved runs between machines without broken references | Implemented; moved-tree recovery/retraining/export tested |
| 3 | Runnable example and downloadable selected-model artifact | Demonstrate the project without running the full benchmark | Examples/bundle tooling implemented; local selected-model bundle verified; public upload pending |
| 4 | Verify and complete Docker inference packaging | Reproduce the local API on a clean machine with fewer setup steps | Implemented; Ubuntu image build/runtime and Windows CPU CI passed; local engine unavailable |
| 5 | Optional hosted prediction demo | Give reviewers a public link to try the trained model | Optional; hosting choice pending |
| 6 | README/resume presentation | Explain the problem, design decisions, bounded results, and runnable workflow | README examples linked; resume wording remains optional |

Review and merge are separate Git actions. This roadmap does not record a merge
or authorize replacing `main`. Features should be committed in reviewable stages
on the working branch before their final integration.

## 1. Portable experiment paths

Current comparison, winner, and training records contain machine-local paths.
Exported inference artifacts already bundle weights/preprocessing and do not
need the original training dataset. Portability work primarily targets raw runs.

Implementation plan:

- Introduce one shared resolver for record references instead of repeating path
  handling across comparison, recovery, training, export, and analysis consumers.
- Store new references relative to the containing record or a clearly declared
  experiment root. Specify one convention and use it everywhere.
- Preserve legacy absolute-path reads. Relocating legacy records should require
  an explicit mapping/root option; avoid guessing which checkpoint belongs to a run.
- Document path encoding, including cross-platform separator handling, and version
  records where a format change needs an explicit recovery boundary.
- Keep provenance, hashes, budgets, and journal replay checks intact. Do not edit
  the historical result snapshot merely to replace its paths.

Acceptance: copy a small real experiment to a different directory, remove its
dependency on the original location, then recover, inspect, retrain its saved
winner, and export from the relocated copy. Check ambiguous/missing references
fail clearly, and existing historical records remain readable.

## 2. Runnable example and model release

Provide two clearly separated examples:

1. A brief offline workflow on Breast Cancer Wisconsin: all three search methods,
   tiny equal budgets, validation-only retraining, and output inspection. Default
   the training example to CUDA, with explicit CPU instructions for other users.
2. A prediction-only example using the already trained Covertype winner, its raw
   bundled input, and preprocessing manifest. No dataset download or retraining.

Publish a versioned model bundle containing weights, manifest, and example;
document its checksum, provenance, measured metrics, input order, and limitations.
Keep large/binary weights out of ordinary Git commits. Choose a release asset or
Hugging Face model repository, and implement explicit versioned retrieval rather
than loading whichever model happens to be newest. Add clear missing-artifact and
checksum errors. Public upload/deployment is a separate delivery step.

Acceptance: follow the documented commands in a fresh location, reproduce the
small workflow, and predict from the released artifact. Label tiny-run results
as integration checks, not evidence that one search algorithm is better.

## 3. Docker: useful, with a narrow first scope

Containerization is useful for reproducible API delivery and demonstrates practical
ML deployment. The repository already has a CPU inference `Dockerfile`; it has not
been built or verified. It installs the Windows-tested CPU pins, copies inference
dependencies, runs as a non-root user, and serves on port 8000. It does not package
a trained artifact or provide a search/training container entry point.

The first deliverable should be a verified inference image. CPU inference is
appropriate for this small saved MLP; it does not change the single-GPU training
workflow. GPU container training is a later optional task, not a prerequisite.

Implementation stages:

1. Confirm Docker Desktop/engine availability and Linux-container support.
   Do not describe an unbuilt image as tested.
2. Verify CPU/service dependencies on Linux. Keep the Windows GPU lock intact;
   introduce a separate Linux/container lock if compatibility requires it.
   Pin the base image/release dependencies so container builds are reproducible.
3. Audit imports and retain the modules required by inference. Do not aggressively
   remove modules just to reduce image size before confirming startup behavior.
4. Exclude all local virtual environments, caches, raw experiments, and `analysis/`
   from the build context. Existing `.dockerignore` lists only `.venv`, so expand
   this to cover environments such as `.venv-gpu-check`.
5. Mount an exported artifact read-only at a documented path; configure
   `NAS_ARTIFACT`, `NAS_DEVICE=cpu`, thread count, and port. Preserve non-root
   execution and avoid embedding credentials or user-local paths in the image.
6. Add build/run examples and a readiness check. Verify `/health`, `/model`, and
   `/predict`; missing/corrupt configured artifacts must fail startup.
7. Compare prediction outputs with the local artifact, test invalid inputs, restart
   the container, and confirm inference needs no dataset cache or writable weights.
8. Once local checks pass, add a bounded container smoke check to CI. Keep the
   existing CPU tests; a container build alone is not prediction verification.

Acceptance: a clean machine builds/starts the image, mounts the released artifact,
and returns the same predictions within a declared numerical tolerance. Save actual
build/test evidence and document Linux compatibility limits. The image need not
provide authentication, monitoring, or a production SLA for this portfolio scope.

Optional GPU stage: a separate CUDA image with compatible Linux dependencies,
NVIDIA runtime support, and `--gpus` instructions. Verify on the available single
GPU, mount datasets/outputs explicitly, and test persistent run recovery. A Docker
GPU image does not make free hosted GPU training available automatically.

## 4. Optional Hugging Face deployment

Deploy a small prediction demo using the existing selected model. Architecture
search/full benchmarks stay local. Path improvements are helpful for distribution
but are not prerequisites for inference from a bundled artifact.

Policy checked on 2026-10-05: regular Docker/CPU Gradio Space creation requires a
paid plan even though CPU Basic has no hourly charge. Eligible free personal
accounts (verified email, older than 30 days) may host up to two Gradio ZeroGPU
Spaces. ZeroGPU requires its Gradio/decorator integration and has usage quotas;
the current FastAPI Docker app is not an unchanged free ZeroGPU deployment.
Recheck eligibility and supported dependencies before implementing the adapter.

Sources: [Spaces overview](https://huggingface.co/docs/hub/spaces-overview),
[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu),
[Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).

For a Docker Space, configure the declared application port to match the service
(or adapt it to the platform default), make artifacts available at startup, and
verify permissions. For the free ZeroGPU route, add a minimal Gradio prediction
adapter and test its dependency/GPU compatibility separately. Do not rebuild a
dashboard. A static Space can show saved results but cannot directly host Python
inference. Do not use free transient/quota-limited hosting for the full suite.

Acceptance: sample input produces class probabilities; model identity, input
contract, and measured results are visible; cold startup and invalid input handling
are verified. Record the actual deployed revision. Do not claim deployment is
complete merely because local tests pass.

## Later pipeline improvements, only if needed

| Improvement | Proposed implementation | Required validation |
| --- | --- | --- |
| Resume interrupted full training | Persist model, optimizer, scheduler, completed epoch, RNG states, best checkpoint, and protocol/source identity atomically | Interrupt/recover and compare with uninterrupted training; reject mismatched state |
| Custom CSV datasets | Explicit target/features schema, stratified split, training-only preprocessing, missing/categorical handling, feature order and data hashes | Leakage, invalid schema/classes, inference preprocessing parity; first dataset chosen deliberately |
| Configurable search objective | Shared explicit objective for accuracy, balanced accuracy, or macro F1; record it in all relevant formats and winner policies | Consistent selection across methods and recovery; rerun benchmark under the new declared objective |
| Broader comparison | More datasets and repeated full-training seeds; optional proxy-rank calibration | New independently labelled results; keep validation/test selection separated |

These are optional expansions, not unfinished requirements of the current release.
Numerical/protocol changes require new experiments; portability/container packaging
alone needs focused integration checks rather than another 750-candidate suite.

## Presentation and guardrails

Polish the root README and prepare resume bullets around the problem, engineering
decisions, bounded algorithm comparison, and verified GPU workflow. Link to the
runnable example and detailed docs. State method means as validation results and
the single selected model's score as test accuracy. Avoid universal superiority
claims. Preserve the [research direction](research-direction.md) for later.

Implement the smallest useful stage, verify it, update documentation, and commit
before expanding scope. No broad rewrite, new dashboard/report system, automatic
paid-resource purchase, or silent historical-result replacement is part of this plan.
