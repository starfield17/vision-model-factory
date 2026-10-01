# Vision Model Factory

Intent: consumer — Transform immutable dataset packages into verified, traceable model packages through constrained training experiments, locked evaluation, and export parity verification.

Done when:
1. Validates immutable Dataset Packages (Draft 2020-12 and semantic checks), maps stable `class_id`s to consecutive decoder indices, and excludes `partial` samples with explicit audit logging.
2. Executes bounded training experiments governed by `ExperimentPolicy` (budget, timeout, parameter whitelist), producing cryptographically verified `RunResult` records without leaking locked test splits to the agent.
3. Exports trained models into self-contained `Model Package` bundles (ONNX CPU FP32/INT8) containing `model.json`, model artifact, upstream `task.json`, and `evaluation.json`, verifying export parity against reference letterbox preprocessor and YOLO score decoder.
4. Atomically stages and publishes verified Model Packages into release registries while preventing unauthorized mutation of published artifacts.

Delivery: Deterministic CLI (`python -m vision_model_factory.cli ...`) and importable Python library executed in the project virtual environment.

Quality:
- Immutability and cryptographic traceability first: all packages, specifications, and runs are keyed by immutable identifiers and SHA-256 digests; model evaluation is strictly segregated from agent optimization loops.
- Explicit contracts over heuristic inference: input/output shapes, decoder semantics, class mappings, and preprocessing steps are declared explicitly in manifests rather than inferred from filenames or runtime guesses.

Avoid:
- Letting LLM agents modify training code or increase compute budgets dynamically -> Agent proposes declarative `ExperimentSpec` validated against strict whitelist and parameter bounds.
- Leaking locked test data into iterative training or hyperparameter search -> Agent reads only train/val metrics and diagnostic summaries; test metrics are calculated independently during acceptance.
- Guessing class index order from class names -> Indices are deterministically mapped and validated against upstream `TaskSpec`.
- In-process arbitrary post-processing scripts inside model packages -> Declarative preprocess and decoder specifications strictly verified against reference implementations.

Decisions:
- Implement bounded contexts: `contracts`, `trainers`, `evaluation`, `export`, `experiments`, `release`, and `cli`.
- Decouple training framework execution via `BaseTrainerAdapter` protocol, providing a standard YOLO adapter supporting both production Ultralytics training and deterministic mock execution for test suites.
- Implement strict architectural boundary enforcement via `check-boundaries` AST inspection, ensuring zero imports from runtime or data-factory source repositories.

## Do not build
- N1: First version task scope is strictly limited to 2D object detection; do not expand into multi-task, pose, segmentation, or tracking architectures.
- N2: Do not implement distributed cluster schedulers, Kubernetes operators, dynamic web dashboards, or remote agent microservices; execution is local deterministic CLI and library calls.
- N3: Do not permit dynamic code execution or custom script injection within model package bundles; decoder and preprocessor routines must match registered declarative IDs.
- N4: Do not allow models or agents to alter upstream dataset packages; data corrections must be requested from Vision Data Factory as a new dataset release.
- N5: Do not report synthetic or fabricated quality scores; failed runs or invalid metrics must remain explicit failures rather than placeholder successes.
- N6: When something missing or broken turns up outside this spec, append it to "Found · Not doing" and keep going. Do not implement it.

## Prior art
- Reusing: Ultralytics YOLOv8 for object detection training baseline.
- Reusing: ONNX and ONNX Runtime CPU execution provider for deployment export and parity validation.

## Found · Not doing
- RKNN / TensorRT hardware-specific acceleration backend exporters in baseline (deferred to post-v1 hardware milestones).
