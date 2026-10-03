# Vision Model Factory

Intent: consumer — Transform immutable dataset packages into verified, traceable model packages through constrained training experiments, locked evaluation, and export parity verification.

Done when:
1. Validates immutable Dataset Packages (Draft 2020-12 and semantic checks), maps stable `class_id`s to consecutive decoder indices, and excludes `partial` samples with explicit audit logging. — met; `validate-dataset` is the same code path publication uses.
2. Executes bounded training experiments governed by `ExperimentPolicy` (budget, timeout, parameter whitelist), producing cryptographically verified `RunResult` records without leaking locked test splits to the agent. — met; all three budget flags are required and a timeout kills the child process.
3. Exports trained models into self-contained `Model Package` bundles (ONNX CPU **FP32 only**) containing `model.json`, model artifact, upstream `task.json`, and `evaluation.json`, verifying export parity against reference letterbox preprocessor and YOLO score decoder. — met for FP32. **INT8 candidate production is not implemented**; `model.json` records the precision read off the graph, so an int8 release cannot be claimed by paperwork (pinned by test).
4. Atomically stages and publishes verified Model Packages into release registries while preventing unauthorized mutation of published artifacts. — met; publication also re-validates the cited Dataset Package, because mutual digest agreement between artifacts says nothing about the soundness of the data itself.

Delivery: Deterministic CLI (`python -m vision_model_factory.cli ...`) and importable Python library executed in the project virtual environment.

Quality:
- Immutability and cryptographic traceability first: all packages, specifications, and runs are keyed by immutable identifiers and SHA-256 digests; model evaluation is strictly segregated from agent optimization loops.
- Explicit contracts over heuristic inference: input/output shapes, decoder semantics, class mappings, and preprocessing steps are declared explicitly in manifests rather than inferred from filenames or runtime guesses.

Avoid:
- Letting LLM agents modify training code or increase compute budgets dynamically -> Agent proposes declarative `ExperimentSpec` validated against strict whitelist and parameter bounds.
- Leaking locked test data into iterative training or hyperparameter search -> Agent reads only train/val metrics and diagnostic summaries; test metrics are calculated independently during acceptance.
- Guessing class index order from class names -> Indices are deterministically mapped and validated against upstream `TaskSpec`.
- Spending the locked test split on anything other than the locked evaluation -> export parity must name a `train`, `val`, or `audit` corpus via `--parity-split`, and `export_parity.parity_split` is read off the records actually consumed. `test` is absent from the contract's type, so the claim cannot be written.
- Recording a gate verdict as stored truth -> the verdict is re-derived from the gate policy digest at publication; `gate.checks[].actual` must equal the measured metrics.
- Shipping a passing parity comparison that compared nothing -> `status: "passed"` structurally requires non-empty detections on both sides and a self-test that detected an injected coordinate perturbation.
- In-process arbitrary post-processing scripts inside model packages -> Declarative preprocess and decoder specifications strictly verified against reference implementations.

Decisions:
- Implement bounded contexts: `contracts`, `trainers`, `evaluation`, `export`, `experiments`, `release`, and `cli`.
- Decouple training framework execution via `BaseTrainerAdapter` protocol, providing a standard YOLO adapter supporting both production Ultralytics training and deterministic mock execution for test suites.
- Implement strict architectural boundary enforcement via `check-boundaries` AST inspection, ensuring zero imports from runtime or data-factory source repositories.
- Treat `contracts/schemas/data_v1/` as the single Model-side copy of Data-owned wire schemas, hash-verified against `contract-manifest.json`. Duplicates of `task_spec`/`dataset_manifest`/`sample_record`/`annotation_record` were removed from `schemas/` root: two copies of a wire contract means one of them drifts silently.
- Keep the reference preprocessor/decoder in `export` and let `evaluation` import it, so the metric path and the parity path cannot drift into two implementations of "how a box is decoded".

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
- **INT8 quantized candidate production.** The calibration-source guard and latency/memory measurement exist, and graph precision is detected for reporting, but nothing here produces a quantized graph. Publishing an int8 target for a float32 graph fails the gate.
- **Dataset repair.** `datasets/ds-african-wildlife-v1` has 20 image byte-hashes crossing splits (test/train 12, train/val 7, test/val 1). Re-splitting is a Data Factory decision under N4; see `ESCALATIONS.md`.
- **Machine-readable fixture for the letterbox pixel convention.** The spec states 较小半边放左/上 in prose; there is no upstream test vector, so a 1px padding disagreement was arguable until pinned by pixel assertions here.
- **Registry as a release authority.** `registry` tracks candidate/approved/active and requires approval evidence, but approval is operator input; it does not verify that the evidence is true.
- **Gate-policy schema artifact.** `GatePolicy` is validated by pydantic only and has no `contracts/schemas/*.json` mirror, because it is an input document rather than a published artifact.
