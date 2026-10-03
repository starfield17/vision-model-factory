# Vision Model Factory

Vision Model Factory provides deterministic tools for training, evaluating, exporting, and publishing object detection vision models. It consumes immutable Dataset Packages published by Vision Data Factory and outputs validated Model Packages ready for consumption by Vision Runtime.

## Key Features

- **Contract-Driven**: Strict validation of schemas and semantic invariants (`TaskSpec`, `DatasetManifest`, `ExperimentSpec`, `RunResult`, `ModelManifest`, `EvaluationReport`).
- **Bounded Experimentation**: Strict budget execution (time, max runs, parameter bounds) protecting training integrity against unconstrained optimization loops.
- **Unified Evaluation**: Independent evaluation of locked test splits computing mAP50, mAP50:95, and per-class precision/recall curves.
- **Export & Parity**: Verified ONNX export against a shared reference preprocessor (`letterbox_rgb_u8_v1`) and decoder (`yolo_xywh_scores_v1`), with a self-test that proves the comparison can see a coordinate error.
- **Atomic Publishing**: Validated staging promotion to immutable release registries, with the release verdict re-derived from an operator-supplied gate policy.

## Getting Started

```bash
python -m pytest tests                                  # run the suite
python -m ruff check src tests scripts                  # lint
PYTHONPATH=src python -m vision_model_factory.check_boundaries
PYTHONPATH=src python -m vision_model_factory.cli --help
```

## The workflow

Every step below takes its thresholds and its corpus as explicit arguments. There are no
built-in defaults for anything that decides a verdict.

```bash
CLI="python -m vision_model_factory.cli"   # with PYTHONPATH=src exported

# 1. Inspect the data you are about to build on. Refuses a package whose image bytes
#    cross splits, whose digests do not match, or whose paths escape the package.
$CLI validate-dataset datasets/ds-african-wildlife-v1

# 2. Run one bounded experiment. All three budget flags are required; a run without a
#    budget is not a bounded run.
$CLI run-experiment spec.json datasets/ds-african-wildlife-v1 \
  --work-dir workspaces/exp --run-id run-001 \
  --max-runs 1 --max-total-seconds 3600 --per-run-timeout-seconds 1800 [--mock]

# 3. Export the trained checkpoint to ONNX. The graph's tensor interface is read back
#    out of the file afterwards; nothing about shapes is taken from the caller.
$CLI export workspaces/exp/run-001/attempt/train/best.pt model.onnx --imgsz 640

# 4. Score the locked test split, verify parity, benchmark the target.
#    `--gate-policy` and `--parity-split` are required. Parity is never measured on the
#    locked test corpus: name `audit` (preferred), `val`, or `train`. Without
#    `--reference-weights` there is nothing to compare against, so parity is recorded as
#    `failed` with `method: "not_run"` - never as skipped, and never as passed.
$CLI evaluate datasets/ds-african-wildlife-v1 model.onnx configs/gate_policies/mine.json \
  --task-json datasets/ds-african-wildlife-v1/task.json --run-id run-001 \
  --class-map-json '[{"index":0,"class_id":"buffalo"}]' \
  --reference-weights best.pt --parity-split audit --output evaluation.json

# 5. Publish. The dataset package is re-validated here and its id/digest are read from it
#    rather than typed, and the gate verdict is re-derived from the same policy document.
$CLI publish --package-id model-wildlife-001 --model-path model.onnx \
  --task-json datasets/ds-african-wildlife-v1/task.json --eval-path evaluation.json \
  --dataset-dir datasets/ds-african-wildlife-v1 --run-id run-001 \
  --class-map-json '[{"index":0,"class_id":"buffalo"}]' \
  --gate-policy configs/gate_policies/mine.json --releases-dir releases

# 6. Verify the published artifact from its bytes, not from the run that made it.
$CLI validate-model releases/model-wildlife-001

# 7. Track the lifecycle: candidate -> approved (evidence required) -> active.
$CLI registry --registry-file releases/registry.json --register model-wildlife-001 \
  --package-dir releases/model-wildlife-001
$CLI registry --registry-file releases/registry.json --approve model-wildlife-001 \
  --evidence '{"mAP50": 0.91}'
$CLI registry --registry-file releases/registry.json --activate model-wildlife-001
```

`evaluate` writes the complete report and *then* exits non-zero if the gate failed: the report
is the measurement, the exit code is the verdict, and discarding a measurement because it
came out bad would destroy the evidence. `publish` works the other way round - it stages in a
sibling temp directory and promotes with a single rename only after full validation, so a
refused publication leaves no partial release on disk, and it rejects a report whose recorded
policy digest does not match the policy you hand it.

## What this repository does not do

- It does not build or repair datasets. `scripts/build_dataset_package.py` is an *importer*
  for a public archive and records that faithfully (`origin: "imported"`,
  `review_state: "unreviewed"`, `audit: null`, `gate: null`); it validates what it produced
  and stops with exit 3 if the data is unsound. Split decisions belong to Data Factory.
- It does not produce INT8 models. `export/quantization.py` guards calibration-sample
  legitimacy; the quantized candidate itself is not part of the supported path (see
  `SPEC.md`).
- It does not ship a gate policy. `configs/gate_policies/` contains a template to edit, and
  no code reads it.
- `datasets/`, `releases/`, `workspaces/`, and `runs/` are gitignored. They are local
  evidence, not delivered artifacts, and a clean clone contains no published model package.
