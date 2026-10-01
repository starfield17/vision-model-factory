# Vision Model Factory

Vision Model Factory provides deterministic tools for training, evaluating, exporting, and publishing object detection vision models. It consumes immutable Dataset Packages published by Vision Data Factory and outputs validated Model Packages ready for consumption by Vision Runtime.

## Key Features

- **Contract-Driven**: Strict validation of schemas and semantic invariants (`TaskSpec`, `DatasetManifest`, `ExperimentSpec`, `RunResult`, `ModelManifest`, `EvaluationReport`).
- **Bounded Experimentation**: Strict budget execution (time, max runs, parameter bounds) protecting training integrity against unconstrained optimization loops.
- **Unified Evaluation**: Independent evaluation of locked test splits computing mAP50, mAP50:95, and per-class precision/recall curves.
- **Export & Parity**: Verified ONNX export supporting declarative preprocessors (`letterbox_rgb_u8_v1`) and decoders (`yolo_xywh_scores_v1`) with zero in-package executable code.
- **Atomic Publishing**: Validated staging promotion to immutable release registries.

## Getting Started

```bash
# Run test suite
python -m pytest tests

# Run boundary checks
python -m vision_model_factory.check_boundaries

# Inspect CLI
python -m vision_model_factory.cli --help
```
