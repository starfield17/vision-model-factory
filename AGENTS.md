# Vision Model Factory

Vision Model Factory executes bounded training experiments on immutable dataset packages, performs independent locked evaluation, verifies export parity, and publishes verified model packages.

## Commands

```bash
# Execute using the project's Python environment (e.g. Lab conda env)
PYTHONPATH=src python -m pytest tests
python -m ruff check src tests
PYTHONPATH=src python -m vision_model_factory.check_boundaries
PYTHONPATH=src python -m vision_model_factory.cli --help
```

## Hard rules

Everything here is enforced by the boundary check (`python -m vision_model_factory.check_boundaries`).
If you think one is wrong, say so — do not work around it.

- `contracts` must never import `trainers`, `evaluation`, `export`, `experiments`, `release`, or `cli`.
- `trainers` and `evaluation` may import `contracts`, but must never import `experiments`, `release`, or `cli`.
- `export` may import `contracts`, but must never import `experiments`, `trainers`, `release`, or `cli`.
- No module within `vision_model_factory` may import `vision_runtime`, `vision_data_factory`, or `backend.app` code. All cross-repository communication is strictly through serialized package artifacts.
- Training datasets exclude samples marked `annotation_status: "partial"`, recording their count in audit logs.
- Test set samples and ground-truth annotations are strictly locked from agent training and hyperparameter search iterations.
- Model package publication is atomic: staging validation must pass completely before release promotion.

## Map

| Directory | Owns | Depends on |
|---|---|---|
| `src/vision_model_factory/contracts` | Schemas, Pydantic models, SHA-256 validation, semantic validators | External libraries only |
| `src/vision_model_factory/trainers` | Trainer adapter protocol, YOLO adapter, dataset conversion, partial filtering | `contracts` |
| `src/vision_model_factory/evaluation` | Detection metrics (mAP, PR, confusion), locked test evaluation | `contracts` |
| `src/vision_model_factory/export` | Letterbox preprocessor, YOLO decoder, ONNX exporter, parity verifier | `contracts` |
| `src/vision_model_factory/experiments` | ExperimentSpec validation, budget policy runner, constrained agent executor | `contracts`, `trainers`, `evaluation` |
| `src/vision_model_factory/release` | Atomic package staging and publishing, registry candidate/approval indexing | `contracts`, `evaluation`, `export` |
| `src/vision_model_factory/cli.py` | Deterministic CLI composition root | All factory modules |

## Where things are that are not obvious from the map

- `src/vision_model_factory/export/preprocessor.py`: Implements reference `letterbox_rgb_u8_v1` specification.
- `src/vision_model_factory/export/decoder.py`: Implements reference `yolo_xywh_scores_v1` specification.
- `src/vision_model_factory/trainers/yolo.py`: Adapts immutable dataset packages into YOLO format while enforcing group-level train/val splits and partial exclusions.
