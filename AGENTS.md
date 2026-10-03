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

The import-direction and cross-repository rules below are enforced by
`python -m vision_model_factory.check_boundaries`, which also catches relative and
string-literal dynamic imports. The rest are enforced elsewhere - by `contracts.validators`
at the package boundary, or by a test - and the checker will **not** stop you breaking
them. If you think one is wrong, say so - do not work around it.

- `contracts` must never import `trainers`, `evaluation`, `export`, `experiments`, `release`, or `cli`.
- `trainers` and `evaluation` may import `contracts`, but must never import `experiments`, `release`, or `cli`. `evaluation` also imports `export`, for the one reference preprocessor/decoder shared with the parity check.
- `export` may import `contracts`, but must never import `experiments`, `trainers`, `release`, or `cli`.
- No module within `vision_model_factory` may import `vision_runtime`, `vision_data_factory`, or `backend.app` code. All cross-repository communication is strictly through serialized package artifacts.
- Training datasets exclude samples marked `annotation_status: "partial"`, recording their count in audit logs.
- Test set samples and ground-truth annotations are strictly locked from agent training and hyperparameter search iterations.
- Export parity is never measured on the `test` split. Parity consumes no annotations, but spending the scored corpus on an artifact check makes "the test set was touched once" unverifiable, so `export_parity.parity_split` cannot record `test` at all.
- A `test`-split class the dataset cannot supply ground truth for is a refusal, not a zero. Macro-mAP over partly unmeasurable classes is not a number.
- Model package publication is atomic: staging validation must pass completely before release promotion. Publication also **re-validates the dataset package it cites**; digest agreement between artifacts proves only that they agree with each other.
- No default thresholds. `evaluate` and `publish` require a gate policy document, and `evaluate` requires `--parity-split`. `configs/gate_policies/` holds examples to edit; nothing in `src/` reads that directory.
- A gate verdict is never stored as a claim. It is re-derived from the policy digest at publication, so editing a `status` string cannot turn a failing model into a releasable one.

## Map

| Directory | Owns | Depends on |
|---|---|---|
| `src/vision_model_factory/contracts` | Schemas, Pydantic models, SHA-256 validation, semantic validators | External libraries only |
| `src/vision_model_factory/trainers` | Trainer adapter protocol, YOLO adapter, dataset conversion, partial filtering | `contracts` |
| `src/vision_model_factory/evaluation` | Detection metrics (mAP, PR, confusion), locked test evaluation | `contracts`, `export` (shared reference preprocess/decode) |
| `src/vision_model_factory/export` | Letterbox preprocessor, YOLO decoder, ONNX exporter, parity verifier | `contracts` |
| `src/vision_model_factory/experiments` | ExperimentSpec validation, budget policy runner, constrained agent executor | `contracts`, `trainers` |
| `src/vision_model_factory/release` | Atomic package staging and publishing, registry candidate/approval indexing | `contracts`, `export` |
| `src/vision_model_factory/cli.py` | Deterministic CLI composition root | All factory modules |

## Where things are that are not obvious from the map

- `src/vision_model_factory/export/preprocessor.py`: Implements reference `letterbox_rgb_u8_v1` specification. Padding puts the **smaller** half on the left/top (`02-model-factory.md` §3: 较小半边放左/上); odd-size cases are pinned pixel-wise in `tests/test_preprocessor_and_decoder.py`.
- `src/vision_model_factory/export/decoder.py`: Implements reference `yolo_xywh_scores_v1` specification.
- `src/vision_model_factory/trainers/yolo.py`: Adapts immutable dataset packages into YOLO format while enforcing group-level train/val splits and partial exclusions. The Data `val` split feeds ultralytics' own validation and checkpoint selection, so `val` is training-visible.
- `src/vision_model_factory/contracts/schemas/data_v1/`: Data-owned wire schemas, mirrored read-only and hash-checked against `contract-manifest.json`. Do not edit them here and do not copy them to `schemas/` root - the mirror is the only Model-side copy.
- `src/vision_model_factory/contracts/models.py::model_json`: The canonical writer for contract artifacts. It omits unset optional fields, because `model_dump_json()` emits `null` and the schemas declare those fields as objects/strings. Absent and explicit-null are different states in the contract; the third state the serializer would have produced was undefined.
- `src/vision_model_factory/release/publisher.py`: Copies `evaluation.json` byte-for-byte and re-derives the verdict from the supplied policy, so an evaluation file's own `gate` cannot disagree with the policy the release was made under. It takes a dataset **directory**, not a typed dataset id.
- `datasets/`, `releases/`, `workspaces/`, `runs/` are gitignored: local evidence, not delivered artifacts.
