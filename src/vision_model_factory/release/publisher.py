"""Atomic Model Package publisher with staging and validation."""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
    DecoderSpec,
    EvaluationReport,
    FileRef,
    InputSpec,
    ModelManifest,
    OutputSpec,
    PostprocessSpec,
    PreprocessSpec,
    TargetSpec,
)
from vision_model_factory.contracts.validators import (
    ValidationError,
    validate_model_package,
)


class ReleasePublicationError(Exception):
    """Raised when model package publication or release gate fails."""


def publish_model_package(
    package_id: str,
    model_onnx_path: Path,
    task_json_path: Path,
    evaluation_json_path: Path,
    dataset_ref: DatasetRef,
    run_id: str,
    class_map: List[ClassMapItem],
    releases_dir: Path,
    target_spec: Optional[TargetSpec] = None,
    postprocess_spec: Optional[PostprocessSpec] = None,
    require_passed_gate: bool = True,
) -> Tuple[Path, ModelManifest]:
    """
    Publish an immutable Model Package atomically.

    Steps:
    1. Stage files in temporary staging directory.
    2. Copy model, upstream task.json (unchanged), and evaluation.json.
    3. Compute SHA-256 for all artifacts.
    4. Verify evaluation gate status.
    5. Generate and write model.json.
    6. Validate staged package completely.
    7. Atomically promote staged package to releases directory.
    """
    releases_dir = releases_dir.resolve()
    releases_dir.mkdir(parents=True, exist_ok=True)

    dest_dir = releases_dir / package_id
    if dest_dir.exists():
        raise ReleasePublicationError(
            f"Package '{package_id}' already exists at {dest_dir}. Immutability forbids overwriting."
        )

    # 1. Stage in temp dir
    with tempfile.TemporaryDirectory(dir=str(releases_dir.parent)) as tmp_stage:
        staging_pkg = Path(tmp_stage) / package_id
        staging_pkg.mkdir(parents=True, exist_ok=True)

        # 2. Copy artifacts
        dest_model = staging_pkg / "model.onnx"
        dest_task = staging_pkg / "task.json"
        dest_eval = staging_pkg / "evaluation.json"

        shutil.copy2(model_onnx_path, dest_model)
        shutil.copy2(task_json_path, dest_task)
        shutil.copy2(evaluation_json_path, dest_eval)

        # 3. Compute digests
        model_sha = compute_sha256_file(dest_model)
        task_sha = compute_sha256_file(dest_task)
        eval_sha = compute_sha256_file(dest_eval)

        # 4. Verify evaluation report gate
        with dest_eval.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)
            eval_report = EvaluationReport.model_validate(eval_data)

        if require_passed_gate:
            gate_status = eval_report.gate.get("status")
            if gate_status != "passed":
                raise ReleasePublicationError(
                    f"Model evaluation gate did not pass (status='{gate_status}'). Cannot publish unverified model."
                )

        # 5. Build model.json
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        num_classes = len(class_map)

        manifest = ModelManifest(
            schema_version="1.0.0",
            package_id=package_id,
            created_at=now_utc,
            task_type="object_detection",
            dataset=dataset_ref,
            run_id=run_id,
            task=FileRef(path="task.json", sha256=task_sha),
            model=FileRef(path="model.onnx", sha256=model_sha),
            evaluation=FileRef(path="evaluation.json", sha256=eval_sha),
            target=target_spec or TargetSpec(backend="onnxruntime", provider="cpu", precision="fp32"),
            input=InputSpec(name="images", dtype="float32", shape=[1, 3, 640, 640]),
            preprocess=PreprocessSpec(id="letterbox_rgb_u8_v1", pad_value=114),
            output=OutputSpec(name="output0", dtype="float32", shape=[1, 4 + num_classes, "N"]),
            decoder=DecoderSpec(id="yolo_xywh_scores_v1"),
            class_map=class_map,
            postprocess=postprocess_spec
            or PostprocessSpec(score_threshold=0.25, nms_iou_threshold=0.45, max_detections=100),
        )

        model_json_path = staging_pkg / "model.json"
        with model_json_path.open("w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        # 6. Validate complete staged package
        try:
            validate_model_package(staging_pkg)
        except ValidationError as e:
            raise ReleasePublicationError(f"Staged model package failed validation: {e}")

        # 7. Atomic promotion
        shutil.move(str(staging_pkg), str(dest_dir))

    return dest_dir, manifest
