"""Semantic validators for dataset packages, model packages, and metadata."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import (
    AnnotationRecord,
    DatasetManifest,
    EvaluationReport,
    ModelManifest,
    SampleRecord,
    TaskSpec,
)


class ValidationError(Exception):
    """Raised when semantic or contract validation fails."""


def validate_relative_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """Validate that a relative path does not escape its base directory."""
    if not path_str or path_str.startswith("/") or path_str.startswith("\\"):
        raise ValidationError(f"Path must be relative, got: {path_str}")

    parts = path_str.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValidationError(f"Path must not traverse parent directories (..): {path_str}")

    if base_dir is not None:
        target = (base_dir / path_str).resolve()
        resolved_base = base_dir.resolve()
        try:
            target.relative_to(resolved_base)
        except ValueError:
            raise ValidationError(f"Path escapes base directory: {path_str}")
        return target
    return Path(path_str)


def validate_bbox_coordinates(
    bbox: List[float], width: int, height: int, sample_id: str = "", annotation_id: str = ""
) -> None:
    """Validate 2D bounding box coordinate bounds [x1, y1, x2, y2]."""
    if len(bbox) != 4:
        raise ValidationError(f"BBox must contain exactly 4 floats, got: {bbox}")

    x1, y1, x2, y2 = bbox
    context = f"(annotation_id={annotation_id}, sample_id={sample_id})"

    if not (0 <= x1 < x2 <= width):
        raise ValidationError(
            f"Invalid x bounds: 0 <= {x1} < {x2} <= {width} violated {context}"
        )
    if not (0 <= y1 < y2 <= height):
        raise ValidationError(
            f"Invalid y bounds: 0 <= {y1} < {y2} <= {height} violated {context}"
        )


def validate_class_map_against_task(
    class_map: List[dict], task: TaskSpec, require_consecutive_zero_indexed: bool = True
) -> None:
    """Validate class_map indices and category bindings against TaskSpec."""
    known_category_ids = {cat.class_id for cat in task.categories}
    seen_indices: Set[int] = set()
    seen_classes: Set[str] = set()

    for item in class_map:
        idx = item.get("index") if isinstance(item, dict) else item.index
        cid = item.get("class_id") if isinstance(item, dict) else item.class_id

        if idx in seen_indices:
            raise ValidationError(f"Duplicate class_map index: {idx}")
        if cid in seen_classes:
            raise ValidationError(f"Duplicate class_map class_id: {cid}")
        if cid not in known_category_ids:
            raise ValidationError(
                f"Class ID '{cid}' in class_map does not exist in TaskSpec categories: {known_category_ids}"
            )

        seen_indices.add(idx)
        seen_classes.add(cid)

    if require_consecutive_zero_indexed:
        expected = set(range(len(class_map)))
        if seen_indices != expected:
            raise ValidationError(
                f"Class map indices must be consecutive integers starting from 0 to {len(class_map)-1}, "
                f"got: {sorted(seen_indices)}"
            )


def validate_dataset_package(
    pkg_dir: Path,
) -> Tuple[DatasetManifest, TaskSpec, List[SampleRecord], List[AnnotationRecord]]:
    """Completely validate an immutable Dataset Package on disk."""
    pkg_dir = pkg_dir.resolve()
    manifest_path = pkg_dir / "dataset.json"
    if not manifest_path.is_file():
        raise ValidationError(f"Missing dataset.json in {pkg_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        try:
            raw_manifest = json.load(f)
        except Exception as e:
            raise ValidationError(f"Failed to parse dataset.json: {e}")

    manifest = DatasetManifest.model_validate(raw_manifest)

    # Verify task file
    task_file = validate_relative_path(manifest.task.path, pkg_dir)
    if not task_file.is_file():
        raise ValidationError(f"Referenced task file does not exist: {manifest.task.path}")
    actual_task_sha = compute_sha256_file(task_file)
    if actual_task_sha != manifest.task.sha256:
        raise ValidationError(
            f"Task file SHA-256 mismatch: expected {manifest.task.sha256}, got {actual_task_sha}"
        )
    with task_file.open("r", encoding="utf-8") as f:
        task = TaskSpec.model_validate(json.load(f))

    # Verify quality file
    quality_file = validate_relative_path(manifest.quality.path, pkg_dir)
    if not quality_file.is_file():
        raise ValidationError(f"Referenced quality file does not exist: {manifest.quality.path}")
    actual_quality_sha = compute_sha256_file(quality_file)
    if actual_quality_sha != manifest.quality.sha256:
        raise ValidationError(
            f"Quality file SHA-256 mismatch: expected {manifest.quality.sha256}, got {actual_quality_sha}"
        )

    # Verify samples file
    samples_file = validate_relative_path(manifest.samples.path, pkg_dir)
    if not samples_file.is_file():
        raise ValidationError(f"Referenced samples file does not exist: {manifest.samples.path}")
    actual_samples_sha = compute_sha256_file(samples_file)
    if actual_samples_sha != manifest.samples.sha256:
        raise ValidationError(
            f"Samples file SHA-256 mismatch: expected {manifest.samples.sha256}, got {actual_samples_sha}"
        )

    samples: List[SampleRecord] = []
    sample_ids: Set[str] = set()
    sample_by_id: Dict[str, SampleRecord] = {}
    group_to_splits: Dict[str, Set[str]] = {}

    with samples_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = SampleRecord.model_validate_json(line)
            except Exception as e:
                raise ValidationError(f"Invalid SampleRecord at line {line_num} in samples.jsonl: {e}")

            if rec.sample_id in sample_ids:
                raise ValidationError(f"Duplicate sample_id '{rec.sample_id}' at line {line_num}")
            sample_ids.add(rec.sample_id)
            sample_by_id[rec.sample_id] = rec
            samples.append(rec)

            # Check image file existence if it exists in bundle
            image_path = validate_relative_path(rec.file.path, pkg_dir)
            if image_path.is_file():
                actual_img_sha = compute_sha256_file(image_path)
                if actual_img_sha != rec.file.sha256:
                    raise ValidationError(
                        f"Image file SHA-256 mismatch for sample {rec.sample_id}: "
                        f"expected {rec.file.sha256}, got {actual_img_sha}"
                    )

            # Track group leakage
            if rec.group_id not in group_to_splits:
                group_to_splits[rec.group_id] = set()
            group_to_splits[rec.group_id].add(rec.split)

    # Enforce split group isolation: no group may span multiple splits
    for gid, splits in group_to_splits.items():
        if len(splits) > 1:
            raise ValidationError(
                f"Data leakage detected: group_id '{gid}' spans multiple splits: {splits}"
            )

    # Verify annotations file
    ann_file = validate_relative_path(manifest.annotations.path, pkg_dir)
    if not ann_file.is_file():
        raise ValidationError(f"Referenced annotations file does not exist: {manifest.annotations.path}")
    actual_ann_sha = compute_sha256_file(ann_file)
    if actual_ann_sha != manifest.annotations.sha256:
        raise ValidationError(
            f"Annotations file SHA-256 mismatch: expected {manifest.annotations.sha256}, got {actual_ann_sha}"
        )

    annotations: List[AnnotationRecord] = []
    annotation_ids: Set[str] = set()
    category_ids = {cat.class_id for cat in task.categories}

    with ann_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ann = AnnotationRecord.model_validate_json(line)
            except Exception as e:
                raise ValidationError(f"Invalid AnnotationRecord at line {line_num}: {e}")

            if ann.annotation_id in annotation_ids:
                raise ValidationError(f"Duplicate annotation_id '{ann.annotation_id}' at line {line_num}")
            annotation_ids.add(ann.annotation_id)

            if ann.sample_id not in sample_by_id:
                raise ValidationError(
                    f"Annotation '{ann.annotation_id}' references non-existent sample_id: '{ann.sample_id}'"
                )

            if ann.class_id not in category_ids:
                raise ValidationError(
                    f"Annotation '{ann.annotation_id}' references unknown class_id '{ann.class_id}' "
                    f"not in TaskSpec: {category_ids}"
                )

            parent_sample = sample_by_id[ann.sample_id]
            validate_bbox_coordinates(
                ann.bbox_xyxy,
                parent_sample.width,
                parent_sample.height,
                sample_id=ann.sample_id,
                annotation_id=ann.annotation_id,
            )
            annotations.append(ann)

    return manifest, task, samples, annotations


def validate_model_package(pkg_dir: Path) -> Tuple[ModelManifest, TaskSpec, EvaluationReport]:
    """Completely validate an immutable Model Package on disk."""
    pkg_dir = pkg_dir.resolve()
    manifest_path = pkg_dir / "model.json"
    if not manifest_path.is_file():
        raise ValidationError(f"Missing model.json in {pkg_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        try:
            raw_manifest = json.load(f)
        except Exception as e:
            raise ValidationError(f"Failed to parse model.json: {e}")

    manifest = ModelManifest.model_validate(raw_manifest)

    # Verify task file
    task_file = validate_relative_path(manifest.task.path, pkg_dir)
    if not task_file.is_file():
        raise ValidationError(f"Referenced task file does not exist: {manifest.task.path}")
    actual_task_sha = compute_sha256_file(task_file)
    if actual_task_sha != manifest.task.sha256:
        raise ValidationError(
            f"Task file SHA-256 mismatch in model package: expected {manifest.task.sha256}, got {actual_task_sha}"
        )
    with task_file.open("r", encoding="utf-8") as f:
        task = TaskSpec.model_validate(json.load(f))

    # Verify model file
    model_file = validate_relative_path(manifest.model.path, pkg_dir)
    if not model_file.is_file():
        raise ValidationError(f"Referenced model file does not exist: {manifest.model.path}")
    actual_model_sha = compute_sha256_file(model_file)
    if actual_model_sha != manifest.model.sha256:
        raise ValidationError(
            f"Model file SHA-256 mismatch: expected {manifest.model.sha256}, got {actual_model_sha}"
        )

    # Verify evaluation file
    eval_file = validate_relative_path(manifest.evaluation.path, pkg_dir)
    if not eval_file.is_file():
        raise ValidationError(f"Referenced evaluation file does not exist: {manifest.evaluation.path}")
    actual_eval_sha = compute_sha256_file(eval_file)
    if actual_eval_sha != manifest.evaluation.sha256:
        raise ValidationError(
            f"Evaluation file SHA-256 mismatch: expected {manifest.evaluation.sha256}, got {actual_eval_sha}"
        )
    with eval_file.open("r", encoding="utf-8") as f:
        evaluation = EvaluationReport.model_validate(json.load(f))

    # Verify class map semantics
    validate_class_map_against_task(manifest.class_map, task, require_consecutive_zero_indexed=True)

    # Verify output shape channel count matches decoder requirement
    if manifest.decoder.id == "yolo_xywh_scores_v1":
        num_classes = len(manifest.class_map)
        expected_channels = 4 + num_classes
        # Output shape is [batch, channels, num_boxes]
        if len(manifest.output.shape) >= 2:
            out_channels = manifest.output.shape[1]
            if isinstance(out_channels, int) and out_channels != expected_channels:
                raise ValidationError(
                    f"Output channels mismatch for decoder 'yolo_xywh_scores_v1': "
                    f"expected 4 + {num_classes} = {expected_channels}, got {out_channels}"
                )

    return manifest, task, evaluation
