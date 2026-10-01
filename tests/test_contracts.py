"""Tests for contracts, schemas, and semantic validators."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from vision_model_factory.contracts.hashing import is_valid_sha256
from vision_model_factory.contracts.models import (
    ClassMapItem,
    FileRef,
    TaskSpec,
)
from vision_model_factory.contracts.validators import (
    ValidationError,
    validate_bbox_coordinates,
    validate_class_map_against_task,
    validate_dataset_package,
)


def test_hashing_and_sha_validation():
    valid_sha = "a" * 64
    assert is_valid_sha256(valid_sha) is True
    assert is_valid_sha256("A" * 64) is False  # Must be lowercase
    assert is_valid_sha256("a" * 63) is False  # Must be 64 chars
    assert is_valid_sha256("g" * 64) is False  # Must be valid hex


def test_file_ref_relative_path_rules():
    valid_sha = "0" * 64
    # Valid relative path
    ref = FileRef(path="images/sample1.jpg", sha256=valid_sha)
    assert ref.path == "images/sample1.jpg"

    # Reject absolute path
    with pytest.raises(PydanticValidationError):
        FileRef(path="/root/images/sample1.jpg", sha256=valid_sha)

    # Reject parent traversal (..)
    with pytest.raises(PydanticValidationError):
        FileRef(path="../images/sample1.jpg", sha256=valid_sha)

    with pytest.raises(PydanticValidationError):
        FileRef(path="images/../../sample1.jpg", sha256=valid_sha)


def test_bbox_coordinate_validation():
    # Valid box
    validate_bbox_coordinates([10.0, 20.0, 100.0, 200.0], width=640, height=480)

    # Inverted x: x2 <= x1
    with pytest.raises(ValidationError):
        validate_bbox_coordinates([100.0, 20.0, 10.0, 200.0], width=640, height=480)

    # Out of bounds x2 > width
    with pytest.raises(ValidationError):
        validate_bbox_coordinates([10.0, 20.0, 700.0, 200.0], width=640, height=480)

    # Negative coordinates
    with pytest.raises(ValidationError):
        validate_bbox_coordinates([-5.0, 20.0, 100.0, 200.0], width=640, height=480)


def test_class_map_validation(sample_task_spec_dict):
    task = TaskSpec.model_validate(sample_task_spec_dict)

    # Valid consecutive 0-indexed class map
    valid_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]
    validate_class_map_against_task(valid_map, task)

    # Non-consecutive indices (e.g. 0, 2)
    invalid_indices_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=2, class_id="can"),
    ]
    with pytest.raises(ValidationError):
        validate_class_map_against_task(invalid_indices_map, task)

    # Unknown category ID
    unknown_cat_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="unknown_item"),
    ]
    with pytest.raises(ValidationError):
        validate_class_map_against_task(unknown_cat_map, task)


def test_synthetic_dataset_package_validation(synthetic_dataset_dir: Path):
    manifest, task, samples, annotations = validate_dataset_package(synthetic_dataset_dir)
    assert manifest.dataset_id == "ds-test-001"
    assert len(samples) == 3
    assert len(annotations) == 3


def test_dataset_group_leakage_rejection(synthetic_dataset_dir: Path):
    """Verify that a group_id spanning multiple splits is strictly rejected."""
    samples_file = synthetic_dataset_dir / "samples.jsonl"
    lines = samples_file.read_text(encoding="utf-8").strip().splitlines()

    # Modify sample 2 to have group-1 (same group as sample 1 in train, but sample 2 is in val)
    s2 = json.loads(lines[1])
    s2["group_id"] = "grp-1"
    lines[1] = json.dumps(s2)

    samples_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update samples sha in dataset.json
    from vision_model_factory.contracts.hashing import compute_sha256_file
    new_sha = compute_sha256_file(samples_file)
    ds_json_path = synthetic_dataset_dir / "dataset.json"
    with ds_json_path.open("r", encoding="utf-8") as f:
        ds_data = json.load(f)
    ds_data["samples"]["sha256"] = new_sha
    with ds_json_path.open("w", encoding="utf-8") as f:
        json.dump(ds_data, f, indent=2)

    # Should raise ValidationError due to group leakage across train and val
    with pytest.raises(ValidationError, match="Data leakage detected: group_id 'grp-1'"):
        validate_dataset_package(synthetic_dataset_dir)
