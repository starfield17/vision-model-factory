"""Tests for atomic model package publication and registry management."""

import json
from pathlib import Path

import pytest

from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
)
from vision_model_factory.contracts.validators import validate_model_package
from vision_model_factory.export.exporter import export_torch_model_to_onnx
from vision_model_factory.release.publisher import ReleasePublicationError, publish_model_package
from vision_model_factory.release.registry import ModelRegistry
from vision_model_factory.trainers.yolo import TinyYoloMockNet


def test_atomic_model_package_publication(tmp_path: Path):
    # 1. Prepare dummy model ONNX file
    model = TinyYoloMockNet(num_classes=2, num_anchors=10)
    onnx_path, _ = export_torch_model_to_onnx(model, tmp_path / "model.onnx")

    # 2. Prepare dummy task.json
    task_json_path = tmp_path / "task.json"
    task_data = {
        "schema_version": "1.0.0",
        "task_id": "test-task",
        "task_type": "object_detection",
        "categories": [
            {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle"},
            {"class_id": "can", "display_name": "Can", "prompt": "can"},
        ],
    }
    task_json_path.write_text(json.dumps(task_data, indent=2), encoding="utf-8")

    # 3. Prepare evaluation.json (gate status: passed)
    eval_json_path = tmp_path / "evaluation.json"
    eval_data = {
        "schema_version": "1.0.0",
        "run_id": "run-001",
        "dataset": {"dataset_id": "ds-001", "manifest_sha256": "0" * 64},
        "test": {"mAP50": 0.90},
        "export_parity": {"status": "passed"},
        "target_benchmarks": [],
        "gate": {"status": "passed"},
    }
    eval_json_path.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")

    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]
    dataset_ref = DatasetRef(dataset_id="ds-001", manifest_sha256="0" * 64)
    releases_dir = tmp_path / "releases"

    # 4. Publish package
    pub_dir, manifest = publish_model_package(
        package_id="model-demo-001",
        model_onnx_path=onnx_path,
        task_json_path=task_json_path,
        evaluation_json_path=eval_json_path,
        dataset_ref=dataset_ref,
        run_id="run-001",
        class_map=class_map,
        releases_dir=releases_dir,
    )

    assert pub_dir.is_dir()
    assert (pub_dir / "model.json").is_file()
    assert (pub_dir / "model.onnx").is_file()
    assert (pub_dir / "task.json").is_file()
    assert (pub_dir / "evaluation.json").is_file()

    # Validate package
    validated_manifest, task, evaluation = validate_model_package(pub_dir)
    assert validated_manifest.package_id == "model-demo-001"
    assert validated_manifest.target.backend == "onnxruntime"

    # Rejection of duplicate package_id (immutability)
    with pytest.raises(ReleasePublicationError, match="already exists"):
        publish_model_package(
            package_id="model-demo-001",
            model_onnx_path=onnx_path,
            task_json_path=task_json_path,
            evaluation_json_path=eval_json_path,
            dataset_ref=dataset_ref,
            run_id="run-001",
            class_map=class_map,
            releases_dir=releases_dir,
        )


def test_rejection_of_failed_gate_release(tmp_path: Path):
    model = TinyYoloMockNet(num_classes=2, num_anchors=10)
    onnx_path, _ = export_torch_model_to_onnx(model, tmp_path / "model.onnx")

    task_json_path = tmp_path / "task.json"
    task_data = {
        "schema_version": "1.0.0",
        "task_id": "test-task",
        "task_type": "object_detection",
        "categories": [
            {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle"},
            {"class_id": "can", "display_name": "Can", "prompt": "can"},
        ],
    }
    task_json_path.write_text(json.dumps(task_data, indent=2), encoding="utf-8")

    # Gate status: failed
    eval_json_path = tmp_path / "evaluation.json"
    eval_data = {
        "schema_version": "1.0.0",
        "run_id": "run-002",
        "dataset": {"dataset_id": "ds-001", "manifest_sha256": "0" * 64},
        "test": {"mAP50": 0.20},
        "export_parity": {"status": "failed"},
        "target_benchmarks": [],
        "gate": {"status": "failed"},
    }
    eval_json_path.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")

    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]
    dataset_ref = DatasetRef(dataset_id="ds-001", manifest_sha256="0" * 64)

    # Must raise ReleasePublicationError
    with pytest.raises(ReleasePublicationError, match="Model evaluation gate did not pass"):
        publish_model_package(
            package_id="model-failed-002",
            model_onnx_path=onnx_path,
            task_json_path=task_json_path,
            evaluation_json_path=eval_json_path,
            dataset_ref=dataset_ref,
            run_id="run-002",
            class_map=class_map,
            releases_dir=tmp_path / "releases",
            require_passed_gate=True,
        )


def test_model_registry_lifecycle(tmp_path: Path):
    reg_file = tmp_path / "registry.json"
    reg = ModelRegistry(reg_file)

    # Register candidate
    reg.register_candidate("model-001", tmp_path / "releases" / "model-001", "0" * 64)
    assert reg.entries["model-001"]["status"] == "candidate"

    # Cannot activate candidate before approval
    with pytest.raises(ValueError, match="Must be approved first"):
        reg.activate_model("model-001")

    # Approve
    reg.approve_model("model-001", evidence={"mAP50": 0.92, "verifier": "audit-suite"})
    assert reg.entries["model-001"]["status"] == "approved"

    # Activate
    reg.activate_model("model-001")
    assert reg.entries["model-001"]["status"] == "active"
    active = reg.get_active_model()
    assert active is not None
    assert active["package_id"] == "model-001"
