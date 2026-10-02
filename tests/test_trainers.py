"""Tests for training adapters and YOLO baseline preparation."""

import json
from pathlib import Path

import yaml

from vision_model_factory.contracts.models import ClassMapItem, ParamsSpec, TaskSpec
from vision_model_factory.trainers.base import TrainerConfig
from vision_model_factory.trainers.registry import TrainerRegistry
from vision_model_factory.trainers.yolo import YoloTrainerAdapter


def test_trainer_registry():
    registry = TrainerRegistry()
    assert registry.is_adapter_allowed("yolo_detection_v1") is True
    assert registry.is_adapter_allowed("unknown_adapter") is False
    assert registry.is_model_allowed("yolo_detection_v1", "mock_yolo_v1") is True
    assert registry.is_model_allowed("yolo_detection_v1", "yolo26n.pt") is True
    assert registry.is_model_allowed("yolo_detection_v1", "yolo11n.pt") is True
    assert registry.is_model_allowed("yolo_detection_v1", "unwhitelisted_model.pt") is False
    expected_sha = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
    assert registry.is_checkpoint_allowed("yolo26n.pt", expected_sha) is True


def test_yolo_dataset_preparation_and_partial_exclusion(synthetic_dataset_dir: Path, tmp_path: Path):
    from vision_model_factory.contracts.hashing import compute_sha256_file
    img_sha = compute_sha256_file(synthetic_dataset_dir / "images" / "img1.jpg")

    # Add a sample with annotation_status: "partial" to synthetic dataset
    samples_file = synthetic_dataset_dir / "samples.jsonl"
    partial_sample = {
        "sample_id": "s-partial-001",
        "file": {"path": "images/img1.jpg", "sha256": img_sha},
        "width": 640,
        "height": 480,
        "group_id": "grp-partial",
        "split": "train",
        "annotation_status": "partial",
    }
    with samples_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(partial_sample) + "\n")

    # Update samples SHA in dataset.json
    from vision_model_factory.contracts.hashing import compute_sha256_file
    new_sha = compute_sha256_file(samples_file)
    ds_json_path = synthetic_dataset_dir / "dataset.json"
    with ds_json_path.open("r", encoding="utf-8") as f:
        ds_data = json.load(f)
    ds_data["samples"]["sha256"] = new_sha
    with ds_json_path.open("w", encoding="utf-8") as f:
        json.dump(ds_data, f, indent=2)

    # Prepare dataset
    adapter = YoloTrainerAdapter()
    work_dir = tmp_path / "yolo_work"
    task = TaskSpec(
        schema_version="1.0.0",
        task_id="t1",
        task_type="object_detection",
        categories=[
            {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle"},
            {"class_id": "can", "display_name": "Can", "prompt": "can"},
        ],
    )
    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]

    yaml_path, stats = adapter.prepare_dataset(synthetic_dataset_dir, work_dir, task, class_map)

    assert yaml_path.is_file()
    assert stats["excluded_partial_samples"] == 1
    assert stats["train_samples"] == 1  # 1 valid train sample (excluding partial)

    with yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    assert data_cfg["names"] == {0: "bottle", 1: "can"}


def test_yolo_mock_training_execution(synthetic_dataset_dir: Path, tmp_path: Path):
    adapter = YoloTrainerAdapter()
    work_dir = tmp_path / "mock_training_run"

    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]
    params = ParamsSpec(imgsz=640, batch=8, epochs=1, seed=42, lr0=0.01, mosaic=1.0)
    config = TrainerConfig(
        experiment_id="exp-001",
        run_id="run-001",
        dataset_dir=synthetic_dataset_dir,
        output_dir=work_dir,
        params=params,
        class_map=class_map,
        model_id="mock_yolo_v1",
        checkpoint_sha256="0" * 64,
        mock_mode=True,
    )

    result = adapter.train(config)
    assert result.status == "succeeded"
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.is_file()
    assert result.checkpoint_sha256 is not None
    assert len(result.checkpoint_sha256) == 64
    assert "mAP50" in result.val_metrics
    assert result.val_metrics["mAP50"] > 0.0
