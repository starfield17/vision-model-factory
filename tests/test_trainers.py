"""Tests for training adapters and YOLO baseline preparation."""

import json
from pathlib import Path

import yaml

from vision_model_factory.contracts.models import ClassMapItem, ParamsSpec
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
    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]

    # prepare_dataset takes no task argument: the authoritative task is read from the
    # validated package, so a caller cannot pair samples with a mismatched category order.
    yaml_path, stats = adapter.prepare_dataset(synthetic_dataset_dir, work_dir, class_map)

    assert yaml_path.is_file()
    # Every `partial` sample is excluded, whichever reason it exists for: the one appended
    # here, and the fixture's unlabelled audit samples, which are partial by construction.
    with samples_file.open("r", encoding="utf-8") as f:
        expected_partial = sum(
            1 for line in f if line.strip() and json.loads(line)["annotation_status"] == "partial"
        )
    assert expected_partial == 3  # 2 audit + s-partial-001, so the count below is not vacuous
    assert stats["excluded_partial_samples"] == expected_partial
    # One train sample carries labels; the partial and audit samples must not appear.
    assert stats["train_samples"] == 1

    with yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    assert data_cfg["names"] == {0: "bottle", 1: "can"}

    # Label files must exist for the included samples and carry indices bound to the
    # package category order (bottle=0, can=1).
    train_labels = list((work_dir / "yolo_data" / "labels" / "train").glob("*.txt"))
    assert len(train_labels) == 1
    index_used = {int(line.split()[0]) for line in train_labels[0].read_text().splitlines() if line}
    assert index_used == {0}

    # The excluded partial sample must leave neither image nor label behind.
    assert not (work_dir / "yolo_data" / "images" / "train" / "s-partial-001.jpg").exists()
    assert not (work_dir / "yolo_data" / "labels" / "train" / "s-partial-001.txt").exists()


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
