"""Tests for command-line interface execution."""

import json
from pathlib import Path
from unittest.mock import patch

from vision_model_factory.cli import main
from vision_model_factory.contracts.models import ClassMapItem, DatasetRef
from vision_model_factory.export.exporter import export_torch_model_to_onnx
from vision_model_factory.release.publisher import publish_model_package
from vision_model_factory.trainers.yolo import TinyYoloMockNet


def test_cli_check_boundaries(capsys):
    with patch("sys.argv", ["model-factory", "check-boundaries"]):
        main()
    captured = capsys.readouterr()
    assert "OK: All architectural boundary checks passed." in captured.out


def test_cli_validate_dataset(synthetic_dataset_dir: Path, capsys):
    with patch("sys.argv", ["model-factory", "validate-dataset", str(synthetic_dataset_dir)]):
        main()
    captured = capsys.readouterr()
    assert "OK: Dataset package 'ds-test-001' valid" in captured.out


def test_cli_export(tmp_path: Path, capsys):
    onnx_out = tmp_path / "cli_model.onnx"
    with patch("sys.argv", ["model-factory", "export", "dummy_path", str(onnx_out), "--classes", "2"]):
        main()
    captured = capsys.readouterr()
    assert "Exported ONNX model to" in captured.out
    assert onnx_out.is_file()


def test_cli_validate_model(tmp_path: Path, capsys):
    # Publish a model package first
    model = TinyYoloMockNet(num_classes=2, num_anchors=10)
    onnx_path, _ = export_torch_model_to_onnx(model, tmp_path / "model.onnx")

    task_json_path = tmp_path / "task.json"
    task_json_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "task_id": "test-task",
                "task_type": "object_detection",
                "categories": [
                    {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle"},
                    {"class_id": "can", "display_name": "Can", "prompt": "can"},
                ],
            }
        ),
        encoding="utf-8",
    )

    eval_json_path = tmp_path / "evaluation.json"
    eval_json_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "run-001",
                "dataset": {"dataset_id": "ds-001", "manifest_sha256": "0" * 64},
                "test": {"mAP50": 0.90},
                "export_parity": {"status": "passed"},
                "target_benchmarks": [],
                "gate": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )

    pub_dir, _ = publish_model_package(
        package_id="model-cli-001",
        model_onnx_path=onnx_path,
        task_json_path=task_json_path,
        evaluation_json_path=eval_json_path,
        dataset_ref=DatasetRef(dataset_id="ds-001", manifest_sha256="0" * 64),
        run_id="run-001",
        class_map=[ClassMapItem(index=0, class_id="bottle"), ClassMapItem(index=1, class_id="can")],
        releases_dir=tmp_path / "releases",
    )

    with patch("sys.argv", ["model-factory", "validate-model", str(pub_dir)]):
        main()
    captured = capsys.readouterr()
    assert "OK: Model package 'model-cli-001' valid" in captured.out
