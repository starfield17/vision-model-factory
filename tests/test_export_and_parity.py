"""Tests for ONNX export, parity verification, and INT8 calibration checks."""

from pathlib import Path

import pytest

from vision_model_factory.contracts.models import ClassMapItem, FileRef, SampleRecord
from vision_model_factory.export.exporter import export_torch_model_to_onnx
from vision_model_factory.export.parity import check_export_parity
from vision_model_factory.export.quantization import benchmark_onnx_model, validate_calibration_samples
from vision_model_factory.trainers.yolo import TinyYoloMockNet


def test_export_and_parity_verification(tmp_path: Path):
    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]
    model = TinyYoloMockNet(num_classes=2, num_anchors=20)
    onnx_file = tmp_path / "model.onnx"

    out_path, sha256 = export_torch_model_to_onnx(
        model=model,
        output_path=onnx_file,
        input_shape=(1, 3, 640, 640),
        input_name="images",
        output_name="output0",
    )

    assert out_path.is_file()
    assert len(sha256) == 64

    # Run parity verification
    parity_result = check_export_parity(
        torch_model=model,
        onnx_path=out_path,
        class_map=class_map,
        test_input_shape=(1, 3, 640, 640),
        tensor_atol=1e-3,
    )

    assert parity_result["status"] == "passed"
    assert parity_result["raw_tensor"]["passed"] is True
    assert parity_result["detections"]["matched"] is True

    # Benchmark ONNX model
    bench = benchmark_onnx_model(out_path, input_shape=(1, 3, 640, 640), warmup_runs=2, benchmark_runs=5)
    assert bench["latency_ms_p50"] > 0.0
    assert bench["latency_ms_p95"] > 0.0


def test_quantization_calibration_sample_split_guard():
    valid_samples = [
        SampleRecord(
            sample_id="s1",
            file=FileRef(path="images/1.jpg", sha256="0" * 64),
            width=640,
            height=480,
            group_id="g1",
            split="train",
            annotation_status="complete_verified",
        )
    ]
    validate_calibration_samples(valid_samples)

    invalid_samples = [
        SampleRecord(
            sample_id="s2",
            file=FileRef(path="images/2.jpg", sha256="0" * 64),
            width=640,
            height=480,
            group_id="g2",
            split="val",
            annotation_status="complete_verified",
        )
    ]
    with pytest.raises(ValueError, match="Calibration forbidden on non-train split"):
        validate_calibration_samples(invalid_samples)
