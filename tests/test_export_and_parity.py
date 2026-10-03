"""Tests for ONNX export, semantic parity verification, and INT8 calibration guards."""

from pathlib import Path

import numpy as np
import pytest
import torch
from factories import make_parity_section
from pydantic import ValidationError as PydanticValidationError

from vision_model_factory.contracts.models import ClassMapItem, ExportParitySection, FileRef, SampleRecord
from vision_model_factory.export.exporter import (
    export_torch_model_to_onnx,
    export_yolo_checkpoint_to_onnx,
)
from vision_model_factory.export.parity import (
    PERTURBATION_PX,
    ParityInputError,
    check_export_parity,
    match_detection_sets,
    perturb_detections,
)
from vision_model_factory.export.quantization import benchmark_onnx_model, validate_calibration_samples
from vision_model_factory.trainers.yolo import TinyYoloMockNet

CLASS_MAP = [ClassMapItem(index=0, class_id="bottle"), ClassMapItem(index=1, class_id="can")]


def _write_samples(tmp_path: Path, count: int = 2):
    """Create real image bytes and SampleRecords for them.

    Parity must run on images, not noise: noise produces zero detections on a real
    detector and a zero-versus-zero comparison verifies nothing.
    """
    from PIL import Image

    from vision_model_factory.contracts.hashing import compute_sha256_file

    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(count):
        # A hard-edged pattern rather than a solid fill, so resize is not a no-op and a
        # coordinate error cannot hide inside a uniform field.
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        arr[:, :] = 40 + 30 * i
        arr[100 + 20 * i : 300, 120 : 480] = 200 - 25 * i
        path = images_dir / f"px{i}.jpg"
        Image.fromarray(arr).save(path)
        samples.append(
            SampleRecord(
                sample_id=f"px-{i}",
                file=FileRef(path=f"images/px{i}.jpg", sha256=compute_sha256_file(path)),
                width=640,
                height=480,
                group_id=f"g-{i}",
                split="test",
                annotation_status="complete_verified",
            )
        )
    return samples


class ShiftedBoxNet(TinyYoloMockNet):
    """Wraps a net and offsets the cx channel of its YOLO-format output.

    Stands in for an export whose coordinate path is subtly wrong: identical scores,
    boxes in the wrong place. Parity must refuse it.
    """

    def __init__(self, inner: TinyYoloMockNet, dx: float):
        super().__init__(num_classes=inner.num_classes, num_anchors=inner.num_anchors)
        self.inner = inner
        self.dx = dx

    def forward(self, x):
        out = self.inner(x)
        out = out.clone()
        out[:, 0, :] = out[:, 0, :] + self.dx
        return out


def _export_mock(tmp_path: Path, num_anchors: int = 20):
    model = TinyYoloMockNet(num_classes=2, num_anchors=num_anchors)
    out_path, sha256 = export_torch_model_to_onnx(
        model=model,
        output_path=tmp_path / "model.onnx",
        input_shape=(1, 3, 640, 640),
        input_name="images",
        output_name="output0",
    )
    return model, out_path, sha256


def test_export_and_parity_verification(tmp_path: Path):
    model, onnx_file, sha256 = _export_mock(tmp_path)
    samples = _write_samples(tmp_path)

    result = check_export_parity(
        torch_model=model,
        onnx_path=onnx_file,
        class_map=CLASS_MAP,
        samples=samples,
        dataset_dir=tmp_path,
        tensor_atol=1e-3,
    )

    assert onnx_file.is_file() and len(sha256) == 64
    assert result["status"] == "passed"
    assert result["raw_tensor"]["passed"] is True
    # Non-empty on both sides is the point: an empty comparison would have "matched" too.
    assert result["detections"]["ref_count"] > 0
    assert result["detections"]["ort_count"] > 0
    assert result["detections"]["matched"] is True
    # The reference input names the actual samples and the reference preprocessor.
    assert "px-0" in result["input_reference"]
    assert "letterbox_rgb_u8_v1" in result["input_reference"]
    # And the comparator proved it can see a shifted box.
    assert result["self_test"]["detected_perturbation"] is True
    assert result["self_test"]["perturbation_px"] == PERTURBATION_PX

    # A parity report must be recordable in the contract without further editing.
    section = ExportParitySection.model_validate(result)
    assert section.status == "passed"

    bench = benchmark_onnx_model(onnx_file, input_shape=(1, 3, 640, 640), warmup_runs=2, benchmark_runs=5)
    assert bench["latency_ms_p50"] > 0.0
    assert bench["latency_ms_p95"] > 0.0


def test_parity_refuses_inputs_that_produce_no_detections(tmp_path: Path):
    """Zero detections on both sides is an error, never a pass.

    The previous implementation computed `matched = len(ref) == len(ort)`, so 0 == 0 was
    a passing parity report and shipped a release on the strength of it.
    """
    class SilentNet(TinyYoloMockNet):
        """Valid-shaped output whose score channels are exactly zero.

        The `0.0 * feat` term keeps the graph a genuine function of its input: a fully
        constant output is pruned to a zero-input graph, which is a different (also
        unacceptable) failure covered separately below.
        """

        def forward(self, x):
            feat = self.conv(x)[:, :, 0, : self.num_anchors]
            out = torch.zeros(
                (x.shape[0], 4 + self.num_classes, self.num_anchors), dtype=torch.float32, device=x.device
            )
            out[:, :4, :] = 100.0  # plausible boxes, but no class ever clears a threshold
            return out + 0.0 * feat

    silent = SilentNet(num_classes=2, num_anchors=20)
    out_path, _ = export_torch_model_to_onnx(silent, tmp_path / "silent.onnx")
    samples = _write_samples(tmp_path, count=1)

    with pytest.raises(ParityInputError, match="zero detections on both sides"):
        check_export_parity(
            torch_model=silent,
            onnx_path=out_path,
            class_map=CLASS_MAP,
            samples=samples,
            dataset_dir=tmp_path,
        )

    with pytest.raises(ParityInputError, match="at least one real reference sample"):
        check_export_parity(
            torch_model=silent, onnx_path=out_path, class_map=CLASS_MAP, samples=[], dataset_dir=tmp_path
        )


def test_parity_detects_a_shifted_export(tmp_path: Path):
    """An export whose coordinates are off by a visible amount must fail, loudly.

    This is the acceptance property: parity exists to catch a broken coordinate path, so a
    graph that returns the same scores at shifted positions cannot be 'passed'. The
    "export" here is the reference net wrapped so its cx channel is offset by 30px, which
    is exactly the class of defect parity must refuse.
    """
    model, clean_path, _ = _export_mock(tmp_path)
    samples = _write_samples(tmp_path, count=1)

    baseline = check_export_parity(
        torch_model=model,
        onnx_path=clean_path,
        class_map=CLASS_MAP,
        samples=samples,
        dataset_dir=tmp_path,
        box_atol=1.0,
    )
    assert baseline["detections"]["matched"] is True

    shifted_module = ShiftedBoxNet(model, dx=30.0)
    shifted_path, _ = export_torch_model_to_onnx(shifted_module, tmp_path / "shifted.onnx")

    result = check_export_parity(
        torch_model=model,
        onnx_path=shifted_path,
        class_map=CLASS_MAP,
        samples=samples,
        dataset_dir=tmp_path,
        box_atol=1.0,
    )
    assert result["detections"]["matched"] is False
    assert result["detections"]["max_box_diff"] > 25.0
    assert result["status"] == "failed"
    assert result["raw_tensor"]["passed"] is False
    # Both sides still found objects: the failure came from disagreeing coordinates,
    # not from an empty comparison.
    assert result["detections"]["ref_count"] > 0
    assert result["detections"]["ort_count"] > 0


def test_self_test_proves_the_comparator_is_not_vacuous():
    """The perturbation self-test is what makes a 'passed' verdict mean something."""
    dets = [
        {"class_id": "bottle", "score": 0.9, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
        {"class_id": "can", "score": 0.7, "bbox_xyxy": [80.0, 80.0, 120.0, 140.0]},
    ]

    same = match_detection_sets(dets, dets, score_atol=1e-3, box_atol=1.0)
    assert same["within_tolerance"] is True

    shifted = match_detection_sets(dets, perturb_detections(dets), score_atol=1e-3, box_atol=1.0)
    assert shifted["within_tolerance"] is False
    assert shifted["max_box_diff"] == pytest.approx(PERTURBATION_PX)

    # Equal counts at different coordinates must fail: count-equality alone was the bug.
    reclassified = [dict(d, class_id="can" if d["class_id"] == "bottle" else "bottle") for d in dets]
    assert match_detection_sets(dets, reclassified, 1e-3, 1.0)["within_tolerance"] is False

    # Same box count, different positions, same class order -> still a failure.
    moved = [dict(d, bbox_xyxy=[c + 40.0 for c in d["bbox_xyxy"]]) for d in dets]
    compared = match_detection_sets(dets, moved, score_atol=1e-3, box_atol=1.0)
    assert compared["counts_match"] is True
    assert compared["within_tolerance"] is False


def test_contract_forbids_a_vacuous_parity_pass():
    """`status="passed"` is structurally unreachable without substantive evidence."""
    # Empty detections on either side.
    for kwargs in ({"ref_count": 0, "ort_count": 0}, {"ref_count": 3, "ort_count": 0},
                   {"ref_count": 0, "ort_count": 3}, {"ref_count": 2, "ort_count": 3}):
        with pytest.raises(PydanticValidationError):
            make_parity_section(status="passed", **kwargs)

    # A self-test that failed to notice the injected shift means the comparator is blind.
    with pytest.raises(PydanticValidationError, match="cannot see coordinate errors"):
        make_parity_section(status="passed", self_test={"perturbation_px": 25.0, "detected_perturbation": False})

    # No self-test at all.
    with pytest.raises(PydanticValidationError, match="without a self_test"):
        make_parity_section(status="passed", self_test=None)

    # Failing raw tensor or unmatched detections.
    with pytest.raises(PydanticValidationError):
        make_parity_section(status="passed", tensor_passed=False)
    with pytest.raises(PydanticValidationError):
        make_parity_section(status="passed", matched=False)

    # A 'failed' verdict stays representable, including "parity was never run".
    make_parity_section(status="failed", ref_count=0, ort_count=0, matched=False, tensor_passed=False,
                        self_test=None)


def test_quantization_calibration_sample_split_guard():
    def sample(sid: str, split: str) -> SampleRecord:
        return SampleRecord(
            sample_id=sid,
            file=FileRef(path=f"images/{sid}.jpg", sha256="0" * 64),
            width=640,
            height=480,
            group_id=f"g{sid}",
            split=split,
            annotation_status="complete_verified",
        )

    validate_calibration_samples([sample("s1", "train")])

    with pytest.raises(ValueError, match="Calibration forbidden on non-train split"):
        validate_calibration_samples([sample("s2", "val")])
    with pytest.raises(ValueError, match="Calibration forbidden on non-train split"):
        validate_calibration_samples([sample("s3", "test")])


def test_yolo_checkpoint_export_and_parity(tmp_path: Path):
    """Integration: export the real pretrained checkpoint and prove parity on real images.

    Requires the locally fetched `yolo26n.pt`. This is the only path that exercises a real
    detector's coordinate recovery; the mock net above cannot prove it.
    """
    ckpt_path = Path("yolo26n.pt")
    if not ckpt_path.is_file():
        pytest.skip("requires the local yolo26n.pt checkpoint (fetched by the training run)")

    from ultralytics import YOLO

    yolo_model = YOLO(str(ckpt_path))
    class_map = [ClassMapItem(index=i, class_id=str(name)) for i, name in yolo_model.names.items()]

    out_onnx = tmp_path / "yolo26n_exported.onnx"
    out_path, sha256 = export_yolo_checkpoint_to_onnx(checkpoint_path=ckpt_path, output_path=out_onnx, imgsz=640)
    assert out_path.is_file() and len(sha256) == 64

    # Pretrained COCO weights detect nothing on synthetic fills, so use real photographs
    # when they are present locally; otherwise assert the export interface only.
    images = sorted(Path("datasets/ds-african-wildlife-v1/images/test").glob("*.jpg"))[:3]
    if len(images) < 2:
        pytest.skip("parity on real images requires the local wildlife dataset")

    import shutil

    from vision_model_factory.contracts.hashing import compute_sha256_file

    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    samples = []
    for i, src in enumerate(images):
        dst = tmp_path / "images" / f"real{i}.jpg"
        shutil.copy2(src, dst)
        from PIL import Image

        with Image.open(dst) as im:
            w, h = im.size
        samples.append(
            SampleRecord(
                sample_id=f"real-{i}",
                file=FileRef(path=f"images/real{i}.jpg", sha256=compute_sha256_file(dst)),
                width=w,
                height=h,
                group_id=f"real-{i}",
                split="test",
                annotation_status="complete_verified",
            )
        )

    parity = check_export_parity(
        torch_model=yolo_model.model,
        onnx_path=out_path,
        class_map=class_map,
        samples=samples,
        dataset_dir=tmp_path,
        score_threshold=0.25,
        tensor_atol=5e-3,
    )
    assert parity["self_test"]["detected_perturbation"] is True
    assert parity["detections"]["ref_count"] > 0, "real images must produce detections to compare"
    # Report the measurement rather than assuming it: if this fails, the exported graph
    # disagrees with the reference and that is a release blocker, not a tolerance problem.
    assert parity["status"] == "passed", (
        f"real parity failed: max_abs_diff={parity['raw_tensor']['max_abs_diff']} "
        f"box_diff={parity['detections']['max_box_diff']} "
        f"ref={parity['detections']['ref_count']} ort={parity['detections']['ort_count']}"
    )
