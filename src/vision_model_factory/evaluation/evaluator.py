"""Locked test split evaluator for independent model quality assessment."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
    EvaluationReport,
    SampleRecord,
)
from vision_model_factory.contracts.validators import validate_dataset_package
from vision_model_factory.evaluation.metrics import compute_class_detection_metrics


class TestEvaluationError(Exception):
    """Raised when test split evaluation constraints are violated."""


def evaluate_test_split(
    dataset_dir: Path,
    predict_fn: Callable[[SampleRecord], List[Dict[str, Any]]],
    class_map: List[ClassMapItem],
    run_id: str,
    export_parity_summary: Optional[Dict[str, Any]] = None,
    target_benchmarks: Optional[List[Dict[str, Any]]] = None,
    min_map50_gate: float = 0.5,
) -> EvaluationReport:
    """
    Perform locked evaluation strictly on the test split of a dataset package.

    Args:
        dataset_dir: Path to validated Dataset Package
        predict_fn: Function mapping SampleRecord to list of detection dicts:
                    [{"class_id": str, "score": float, "bbox_xyxy": [x1, y1, x2, y2]}]
        class_map: Model class map
        run_id: Identifier of the training run
        export_parity_summary: Evidence of export parity
        target_benchmarks: Latency / memory benchmarks
        min_map50_gate: Gate threshold for release validation

    Returns:
        EvaluationReport instance conforming to schema.
    """
    manifest, task, samples, annotations = validate_dataset_package(dataset_dir)

    # Filter test split
    test_samples: List[SampleRecord] = []
    excluded_partial_count = 0

    for s in samples:
        if s.split != "test":
            continue
        if s.annotation_status == "partial":
            excluded_partial_count += 1
            continue
        test_samples.append(s)

    if len(test_samples) == 0:
        raise TestEvaluationError("No valid non-partial samples found in test split for evaluation.")

    # Index ground-truth annotations for test samples
    test_sample_ids = {s.sample_id for s in test_samples}
    test_gts = [
        {"class_id": a.class_id, "bbox_xyxy": a.bbox_xyxy}
        for a in annotations
        if a.sample_id in test_sample_ids
    ]

    # Collect predictions across all test samples
    all_preds: List[Dict[str, Any]] = []
    for s in test_samples:
        preds = predict_fn(s)
        all_preds.extend(preds)

    # Calculate metrics per category
    categories = [cat.class_id for cat in task.categories]
    per_class_metrics: Dict[str, Any] = {}
    map50_list: List[float] = []
    map50_95_list: List[float] = []

    for cid in categories:
        res = compute_class_detection_metrics(all_preds, test_gts, class_id=cid)
        per_class_metrics[cid] = res
        map50_list.append(res["ap50"])
        map50_95_list.append(res["ap50_95"])

    mean_map50 = round(float(np.mean(map50_list)), 4) if map50_list else 0.0
    mean_map50_95 = round(float(np.mean(map50_95_list)), 4) if map50_95_list else 0.0

    test_summary = {
        "sample_count": len(test_samples),
        "excluded_partial_count": excluded_partial_count,
        "mAP50": mean_map50,
        "mAP50_95": mean_map50_95,
        "classes": per_class_metrics,
    }

    # Release gate verification
    gate_passed = mean_map50 >= min_map50_gate
    gate = {
        "status": "passed" if gate_passed else "failed",
        "checks": [
            {
                "metric": "mAP50",
                "threshold": min_map50_gate,
                "actual": mean_map50,
                "passed": gate_passed,
            }
        ],
    }

    report = EvaluationReport(
        schema_version="1.0.0",
        run_id=run_id,
        dataset=DatasetRef(
            dataset_id=manifest.dataset_id,
            manifest_sha256=compute_dataset_manifest_sha(dataset_dir),
        ),
        test=test_summary,
        export_parity=export_parity_summary or {"status": "skipped"},
        target_benchmarks=target_benchmarks or [],
        gate=gate,
    )

    return report


def compute_dataset_manifest_sha(dataset_dir: Path) -> str:
    """Compute SHA-256 digest of dataset.json manifest file."""
    from vision_model_factory.contracts.hashing import compute_sha256_file

    return compute_sha256_file(dataset_dir / "dataset.json")


class YoloOnnxPredictor:
    """Predictor using exported ONNX model and reference letterbox preprocessor + decoder."""

    def __init__(self, onnx_model_path: Path, class_map: List[ClassMapItem], dataset_dir: Path):
        import onnxruntime as ort

        self.dataset_dir = Path(dataset_dir)
        self.class_map = class_map
        self.session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, sample: SampleRecord) -> List[Dict[str, Any]]:
        from PIL import Image

        from vision_model_factory.export.decoder import yolo_xywh_scores_v1
        from vision_model_factory.export.preprocessor import letterbox_rgb_u8_v1

        img_path = self.dataset_dir / sample.file.path
        if not img_path.is_file():
            return []

        with Image.open(img_path) as pil_img:
            img_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        tensor, meta = letterbox_rgb_u8_v1(img_np, target_shape=(640, 640))
        out = self.session.run(None, {self.input_name: tensor})[0]

        return yolo_xywh_scores_v1(
            out,
            class_map=self.class_map,
            orig_shape=(sample.height, sample.width),
            preprocess_meta=meta,
            score_threshold=0.25,
            nms_iou_threshold=0.45,
            max_detections=100,
        )


class YoloTorchPredictor:
    """Predictor using PyTorch YOLO checkpoint and reference letterbox preprocessor + decoder."""

    def __init__(self, weights_path: Path, class_map: List[ClassMapItem], dataset_dir: Path):
        from ultralytics import YOLO

        self.dataset_dir = Path(dataset_dir)
        self.class_map = class_map
        m = YOLO(str(weights_path))
        self.model = m.model
        self.model.eval()

    def predict(self, sample: SampleRecord) -> List[Dict[str, Any]]:
        import torch
        from PIL import Image

        from vision_model_factory.export.decoder import yolo_xywh_scores_v1
        from vision_model_factory.export.preprocessor import letterbox_rgb_u8_v1

        img_path = self.dataset_dir / sample.file.path
        if not img_path.is_file():
            return []

        with Image.open(img_path) as pil_img:
            img_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        tensor_np, meta = letterbox_rgb_u8_v1(img_np, target_shape=(640, 640))
        tensor = torch.from_numpy(tensor_np)

        with torch.no_grad():
            out = self.model(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out_np = out.cpu().numpy()

        return yolo_xywh_scores_v1(
            out_np,
            class_map=self.class_map,
            orig_shape=(sample.height, sample.width),
            preprocess_meta=meta,
            score_threshold=0.25,
            nms_iou_threshold=0.45,
            max_detections=100,
        )
