"""Tests for evaluation metrics and locked test split evaluation."""

from pathlib import Path

import numpy as np

from vision_model_factory.contracts.models import ClassMapItem, SampleRecord
from vision_model_factory.evaluation.evaluator import evaluate_test_split
from vision_model_factory.evaluation.metrics import (
    compute_ap,
    match_detections_to_ground_truth,
)


def test_match_detections_to_ground_truth():
    gt_boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 100.0, 100.0],
    ]
    # Prediction 1: perfect match with gt 0, score 0.9
    # Prediction 2: duplicate match with gt 0, score 0.8 -> should be marked FP (already matched)
    # Prediction 3: partial match with gt 1 (IoU > 0.5), score 0.7 -> marked TP
    # Prediction 4: background false positive, score 0.6 -> marked FP
    pred_boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [52.0, 52.0, 100.0, 100.0],
        [200.0, 200.0, 300.0, 300.0],
    ]
    pred_scores = [0.9, 0.8, 0.7, 0.6]

    tp, fp = match_detections_to_ground_truth(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

    assert tp[0] is np.True_  # matches gt 0
    assert fp[1] is np.True_  # gt 0 already claimed
    assert tp[2] is np.True_  # matches gt 1
    assert fp[3] is np.True_  # background


def test_compute_ap():
    # Perfect detector: precision=1.0 at all recall points
    prec = np.array([1.0, 1.0, 1.0])
    rec = np.array([0.33, 0.66, 1.0])
    ap = compute_ap(prec, rec)
    assert abs(ap - 1.0) < 1e-4


def test_evaluate_test_split(synthetic_dataset_dir: Path):
    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]

    # Predict function returning ground truth prediction for sample s-003
    def mock_predict(sample: SampleRecord):
        if sample.sample_id == "s-003":
            return [
                {
                    "class_id": "bottle",
                    "score": 0.95,
                    "bbox_xyxy": [80.0, 90.0, 220.0, 310.0],
                }
            ]
        return []

    # Run locked test split evaluation
    report = evaluate_test_split(
        dataset_dir=synthetic_dataset_dir,
        predict_fn=mock_predict,
        class_map=class_map,
        run_id="run-test-eval-01",
        min_map50_gate=0.5,
    )

    assert report.schema_version == "1.0.0"
    assert report.test["sample_count"] == 1
    assert report.test["mAP50"] > 0.0
    assert report.gate["status"] == "passed"
    assert report.gate["checks"][0]["passed"] is True
