"""Evaluation and metrics module."""

from vision_model_factory.evaluation.evaluator import (
    TestEvaluationError,
    compute_dataset_manifest_sha,
    evaluate_test_split,
)
from vision_model_factory.evaluation.metrics import (
    compute_ap,
    compute_class_detection_metrics,
    match_detections_to_ground_truth,
)

__all__ = [
    "TestEvaluationError",
    "compute_ap",
    "compute_class_detection_metrics",
    "compute_dataset_manifest_sha",
    "evaluate_test_split",
    "match_detections_to_ground_truth",
]
