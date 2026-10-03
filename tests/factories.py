"""Shared builders for contract artifacts used across the test suite.

These produce complete, schema-valid artifacts. The gate inside an evaluation report is
always *derived* from a policy here rather than written by hand, so a test cannot ask for
a verdict the recorded measurements do not support — which is exactly the failure mode
`release.publisher` is supposed to refuse.
"""

from typing import Any, Dict, List, Optional

from vision_model_factory.contracts.gate_policy import GatePolicy, build_gate
from vision_model_factory.contracts.models import (
    DatasetRef,
    EvaluationReport,
    ExportParitySection,
    InferenceConfig,
    PerClassDetectionMetrics,
    TestEvaluationSection,
)

DEFAULT_CONFIG = InferenceConfig(
    score_threshold=0.25, nms_iou_threshold=0.45, max_detections=100, target_shape=[640, 640]
)


def make_gate_policy(
    policy_id: str = "test-policy-v1",
    min_map50: float = 0.5,
    min_map50_95: float = 0.4,
    require_export_parity_passed: bool = False,
    **extra: Any,
) -> GatePolicy:
    """A publication policy for tests. Thresholds are stated, never defaulted by code."""
    return GatePolicy(
        policy_id=policy_id,
        min_map50=min_map50,
        min_map50_95=min_map50_95,
        require_export_parity_passed=require_export_parity_passed,
        **extra,
    )


_UNSET = object()


def make_parity_section(
    *,
    status: str = "passed",
    inference_config: InferenceConfig = DEFAULT_CONFIG,
    ref_count: int = 3,
    ort_count: int = 3,
    matched: bool = True,
    tensor_passed: bool = True,
    self_test: Any = _UNSET,
    **overrides: Any,
) -> ExportParitySection:
    """Build an ExportParitySection; the contract decides whether 'passed' is legal.

    Deliberately permissive: a caller that asks for `status="passed"` with empty
    detections, or with a self-test that missed the perturbation, must be refused by the
    contract. That refusal is what several regression tests assert, so this helper must
    not pre-empt it. `self_test` therefore distinguishes "omitted" (gets a passing
    self-test, for a normal passing report) from an explicit `None` (stays absent), which
    is a case the contract is expected to reject.
    """
    if self_test is _UNSET:
        self_test = {"perturbation_px": 25.0, "detected_perturbation": True} if status == "passed" else None
    payload: Dict[str, Any] = {
        "status": status,
        "method": "pytorch_vs_onnxruntime_cpu/letterbox_decode_parity_v1",
        "input_reference": "dataset:s-003:letterbox_rgb_u8_v1",
        "matching_method": "greedy_one_to_one_by_class_iou_then_score",
        "tolerances": {"tensor_atol": 1e-2, "score_atol": 1e-3, "box_atol": 1.0},
        "raw_tensor": {
            "max_abs_diff": 1e-4 if tensor_passed else 1.0,
            "mean_abs_diff": 1e-5 if tensor_passed else 0.5,
            "passed": tensor_passed,
        },
        "detections": {
            "ref_count": ref_count,
            "ort_count": ort_count,
            "matched": matched,
            "max_score_diff": 0.0,
            "max_box_diff": 0.0,
        },
        "self_test": self_test,
        "exceptions": [],
        "inference_config": inference_config,
    }
    payload.update(overrides)
    return ExportParitySection.model_validate(payload)


def make_evaluation_report(
    *,
    run_id: str = "run-001",
    dataset_id: str = "ds-001",
    manifest_sha256: str = "a" * 64,
    mAP50: float = 0.9,
    mAP50_95: float = 0.7,
    classes: Optional[Dict[str, PerClassDetectionMetrics]] = None,
    policy: Optional[GatePolicy] = None,
    parity: Optional[ExportParitySection] = None,
    inference_config: InferenceConfig = DEFAULT_CONFIG,
    sample_count: int = 2,
    target_benchmarks: Optional[List[Any]] = None,
) -> EvaluationReport:
    """Build a complete EvaluationReport whose gate is derived, never hand-written.

    A test that wants a failing gate lowers `mAP50` or tightens the policy; it does not
    edit a `status` string. The fixture then has the same shape as real evaluator output,
    so the publisher's re-derivation check is exercised instead of accidentally bypassed.
    """
    if classes is None:
        classes = {
            "bottle": PerClassDetectionMetrics(
                class_id="bottle", ap50=mAP50, ap50_95=mAP50_95,
                precision=mAP50, recall=mAP50, total_gt=4, total_pred=4,
            ),
            "can": PerClassDetectionMetrics(
                class_id="can", ap50=mAP50, ap50_95=mAP50_95,
                precision=mAP50, recall=mAP50, total_gt=3, total_pred=3,
            ),
        }
    policy = policy or make_gate_policy()
    parity = parity or make_parity_section(inference_config=inference_config)

    test_section = TestEvaluationSection(
        protocol_id="locked_test_per_image_ap_v1",
        ground_truth="human_verified",
        sample_count=sample_count,
        gt_annotation_count=sum(c.total_gt for c in classes.values()),
        excluded_partial_count=0,
        mAP50=mAP50,
        mAP50_95=mAP50_95,
        classes=classes,
        inference_config=inference_config,
    )
    return EvaluationReport(
        schema_version="1.0.0",
        run_id=run_id,
        dataset=DatasetRef(dataset_id=dataset_id, manifest_sha256=manifest_sha256),
        test=test_section,
        export_parity=parity,
        target_benchmarks=target_benchmarks or [],
        gate=build_gate(policy, test_section, parity, target_benchmarks or []),
    )


__all__ = [
    "DEFAULT_CONFIG",
    "make_evaluation_report",
    "make_gate_policy",
    "make_parity_section",
]
