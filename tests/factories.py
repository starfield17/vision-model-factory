"""Shared builders for contract artifacts used across the test suite.

These produce complete, schema-valid artifacts. The gate inside an evaluation report is
always *derived* from a policy here rather than written by hand, so a test cannot ask for
a verdict the recorded measurements do not support — which is exactly the failure mode
`release.publisher` is supposed to refuse.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from vision_model_factory.contracts.gate_policy import GatePolicy, build_gate
from vision_model_factory.contracts.hashing import compute_sha256_file
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
    parity_split: Any = "audit",
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

    `parity_split` defaults to `"audit"` because a passing parity claim must attribute its
    corpus. `"test"` is not a legal value in the contract, so a caller that wants to prove
    the refusal has to ask for it explicitly.
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
            "max_rel_diff": 1e-6 if tensor_passed else 1.0,
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
        "parity_split": parity_split,
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


DEFAULT_TASK_SPEC = {
    "schema_version": "1.0.0",
    "task_id": "test-task-001",
    "task_type": "object_detection",
    "categories": [
        {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle on conveyor"},
        {"class_id": "can", "display_name": "Can", "prompt": "metal can on conveyor"},
    ],
}


def write_dataset_package(
    ds_dir: Path,
    task_spec: Optional[dict] = None,
    with_audit: bool = True,
    dataset_id: str = "ds-test-001",
) -> Path:
    """Write a fully valid immutable Dataset Package into `ds_dir`, returning its path.

    Lives here rather than only inside a fixture because publication re-validates the data
    package it releases: a publishing test needs a package whose real manifest digest it
    knows, not a placeholder hash.

    `with_audit` appends `audit`-split samples. Export parity must be measured on a corpus
    that is not the locked `test` split, and `audit` exists in the Data vocabulary for spare
    samples of exactly this kind. They carry no annotations and are marked `partial`, the
    only non-complete status the Data-owned enum offers, which is also what excludes them
    from training and from scoring.
    """
    task_spec = task_spec if task_spec is not None else DEFAULT_TASK_SPEC

    ds_dir.mkdir(parents=True, exist_ok=True)
    images_dir = ds_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create task.json
    task_file = ds_dir / "task.json"
    with task_file.open("w", encoding="utf-8") as f:
        json.dump(task_spec, f, indent=2)
    task_sha = compute_sha256_file(task_file)

    # 2. Create image files (640x480)
    img1_file = images_dir / "img1.jpg"
    img2_file = images_dir / "img2.jpg"
    img3_file = images_dir / "img3.jpg"
    img4_file = images_dir / "img4.jpg"

    arr1 = np.full((480, 640, 3), 100, dtype=np.uint8)
    arr2 = np.full((480, 640, 3), 150, dtype=np.uint8)
    arr3 = np.full((480, 640, 3), 200, dtype=np.uint8)
    arr4 = np.full((480, 640, 3), 220, dtype=np.uint8)

    Image.fromarray(arr1).save(img1_file)
    Image.fromarray(arr2).save(img2_file)
    Image.fromarray(arr3).save(img3_file)
    Image.fromarray(arr4).save(img4_file)

    img1_sha = compute_sha256_file(img1_file)
    img2_sha = compute_sha256_file(img2_file)
    img3_sha = compute_sha256_file(img3_file)
    img4_sha = compute_sha256_file(img4_file)

    # 3. Create samples.jsonl (train, val, and two test samples)
    samples = [
        {
            "sample_id": "s-001",
            "file": {"path": "images/img1.jpg", "sha256": img1_sha},
            "width": 640,
            "height": 480,
            "group_id": "grp-1",
            "split": "train",
            "annotation_status": "complete_verified",
        },
        {
            "sample_id": "s-002",
            "file": {"path": "images/img2.jpg", "sha256": img2_sha},
            "width": 640,
            "height": 480,
            "group_id": "grp-2",
            "split": "val",
            "annotation_status": "complete_verified",
        },
        {
            "sample_id": "s-003",
            "file": {"path": "images/img3.jpg", "sha256": img3_sha},
            "width": 640,
            "height": 480,
            "group_id": "grp-3",
            "split": "test",
            "annotation_status": "complete_verified",
        },
        # Second test-split sample. A locked test set must carry ground truth for every
        # class the model declares, otherwise macro-averaged mAP is not well defined and
        # the evaluator refuses to score such a package at all.
        {
            "sample_id": "s-004",
            "file": {"path": "images/img4.jpg", "sha256": img4_sha},
            "width": 640,
            "height": 480,
            "group_id": "grp-4",
            "split": "test",
            "annotation_status": "complete_verified",
        },
    ]
    samples_file = ds_dir / "samples.jsonl"
    with samples_file.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    samples_sha = compute_sha256_file(samples_file)

    # 4. Create annotations.jsonl
    annotations = [
        {
            "annotation_id": "ann-001",
            "sample_id": "s-001",
            "class_id": "bottle",
            "bbox_xyxy": [50.0, 60.0, 200.0, 300.0],
            "origin": "human",
            "annotator_run_id": "run-ann-01",
            "review_state": "human_verified",
            "score": 1.0,
        },
        {
            "annotation_id": "ann-002",
            "sample_id": "s-002",
            "class_id": "can",
            "bbox_xyxy": [100.0, 120.0, 250.0, 350.0],
            "origin": "human",
            "annotator_run_id": "run-ann-01",
            "review_state": "human_verified",
            "score": 1.0,
        },
        {
            "annotation_id": "ann-003",
            "sample_id": "s-003",
            "class_id": "bottle",
            "bbox_xyxy": [80.0, 90.0, 220.0, 310.0],
            "origin": "human",
            "annotator_run_id": "run-ann-01",
            "review_state": "human_verified",
            "score": 1.0,
        },
        {
            "annotation_id": "ann-004",
            "sample_id": "s-004",
            "class_id": "can",
            "bbox_xyxy": [300.0, 40.0, 420.0, 180.0],
            "origin": "human",
            "annotator_run_id": "run-ann-01",
            "review_state": "human_verified",
            "score": 1.0,
        },
    ]
    ann_file = ds_dir / "annotations.jsonl"
    with ann_file.open("w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a) + "\n")
    ann_sha = compute_sha256_file(ann_file)

    # 5. Create quality.json
    quality_content = {
        "schema_version": "1.0.0",
        "annotation_runs": [],
        "reviewer_runs": [],
        "audit": {"sample_count": 4},
        "gate": {"status": "passed"},
    }
    quality_file = ds_dir / "quality.json"
    with quality_file.open("w", encoding="utf-8") as f:
        json.dump(quality_content, f, indent=2)
    quality_sha = compute_sha256_file(quality_file)

    # 6. Create dataset.json manifest
    dataset_manifest = {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "created_at": "2026-10-01T15:10:10Z",
        "task": {"path": "task.json", "sha256": task_sha},
        "samples": {"path": "samples.jsonl", "sha256": samples_sha},
        "annotations": {"path": "annotations.jsonl", "sha256": ann_sha},
        "quality": {"path": "quality.json", "sha256": quality_sha},
    }
    manifest_file = ds_dir / "dataset.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    if with_audit:
        audit_rows = []
        for i, value in enumerate((170, 190), start=5):
            audit_file = images_dir / f"img{i}.jpg"
            Image.fromarray(np.full((480, 640, 3), value, dtype=np.uint8)).save(audit_file)
            audit_rows.append({
                "sample_id": f"s-00{i}",
                "file": {"path": f"images/img{i}.jpg", "sha256": compute_sha256_file(audit_file)},
                "width": 640,
                "height": 480,
                "group_id": f"grp-{i}",
                "split": "audit",
                "annotation_status": "partial",
            })
        with samples_file.open("a", encoding="utf-8") as f:
            for row in audit_rows:
                f.write(json.dumps(row) + "\n")
        # The manifest digests the samples file, so it has to be rewritten to cover the
        # appended rows instead of describing the earlier content.
        dataset_manifest["samples"]["sha256"] = compute_sha256_file(samples_file)
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(dataset_manifest, f, indent=2)

    return ds_dir
