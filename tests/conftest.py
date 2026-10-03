"""Pytest fixtures for Vision Model Factory tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_model_factory.contracts.hashing import compute_sha256_file


@pytest.fixture
def sample_task_spec_dict():
    return {
        "schema_version": "1.0.0",
        "task_id": "test-task-001",
        "task_type": "object_detection",
        "categories": [
            {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle on conveyor"},
            {"class_id": "can", "display_name": "Can", "prompt": "metal can on conveyor"},
        ],
    }


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path, sample_task_spec_dict: dict) -> Path:
    """Create a fully valid immutable Dataset Package for tests."""
    ds_dir = tmp_path / "test_dataset_pkg"
    ds_dir.mkdir(parents=True, exist_ok=True)
    images_dir = ds_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create task.json
    task_file = ds_dir / "task.json"
    with task_file.open("w", encoding="utf-8") as f:
        json.dump(sample_task_spec_dict, f, indent=2)
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
        "dataset_id": "ds-test-001",
        "created_at": "2026-10-01T15:10:10Z",
        "task": {"path": "task.json", "sha256": task_sha},
        "samples": {"path": "samples.jsonl", "sha256": samples_sha},
        "annotations": {"path": "annotations.jsonl", "sha256": ann_sha},
        "quality": {"path": "quality.json", "sha256": quality_sha},
    }
    manifest_file = ds_dir / "dataset.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    return ds_dir
