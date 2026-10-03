"""Pytest fixtures for Vision Model Factory tests."""

from pathlib import Path

import pytest


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
    from factories import write_dataset_package

    return write_dataset_package(tmp_path / "test_dataset_pkg", task_spec=sample_task_spec_dict)
