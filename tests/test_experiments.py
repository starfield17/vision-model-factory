"""Tests for bounded experiment agent proposal validation and runner budget enforcement."""

from pathlib import Path

import pytest

from vision_model_factory.contracts.models import (
    DatasetRef,
    ExperimentSpec,
    ParamsSpec,
    TrainerSpec,
)
from vision_model_factory.experiments.agent import (
    AgentProposalViolation,
    validate_agent_proposal,
)
from vision_model_factory.experiments.policy import ExperimentPolicy
from vision_model_factory.experiments.runner import BudgetExceededError, ExperimentRunner


def test_agent_proposal_validation():
    policy = ExperimentPolicy(
        imgsz_range=(320, 1280),
        batch_range=(1, 64),
        epochs_range=(1, 300),
        lr0_range=(1e-5, 0.1),
        mosaic_range=(0.0, 1.0),
    )

    valid_proposal = {
        "schema_version": "1.0.0",
        "experiment_id": "exp-001",
        "dataset": {"dataset_id": "ds-001", "manifest_sha256": "0" * 64},
        "trainer": {
            "adapter_id": "yolo_detection_v1",
            "model_id": "mock_yolo_v1",
            "checkpoint_sha256": "0" * 64,
        },
        "params": {
            "imgsz": 640,
            "batch": 16,
            "epochs": 10,
            "seed": 42,
            "lr0": 0.01,
            "mosaic": 1.0,
        },
        "reason": "Test baseline configuration",
    }

    # Should validate cleanly
    spec = validate_agent_proposal(valid_proposal, policy)
    assert spec.experiment_id == "exp-001"

    # Out-of-bounds parameter: lr0 too high
    invalid_lr = dict(valid_proposal)
    invalid_lr["params"] = dict(valid_proposal["params"], lr0=0.5)
    with pytest.raises(AgentProposalViolation, match="Param 'lr0'=0.5 outside allowed range"):
        validate_agent_proposal(invalid_lr, policy)

    # Unwhitelisted trainer adapter
    invalid_adapter = dict(valid_proposal)
    invalid_adapter["trainer"] = dict(valid_proposal["trainer"], adapter_id="unauthorized_adapter")
    with pytest.raises(AgentProposalViolation, match="not in policy whitelist"):
        validate_agent_proposal(invalid_adapter, policy)


def test_experiment_runner_budget_limits(synthetic_dataset_dir: Path, tmp_path: Path):
    policy = ExperimentPolicy(
        max_runs=2,
        max_total_seconds=100.0,
        per_run_timeout_seconds=30.0,
    )
    runner = ExperimentRunner(policy=policy, work_dir=tmp_path / "exp_work")

    spec = ExperimentSpec(
        schema_version="1.0.0",
        experiment_id="exp-runner-01",
        dataset=DatasetRef(dataset_id="ds-test-001", manifest_sha256="0" * 64),
        trainer=TrainerSpec(
            adapter_id="yolo_detection_v1",
            model_id="mock_yolo_v1",
            checkpoint_sha256="0" * 64,
        ),
        params=ParamsSpec(imgsz=640, batch=8, epochs=1, seed=42, lr0=0.01, mosaic=1.0),
        reason="Verify budget runner",
    )

    # Run 1: succeeds
    res1 = runner.run_experiment(spec, synthetic_dataset_dir, run_id="run-1", mock_mode=True)
    assert res1.status == "succeeded"
    assert len(runner.completed_runs) == 1

    # Run 2: succeeds
    res2 = runner.run_experiment(spec, synthetic_dataset_dir, run_id="run-2", mock_mode=True)
    assert res2.status == "succeeded"
    assert len(runner.completed_runs) == 2

    # Run 3: exceeds max_runs (2) -> raises BudgetExceededError
    with pytest.raises(BudgetExceededError, match="Max runs budget exceeded: 2 >= 2"):
        runner.run_experiment(spec, synthetic_dataset_dir, run_id="run-3", mock_mode=True)


def test_experiment_runner_timeout(synthetic_dataset_dir: Path, tmp_path: Path):
    # Set per_run_timeout_seconds to 0.0 to trigger timeout
    policy = ExperimentPolicy(
        max_runs=5,
        max_total_seconds=100.0,
        per_run_timeout_seconds=0.0,
    )
    runner = ExperimentRunner(policy=policy, work_dir=tmp_path / "timeout_work")
    spec = ExperimentSpec(
        schema_version="1.0.0",
        experiment_id="exp-timeout-01",
        dataset=DatasetRef(dataset_id="ds-test-001", manifest_sha256="0" * 64),
        trainer=TrainerSpec(
            adapter_id="yolo_detection_v1",
            model_id="mock_yolo_v1",
            checkpoint_sha256="0" * 64,
        ),
        params=ParamsSpec(imgsz=640, batch=8, epochs=1, seed=42, lr0=0.01, mosaic=1.0),
        reason="Verify timeout handling",
    )

    res = runner.run_experiment(spec, synthetic_dataset_dir, run_id="run-timeout", mock_mode=True)
    assert res.status == "timeout"
    assert "Run exceeded timeout limit" in res.diagnostics[0]
