"""Tests for bounded experiment agent validation, budget enforcement, and timeouts."""

import json
import signal
import sys
import time
from pathlib import Path

import pytest

from vision_model_factory.contracts.hashing import compute_sha256_file
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
from vision_model_factory.experiments.policy import BudgetConfigError, ExperimentPolicy
from vision_model_factory.experiments.runner import BudgetExceededError, ExperimentRunner
from vision_model_factory.experiments.subprocess_util import Outcome, run_supervised
from vision_model_factory.trainers.registry import RegistryError, load_production_registry


def _spec(dataset_dir: Path, experiment_id: str = "exp-001") -> ExperimentSpec:
    return ExperimentSpec(
        schema_version="1.0.0",
        experiment_id=experiment_id,
        dataset=DatasetRef(
            dataset_id="ds-test-001", manifest_sha256=compute_sha256_file(dataset_dir / "dataset.json")
        ),
        trainer=TrainerSpec(
            adapter_id="yolo_detection_v1", model_id="mock_yolo_v1", checkpoint_sha256="0" * 64
        ),
        params=ParamsSpec(imgsz=640, batch=8, epochs=1, seed=42, lr0=0.01, mosaic=1.0),
        reason="Verify bounded execution",
    )


def _proposal() -> dict:
    return {
        "schema_version": "1.0.0",
        "experiment_id": "exp-001",
        "dataset": {"dataset_id": "ds-001", "manifest_sha256": "0" * 64},
        "trainer": {
            "adapter_id": "yolo_detection_v1",
            "model_id": "mock_yolo_v1",
            "checkpoint_sha256": "0" * 64,
        },
        "params": {"imgsz": 640, "batch": 16, "epochs": 10, "seed": 42, "lr0": 0.01, "mosaic": 1.0},
        "reason": "Test baseline configuration",
    }


def test_agent_proposal_validation():
    policy = ExperimentPolicy(
        imgsz_range=(320, 1280),
        batch_range=(1, 64),
        epochs_range=(1, 300),
        lr0_range=(1e-5, 0.1),
        mosaic_range=(0.0, 1.0),
    )
    registry = load_production_registry()

    spec = validate_agent_proposal(_proposal(), policy, registry)
    assert spec.experiment_id == "exp-001"

    # Out-of-bounds parameter: lr0 too high
    invalid_lr = _proposal()
    invalid_lr["params"]["lr0"] = 0.5
    with pytest.raises(AgentProposalViolation, match="param 'lr0'=0.5 outside allowed range"):
        validate_agent_proposal(invalid_lr, policy, registry)

    # Adapters outside the registry whitelist are refused by the registry, which is also
    # what the runner enforces at execution time.
    invalid_adapter = _proposal()
    invalid_adapter["trainer"]["adapter_id"] = "unauthorized_adapter"
    with pytest.raises(AgentProposalViolation, match="rejected by registry"):
        validate_agent_proposal(invalid_adapter, policy, registry)

    # A model the registry does not know about is refused even when the adapter is fine.
    unknown_model = _proposal()
    unknown_model["trainer"]["model_id"] = "some_random_net.pt"
    with pytest.raises(AgentProposalViolation, match="rejected by registry"):
        validate_agent_proposal(unknown_model, policy, registry)

    # An unpinned checkpoint digest for a real model is refused: a whitelist that admits
    # anything shaped like a hash is not a whitelist.
    unpinned = _proposal()
    unpinned["trainer"]["model_id"] = "yolo26n.pt"
    unpinned["trainer"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(AgentProposalViolation, match="not pinned"):
        validate_agent_proposal(unpinned, policy, registry)


def test_agent_proposal_refuses_a_model_the_registry_does_not_pin():
    """Registry pinning is enforced through the agent path, not just at execution."""
    registry = load_production_registry()
    policy = ExperimentPolicy()
    proposal = _proposal()
    proposal["trainer"] = {
        "adapter_id": "yolo_detection_v1",
        "model_id": "yolov8n.pt",
        "checkpoint_sha256": "0" * 64,  # whitelisted model id, digest nobody pinned
    }
    with pytest.raises(AgentProposalViolation, match="not pinned"):
        validate_agent_proposal(proposal, policy, registry)

    # The same spec, with the digest the registry actually pins, is accepted.
    proposal["trainer"]["checkpoint_sha256"] = registry.pinned_checkpoints("yolov8n.pt")[0]
    assert validate_agent_proposal(proposal, policy, registry).trainer.model_id == "yolov8n.pt"


def test_experiment_runner_requires_a_registry(tmp_path: Path):
    with pytest.raises(TypeError):
        ExperimentRunner(policy=ExperimentPolicy(max_runs=1, max_total_seconds=10, per_run_timeout_seconds=5),
                         work_dir=tmp_path)


def test_experiment_runner_refuses_a_dataset_the_spec_does_not_pin(
    synthetic_dataset_dir: Path, tmp_path: Path
):
    """Training against bytes other than the run record names would break traceability."""
    runner = ExperimentRunner(
        policy=ExperimentPolicy(max_runs=2, max_total_seconds=120.0, per_run_timeout_seconds=60.0),
        work_dir=tmp_path / "exp_work",
        registry=load_production_registry(),
    )
    spec = _spec(synthetic_dataset_dir)
    tampered = spec.model_copy(
        update={"dataset": DatasetRef(dataset_id="ds-test-001", manifest_sha256="b" * 64)}
    )

    with pytest.raises(ValueError, match="Dataset package digest mismatch"):
        runner.run_experiment(tampered, synthetic_dataset_dir, run_id="run-tampered", mock_mode=True)


def test_experiment_runner_budget_limits(synthetic_dataset_dir: Path, tmp_path: Path):
    policy = ExperimentPolicy(
        max_runs=2,
        max_total_seconds=600.0,
        per_run_timeout_seconds=120.0,
    )
    runner = ExperimentRunner(
        policy=policy, work_dir=tmp_path / "exp_work", registry=load_production_registry()
    )
    spec = _spec(synthetic_dataset_dir)

    res1 = runner.run_experiment(spec, synthetic_dataset_dir, run_id="run-1", mock_mode=True)
    assert res1.status == "succeeded"
    assert res1.val_metrics is not None
    assert "checkpoint" in res1.artifacts
    assert len(runner.completed_runs) == 1

    res2 = runner.run_experiment(spec, synthetic_dataset_dir, run_id="run-2", mock_mode=True)
    assert res2.status == "succeeded"
    assert len(runner.completed_runs) == 2

    # The third run is refused before anything is spawned, so the budget is a ceiling
    # rather than a number that overshoots.
    assert runner.budget_exhausted is True
    with pytest.raises(BudgetExceededError, match="max runs budget exceeded: 2 >= 2"):
        runner.run_experiment(spec, synthetic_dataset_dir, run_id="run-3", mock_mode=True)
    assert len(runner.completed_runs) == 2
    assert not (tmp_path / "exp_work" / "run-3").exists()


def test_experiment_runner_persists_run_result(synthetic_dataset_dir: Path, tmp_path: Path):
    runner = ExperimentRunner(
        policy=ExperimentPolicy(max_runs=1, max_total_seconds=120.0, per_run_timeout_seconds=60.0),
        work_dir=tmp_path / "persist_work",
        registry=load_production_registry(),
    )
    result = runner.run_experiment(
        _spec(synthetic_dataset_dir), synthetic_dataset_dir, run_id="run-persist", mock_mode=True
    )

    record = tmp_path / "persist_work" / "run-persist" / "run_result.json"
    assert record.is_file()
    on_disk = json.loads(record.read_text(encoding="utf-8"))
    assert on_disk["status"] == "succeeded"
    assert on_disk["effective_config_sha256"] == result.effective_config_sha256
    # The recorded checkpoint reference must describe the bytes that actually exist.
    ckpt = tmp_path / "persist_work" / "run-persist" / on_disk["artifacts"]["checkpoint"]["path"]
    assert ckpt.is_file()
    assert compute_sha256_file(ckpt) == on_disk["artifacts"]["checkpoint"]["sha256"]


def test_run_supervised_actually_terminates_a_long_child(tmp_path: Path):
    """The budget must stop work, not merely label it afterwards.

    A blocking trainer that runs 30s under a 1s budget has to die inside ~1s; anything
    else means the timeout only renames the outcome after the compute was already spent.
    """
    victim = tmp_path / "slow_child.py"
    victim.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    start = time.monotonic()
    run = run_supervised(
        [sys.executable, "-u", str(victim)],
        timeout_seconds=1.0,
        log_path=tmp_path / "child.log",
    )
    elapsed = time.monotonic() - start

    assert run.outcome is Outcome.TIMED_OUT
    assert elapsed < 10.0, f"timeout did not stop the child (waited {elapsed:.1f}s)"
    # Terminated by a signal, and its log stayed empty: the child never got as far as
    # producing output, let alone finishing its sleep.
    assert run.exit_code in (124, -signal.SIGTERM, -signal.SIGKILL)
    assert run.log_path.read_text(encoding="utf-8") == ""


def test_experiment_runner_timeout(synthetic_dataset_dir: Path, tmp_path: Path):
    """A run that cannot finish inside its budget is a timeout with no artifacts."""
    runner = ExperimentRunner(
        # Positive but tiny: the child cannot even finish importing the package in 0.2s,
        # so the supervisor has to terminate it rather than read an outcome.
        policy=ExperimentPolicy(max_runs=5, max_total_seconds=120.0, per_run_timeout_seconds=0.2),
        work_dir=tmp_path / "timeout_work",
        registry=load_production_registry(),
    )

    res = runner.run_experiment(
        _spec(synthetic_dataset_dir), synthetic_dataset_dir, run_id="run-timeout", mock_mode=True
    )

    assert res.status == "timeout"
    assert any("timeout" in d.lower() for d in res.diagnostics)
    assert res.val_metrics is None
    assert res.artifacts == {}
    # Nothing was fabricated: no outcome record and no weights were left behind.
    assert not (tmp_path / "timeout_work" / "run-timeout" / "attempt" / "attempt_result.json").exists()
    assert list((tmp_path / "timeout_work" / "run-timeout").glob("**/best.pt")) == []
    # The budget still consumed wall clock, so the loop cannot pretend the run was free.
    assert runner.total_elapsed_seconds > 0.0


def test_policy_rejects_unusable_budgets():
    """Budgets that could never be observed are refused at construction."""
    with pytest.raises(BudgetConfigError):
        ExperimentPolicy(per_run_timeout_seconds=0.0)
    with pytest.raises(BudgetConfigError):
        ExperimentPolicy(max_runs=0)
    with pytest.raises(BudgetConfigError):
        ExperimentPolicy(max_total_seconds=10.0, per_run_timeout_seconds=60.0)


def test_registry_rejects_unpinned_models_without_a_list():
    """An unknown model id is refused rather than admitted on a hash-shaped string."""
    registry = load_production_registry()
    assert registry.is_checkpoint_allowed("typo_model.pt", "a" * 64) is False
    with pytest.raises(RegistryError):
        registry.resolve_adapter(
            TrainerSpec(adapter_id="yolo_detection_v1", model_id="typo_model.pt", checkpoint_sha256="a" * 64)
        )

