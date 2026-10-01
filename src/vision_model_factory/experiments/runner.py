"""Experiment runner managing budget execution, timeouts, and RunResult tracking."""

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from vision_model_factory.contracts.hashing import compute_sha256_file, compute_sha256_str
from vision_model_factory.contracts.models import (
    ClassMapItem,
    ExperimentSpec,
    FileRef,
    RunResult,
    TaskSpec,
)
from vision_model_factory.experiments.policy import ExperimentPolicy
from vision_model_factory.trainers.base import TrainerConfig
from vision_model_factory.trainers.yolo import YoloTrainerAdapter


class BudgetExceededError(Exception):
    """Raised when experiment execution attempts to exceed budget limits."""


class ExperimentRunner:
    """Orchestrates bounded training experiment runs under policy constraints."""

    def __init__(self, policy: ExperimentPolicy, work_dir: Path):
        self.policy = policy
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.completed_runs: List[RunResult] = []
        self.total_elapsed_seconds: float = 0.0

    def capture_environment(self) -> Dict[str, Any]:
        """Capture framework and execution environment versions."""
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available(),
        }

    def run_experiment(
        self,
        spec: ExperimentSpec,
        dataset_dir: Path,
        run_id: str,
        mock_mode: bool = False,
    ) -> RunResult:
        """
        Execute a single run governed by the experiment spec and budget policy.
        """
        # Check budget limits
        if len(self.completed_runs) >= self.policy.max_runs:
            raise BudgetExceededError(
                f"Max runs budget exceeded: {len(self.completed_runs)} >= {self.policy.max_runs}"
            )
        if self.total_elapsed_seconds >= self.policy.max_total_seconds:
            raise BudgetExceededError(
                f"Total time budget exceeded: {self.total_elapsed_seconds:.1f}s >= {self.policy.max_total_seconds}s"
            )

        run_output_dir = self.work_dir / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        effective_config = spec.model_dump()
        config_str = json.dumps(effective_config, sort_keys=True)
        config_sha256 = compute_sha256_str(config_str)

        # Build class_map from task.json
        task_path = dataset_dir / "task.json"
        with task_path.open("r", encoding="utf-8") as f:
            task = TaskSpec.model_validate(json.load(f))
        class_map = [
            ClassMapItem(index=i, class_id=cat.class_id)
            for i, cat in enumerate(task.categories)
        ]

        trainer_config = TrainerConfig(
            experiment_id=spec.experiment_id,
            run_id=run_id,
            dataset_dir=dataset_dir,
            output_dir=run_output_dir,
            params=spec.params,
            class_map=class_map,
            model_id=spec.trainer.model_id,
            checkpoint_sha256=spec.trainer.checkpoint_sha256,
            mock_mode=mock_mode,
        )

        adapter = YoloTrainerAdapter()
        t0 = time.perf_counter()

        try:
            res = adapter.train(trainer_config)
            duration = time.perf_counter() - t0

            # Check per-run timeout
            if duration > self.policy.per_run_timeout_seconds:
                run_result = RunResult(
                    schema_version="1.0.0",
                    run_id=run_id,
                    experiment_id=spec.experiment_id,
                    status="timeout",
                    dataset=spec.dataset,
                    effective_config_sha256=config_sha256,
                    environment=self.capture_environment(),
                    duration_seconds=round(duration, 3),
                    artifacts={},
                    diagnostics=[
                        f"Run exceeded timeout limit ({duration:.1f}s > {self.policy.per_run_timeout_seconds}s)"
                    ],
                )
            elif res.status == "succeeded" and res.checkpoint_path is not None:
                # Relative artifacts
                rel_artifacts = {
                    "checkpoint": FileRef(
                        path=str(res.checkpoint_path.relative_to(run_output_dir)),
                        sha256=res.checkpoint_sha256 or compute_sha256_file(res.checkpoint_path),
                    )
                }
                run_result = RunResult(
                    schema_version="1.0.0",
                    run_id=run_id,
                    experiment_id=spec.experiment_id,
                    status="succeeded",
                    dataset=spec.dataset,
                    effective_config_sha256=config_sha256,
                    environment=self.capture_environment(),
                    duration_seconds=round(duration, 3),
                    artifacts=rel_artifacts,
                    val_metrics=res.val_metrics,
                    diagnostics=res.diagnostics,
                )
            else:
                run_result = RunResult(
                    schema_version="1.0.0",
                    run_id=run_id,
                    experiment_id=spec.experiment_id,
                    status="failed",
                    dataset=spec.dataset,
                    effective_config_sha256=config_sha256,
                    environment=self.capture_environment(),
                    duration_seconds=round(duration, 3),
                    artifacts={},
                    diagnostics=res.diagnostics or ["Training run failed"],
                )
        except Exception as e:
            duration = time.perf_counter() - t0
            run_result = RunResult(
                schema_version="1.0.0",
                run_id=run_id,
                experiment_id=spec.experiment_id,
                status="failed",
                dataset=spec.dataset,
                effective_config_sha256=config_sha256,
                environment=self.capture_environment(),
                duration_seconds=round(duration, 3),
                artifacts={},
                diagnostics=[f"Unexpected exception during run: {str(e)}"],
            )

        self.total_elapsed_seconds += run_result.duration_seconds
        self.completed_runs.append(run_result)

        # Save run_result.json in run output dir
        res_file = run_output_dir / "run_result.json"
        with res_file.open("w", encoding="utf-8") as f:
            f.write(run_result.model_dump_json(indent=2))

        return run_result
