"""Experiment runner: budget enforcement, per-run timeouts, and RunResult records.

Two properties make the budget real rather than decorative:

* Each attempt executes in a supervised child process, so `per_run_timeout_seconds`
  terminates work instead of only labelling it afterwards.
* Budget accounting accumulates across runs in the runner and is checked before an
  attempt starts, so the loop stops at the ceiling rather than overshooting it.

Failures keep their own identity. A run that timed out, failed, or produced no outcome
record is recorded as such with a non-empty diagnosis; nothing here manufactures metrics
for a run that did not produce them.
"""

import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from vision_model_factory.contracts.hashing import compute_sha256_file, compute_sha256_str
from vision_model_factory.contracts.models import (
    ExperimentSpec,
    FileRef,
    RunResult,
    model_json,
)
from vision_model_factory.experiments.policy import ExperimentPolicy
from vision_model_factory.experiments.run_attempt import (
    MODULE_NAME as ATTEMPT_MODULE,
)
from vision_model_factory.experiments.run_attempt import read_attempt_result
from vision_model_factory.experiments.subprocess_util import Outcome, child_command, run_supervised
from vision_model_factory.trainers.registry import TrainerRegistry

RUN_RESULT_FILENAME = "run_result.json"


class BudgetExceededError(RuntimeError):
    """Raised when experiment execution attempts to exceed budget limits."""


class AttemptOutcomeMissingError(RuntimeError):
    """Raised when a supervised attempt produced no outcome record to read back."""


class ExperimentRunner:
    """Orchestrates bounded training experiment runs under policy constraints."""

    def __init__(self, policy: ExperimentPolicy, work_dir: Path, registry: TrainerRegistry):
        if policy is None:
            raise BudgetExceededError(
                "An experiment loop requires an explicit policy; running without a budget is refused."
            )
        self.policy = policy
        self.registry = registry
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.completed_runs: List[RunResult] = []
        self.total_elapsed_seconds: float = 0.0

    @property
    def budget_exhausted(self) -> bool:
        """Whether another run may be attempted under the remaining budget."""
        return self._budget_block_reason() is not None

    def _budget_block_reason(self) -> Optional[str]:
        if len(self.completed_runs) >= self.policy.max_runs:
            return f"max runs budget exceeded: {len(self.completed_runs)} >= {self.policy.max_runs}"
        if self.total_elapsed_seconds >= self.policy.max_total_seconds:
            return (
                f"total time budget exceeded: {self.total_elapsed_seconds:.1f}s >= "
                f"{self.policy.max_total_seconds}s"
            )
        return None

    def remaining_runs(self) -> int:
        return max(0, self.policy.max_runs - len(self.completed_runs))

    def capture_environment(self) -> Dict[str, Any]:
        """Capture framework and execution environment versions."""
        mps = getattr(torch.backends, "mps", None)
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": mps is not None and mps.is_available(),
        }

    def run_experiment(
        self,
        spec: ExperimentSpec,
        dataset_dir: Path,
        run_id: str,
        mock_mode: bool = False,
        checkpoint_path: Optional[Path] = None,
    ) -> RunResult:
        """Execute one run under the policy and persist its RunResult."""
        block = self._budget_block_reason()
        if block:
            raise BudgetExceededError(block)

        # Resolve through the registry so adapter id, model id and the pinned checkpoint
        # digest are all validated before any process is spawned.
        self.registry.resolve_adapter(spec.trainer)
        resolved_checkpoint = self._resolve_checkpoint(spec, mock_mode, checkpoint_path)

        dataset_dir = Path(dataset_dir).resolve()
        self._validate_dataset_digest(spec, dataset_dir)

        run_dir = self.work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = run_dir / "attempt"

        config_sha256 = compute_sha256_str(json.dumps(spec.model_dump(mode="json"), sort_keys=True))
        request_path = self._write_request(
            spec=spec,
            run_id=run_id,
            dataset_dir=dataset_dir,
            attempt_dir=attempt_dir,
            mock_mode=mock_mode,
            checkpoint_path=resolved_checkpoint,
        )

        argv, env = child_command(ATTEMPT_MODULE, [str(request_path)])
        supervised = run_supervised(
            argv,
            timeout_seconds=self.policy.per_run_timeout_seconds,
            log_path=run_dir / "attempt.log",
            env=env,
            cwd=self.work_dir.parent,
        )

        outcome, val_metrics, diagnostics, artifacts = self._collect_outcome(
            supervised=supervised,
            attempt_dir=attempt_dir,
            run_dir=run_dir,
        )

        self.total_elapsed_seconds += supervised.duration_seconds

        run_result = RunResult(
            schema_version="1.0.0",
            run_id=run_id,
            experiment_id=spec.experiment_id,
            status=outcome,
            dataset=spec.dataset,
            effective_config_sha256=config_sha256,
            environment=self.capture_environment(),
            duration_seconds=round(supervised.duration_seconds, 3),
            artifacts=artifacts,
            diagnostics=diagnostics,
            val_metrics=val_metrics,
        )

        (run_dir / RUN_RESULT_FILENAME).write_text(model_json(run_result, indent=2), encoding="utf-8")
        self.completed_runs.append(run_result)
        return run_result

    def _write_request(
        self,
        spec: ExperimentSpec,
        run_id: str,
        dataset_dir: Path,
        attempt_dir: Path,
        mock_mode: bool,
        checkpoint_path: Optional[Path],
    ) -> Path:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        spec_path = self.work_dir / f"{run_id}.spec.json"
        spec_path.write_text(model_json(spec, indent=2), encoding="utf-8")
        request = {
            "run_id": run_id,
            "spec_path": str(spec_path),
            "spec_sha256": compute_sha256_file(spec_path),
            "dataset_dir": str(dataset_dir),
            "attempt_dir": str(attempt_dir),
            "mock_mode": mock_mode,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "per_run_timeout_seconds": self.policy.per_run_timeout_seconds,
        }
        request_path = attempt_dir / "request.json"
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
        return request_path

    def _resolve_checkpoint(
        self, spec: ExperimentSpec, mock_mode: bool, checkpoint_path: Optional[Path]
    ) -> Optional[Path]:
        """Locate the base checkpoint file to execute against, digest-verified.

        Mock runs synthesise their own weights and need no file. A real run must be given
        a local file: silently downloading a bare model name would execute weights the
        registry's digest pin never approved.
        """
        if mock_mode or self.registry.is_mock_model(spec.trainer.model_id):
            return checkpoint_path
        candidates: List[Path] = []
        if checkpoint_path is not None:
            candidates.append(Path(checkpoint_path))
        candidates.extend([Path.cwd() / spec.trainer.model_id, self.work_dir / spec.trainer.model_id])
        for candidate in candidates:
            if candidate.is_file():
                self.registry.verify_checkpoint_file(spec.trainer, candidate)
                return candidate
        from vision_model_factory.trainers.registry import RegistryError

        raise RegistryError(
            f"No local base checkpoint found for '{spec.trainer.model_id}' (looked in: "
            f"{', '.join(str(c) for c in candidates)}). Pass --checkpoint with a local file; "
            "downloading an unpinned copy is refused."
        )

    def _validate_dataset_digest(self, spec: ExperimentSpec, dataset_dir: Path) -> None:
        """Refuse to train against a dataset package that is not the one the spec names."""
        manifest_path = dataset_dir / "dataset.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"dataset.json not found in dataset package: {dataset_dir}")
        actual = compute_sha256_file(manifest_path)
        if actual != spec.dataset.manifest_sha256:
            raise ValueError(
                f"Dataset package digest mismatch for '{spec.dataset.dataset_id}': the spec pins "
                f"{spec.dataset.manifest_sha256} but {manifest_path} hashes to {actual}. Training "
                "against different bytes than the run record claims would break traceability."
            )

    def _collect_outcome(
        self,
        supervised,
        attempt_dir: Path,
        run_dir: Path,
    ) -> Tuple[str, Optional[Dict[str, Any]], List[str], Dict[str, FileRef]]:
        """Map a supervised attempt onto a RunResult status, metrics and artifacts.

        The child's outcome file is authoritative for success; the exit code alone never
        upgrades a run, and a missing file is reported as an unknown outcome rather than
        guessed at.
        """
        diagnostics: List[str] = []
        try:
            attempt = read_attempt_result(attempt_dir)
        except FileNotFoundError as exc:
            if supervised.outcome is Outcome.TIMED_OUT:
                diagnostics.append(
                    f"Run exceeded the per-run timeout of {self.policy.per_run_timeout_seconds}s "
                    "and was terminated before it could record an outcome."
                )
                return "timeout", None, diagnostics, {}
            raise AttemptOutcomeMissingError(str(exc)) from exc

        raw_diagnostics = attempt.get("diagnostics") or []
        diagnostics.extend(str(d) for d in raw_diagnostics)
        child_status = attempt.get("status")
        child_metrics = attempt.get("val_metrics") or {}
        duration = supervised.duration_seconds

        if supervised.outcome is Outcome.CANCELLED:
            diagnostics.append("Run was cancelled by the operator.")
            return "cancelled", None, diagnostics or ["Run cancelled"], {}

        if supervised.outcome is Outcome.TIMED_OUT:
            diagnostics.append(
                f"Run exceeded the per-run timeout of {self.policy.per_run_timeout_seconds}s "
                "and was terminated."
            )
            return "timeout", None, diagnostics, {}

        if child_status != "succeeded":
            diagnostics.append("Training attempt did not succeed.")
            return "failed", None, diagnostics or ["Training run failed"], {}

        # A completed child reporting success must still fit inside the budget: the budget
        # is a wall-clock ceiling, so an overrunning-but-finished attempt is a timeout.
        if duration > self.policy.per_run_timeout_seconds:
            diagnostics.append(
                f"Attempt completed in {duration:.1f}s, exceeding the per-run timeout of "
                f"{self.policy.per_run_timeout_seconds}s."
            )
            return "timeout", None, diagnostics, {}

        checkpoint_value = attempt.get("checkpoint_path")
        if not checkpoint_value:
            diagnostics.append("Attempt reported success without a checkpoint artifact.")
            return "failed", None, diagnostics, {}

        checkpoint_file = Path(checkpoint_value)
        if not checkpoint_file.is_file():
            diagnostics.append(f"Checkpoint artifact missing on disk: {checkpoint_file}")
            return "failed", None, diagnostics, {}

        recorded_digest = attempt.get("checkpoint_sha256")
        actual_digest = compute_sha256_file(checkpoint_file)
        if recorded_digest != actual_digest:
            diagnostics.append(
                f"Checkpoint digest mismatch: the attempt recorded {recorded_digest} but the file "
                f"on disk hashes to {actual_digest}."
            )
            return "failed", None, diagnostics, {}

        try:
            relative = checkpoint_file.relative_to(run_dir)
        except ValueError as exc:
            raise AttemptOutcomeMissingError(
                f"Checkpoint {checkpoint_file} is outside the run directory {run_dir}; "
                "package-relative artifact references cannot be produced."
            ) from exc

        artifacts = {"checkpoint": FileRef(path=str(relative), sha256=actual_digest)}
        if not child_metrics:
            diagnostics.append("Attempt succeeded but reported no validation metrics.")
        return "succeeded", child_metrics or None, diagnostics, artifacts


__all__ = [
    "AttemptOutcomeMissingError",
    "BudgetExceededError",
    "ExperimentRunner",
    "RUN_RESULT_FILENAME",
]
