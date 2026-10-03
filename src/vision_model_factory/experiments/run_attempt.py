"""Child-process entry point for one supervised training run.

The runner executes each attempt in a separate process so the per-run budget can actually
terminate it. This module is invoked as `python -m vision_model_factory.experiments.run_attempt`
by `ExperimentRunner`; it is not part of the public CLI surface.

The handoff is deliberately narrow: the child receives a JSON request, writes exactly one
`attempt_result.json` describing what happened, and exits. Only the attempt outcome file is
read back by the parent - the parent never trusts the child's exit code to decide success,
and refuses to invent an outcome when the file is missing.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import ClassMapItem, ExperimentSpec, TaskSpec
from vision_model_factory.trainers.base import TrainerConfig
from vision_model_factory.trainers.registry import TrainerRegistry

ATTEMPT_RESULT_FILENAME = "attempt_result.json"
MODULE_NAME = "vision_model_factory.experiments.run_attempt"


def write_attempt_result(attempt_dir: Path, payload: Dict[str, Any]) -> Path:
    """Persist the attempt outcome for the parent process to read back."""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    out = attempt_dir / ATTEMPT_RESULT_FILENAME
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_attempt_result(attempt_dir: Path) -> Dict[str, Any]:
    """Read the child's outcome, raising when the child produced none."""
    path = attempt_dir / ATTEMPT_RESULT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Supervised training attempt wrote no {ATTEMPT_RESULT_FILENAME} in {attempt_dir}; "
            "the outcome of this attempt is unknown and must not be guessed."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_request(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_digest(path: Path, expected: str, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = compute_sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} digest mismatch: expected {expected}, got {actual}")
    return path


def execute_attempt(request_path: Path) -> int:
    """Run one training attempt described by a request file. Returns a process exit code."""
    request = _load_request(request_path)
    attempt_dir = Path(request["attempt_dir"]).resolve()

    spec_path = _validate_digest(
        Path(request["spec_path"]).resolve(), request["spec_sha256"], "ExperimentSpec file"
    )
    spec = ExperimentSpec.model_validate(json.loads(spec_path.read_text(encoding="utf-8")))

    # The parent resolved the adapter and the checkpoint file; the child re-verifies the
    # bytes it is about to execute so a swapped file cannot slip through the handoff.
    registry = TrainerRegistry()
    adapter = registry.resolve_adapter(spec.trainer)
    raw_checkpoint = request.get("checkpoint_path")
    checkpoint_path = Path(raw_checkpoint).resolve() if raw_checkpoint else None
    if not registry.is_mock_model(spec.trainer.model_id):
        if checkpoint_path is None:
            raise ValueError(
                f"Model '{spec.trainer.model_id}' requires a resolved checkpoint file; none was supplied."
            )
        actual = compute_sha256_file(checkpoint_path)
        if actual != spec.trainer.checkpoint_sha256:
            raise ValueError(
                f"Checkpoint digest changed between scheduling and execution: expected "
                f"{spec.trainer.checkpoint_sha256}, got {actual}"
            )

    task_path = Path(request["dataset_dir"]).resolve() / "task.json"
    if not task_path.is_file():
        raise FileNotFoundError(f"task.json not found in dataset package: {task_path}")
    task = TaskSpec.model_validate(json.loads(task_path.read_text(encoding="utf-8")))

    output_dir = attempt_dir / "train"
    config = TrainerConfig(
        experiment_id=spec.experiment_id,
        run_id=request["run_id"],
        dataset_dir=Path(request["dataset_dir"]).resolve(),
        output_dir=output_dir,
        params=spec.params,
        class_map=_class_map(task),
        model_id=spec.trainer.model_id,
        checkpoint_sha256=spec.trainer.checkpoint_sha256,
        mock_mode=bool(request.get("mock_mode", False)),
        checkpoint_path=checkpoint_path,
        per_run_timeout_seconds=request.get("per_run_timeout_seconds"),
    )

    start = time.monotonic()
    result = adapter.train(config)
    duration = time.monotonic() - start

    payload: Dict[str, Any] = {
        "status": result.status,
        "duration_seconds": round(duration, 3),
        "diagnostics": list(result.diagnostics),
        "val_metrics": dict(result.val_metrics),
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
        "checkpoint_sha256": result.checkpoint_sha256,
        "attempt_dir": str(attempt_dir),
    }
    write_attempt_result(attempt_dir, payload)
    return 0 if result.status == "succeeded" else 1


def _class_map(task: TaskSpec) -> list:
    """Bind decoder indices to the upstream TaskSpec category order.

    Indices come from the task's declared category order, never from a class-name guess.
    """
    return [ClassMapItem(index=i, class_id=c.class_id) for i, c in enumerate(task.categories)]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME, description="Execute one supervised training attempt."
    )
    parser.add_argument("request", type=Path, help="Path to the attempt request JSON file.")
    args = parser.parse_args(argv)

    attempt_dir = _attempt_dir_from_request(args.request)
    try:
        exit_code = execute_attempt(args.request)
    except Exception as exc:  # noqa: BLE001 - the parent reads the outcome file, not a traceback
        write_attempt_result(
            attempt_dir,
            {
                "status": "failed",
                "duration_seconds": 0.0,
                "diagnostics": [f"{type(exc).__name__}: {exc}"],
                "val_metrics": {},
                "checkpoint_path": None,
                "checkpoint_sha256": None,
                "attempt_dir": str(attempt_dir),
            },
        )
        print(f"training attempt failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(exit_code)


def _attempt_dir_from_request(request_path: Path) -> Path:
    """Best-effort attempt directory, so a failure still leaves an outcome record."""
    try:
        return Path(_load_request(request_path)["attempt_dir"]).resolve()
    except Exception:
        return request_path.resolve().parent


if __name__ == "__main__":
    main()
