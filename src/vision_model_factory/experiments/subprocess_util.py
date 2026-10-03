"""Subprocess helpers for enforcing compute budgets rather than merely recording them.

A per-run timeout that is checked after a blocking call returns cannot stop anything:
the work already happened. Long-running training therefore executes in a child process
that the supervisor can terminate. The exit code distinguishes "the run failed" from
"the supervisor ended it".
"""

import enum
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Exit codes meaning "the supervisor ended this run" rather than "the run failed".
# 124 is the conventional timeout exit code; negative values are POSIX signal deaths.
TIMEOUT_EXIT_CODES = (124, -int(signal.SIGTERM), -int(signal.SIGKILL))


class Outcome(enum.Enum):
    """Result classification for a supervised run."""

    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SupervisedRun:
    """Outcome of one supervised child-process execution."""

    outcome: Outcome
    duration_seconds: float
    exit_code: Optional[int]
    log_path: Path


def child_command(module: str, argv: list) -> Tuple[list, dict]:
    """Build the command and environment used to supervise a module entry point.

    `sys.executable -u -m <module>` with `PYTHONPATH` pointed at this package's `src`
    directory, so the child resolves `vision_model_factory` exactly like the parent.
    """
    src_dir = str(Path(__file__).resolve().parent.parent.parent)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else src_dir
    env["PYTHONUNBUFFERED"] = "1"
    return [sys.executable, "-u", "-m", module, *argv], env


def run_supervised(
    argv: list,
    timeout_seconds: float,
    log_path: Path,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
) -> SupervisedRun:
    """Run a child process, terminating it if it exceeds `timeout_seconds`.

    Termination escalates SIGTERM -> SIGKILL so a wedged trainer cannot outlive its
    budget. The child's combined stdout/stderr is streamed to `log_path` instead of a
    pipe, which avoids the deadlock a full pipe buffer would cause during long runs.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        timed_out = False
        cancelled = False
        try:
            exit_code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(proc)
            exit_code = proc.wait()
        except KeyboardInterrupt:
            cancelled = True
            _terminate(proc)
            exit_code = proc.wait()
    duration = time.monotonic() - start

    outcome = _classify(timed_out=timed_out, cancelled=cancelled, exit_code=exit_code)
    return SupervisedRun(outcome=outcome, duration_seconds=duration, exit_code=exit_code, log_path=log_path)


def _classify(timed_out: bool, cancelled: bool, exit_code: int) -> Outcome:
    if cancelled:
        return Outcome.CANCELLED
    if timed_out or exit_code in TIMEOUT_EXIT_CODES:
        return Outcome.TIMED_OUT
    return Outcome.COMPLETED if exit_code == 0 else Outcome.FAILED


def _terminate(proc: "subprocess.Popen") -> None:
    """Escalate from a polite stop to a forced kill."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
    except OSError:  # process already gone
        pass


__all__ = ["TIMEOUT_EXIT_CODES", "Outcome", "SupervisedRun", "child_command", "run_supervised"]
