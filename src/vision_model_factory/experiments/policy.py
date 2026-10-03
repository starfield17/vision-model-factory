"""Experiment execution policy: budgets, parameter bounds and seed discipline.

The policy is the governor of the Agent loop. It carries no model or adapter whitelist:
`trainers.registry` is the single source of truth for what may be executed, and the two
lists were previously duplicated here with no guarantee they stayed in step.

There is no usable default policy object. A run started without an explicitly supplied
policy cannot claim a budget, so callers must construct one (the CLI builds it from CLI
flags).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from vision_model_factory.contracts.models import ParamsSpec


class BudgetConfigError(ValueError):
    """Raised when a policy is internally inconsistent or unusable."""


@dataclass
class ExperimentPolicy:
    """Budget and constraint policy governing experiment iterations."""

    max_runs: int = 8
    max_total_seconds: float = 3600.0
    per_run_timeout_seconds: float = 600.0

    # Allowed parameter ranges. Bounds are inclusive.
    imgsz_range: Tuple[int, int] = (320, 1280)
    batch_range: Tuple[int, int] = (1, 64)
    epochs_range: Tuple[int, int] = (1, 300)
    lr0_range: Tuple[float, float] = (1e-5, 0.1)
    mosaic_range: Tuple[float, float] = (0.0, 1.0)

    # Seed policy: when set, every experiment must use exactly this seed, so runs stay
    # comparable. When None, the experiment chooses but the value stays pinned in the
    # run's effective config digest.
    fixed_seed: Optional[int] = 42

    # Stopping rules evaluated by the Agent loop against val metrics.
    target_val_map50: float = 0.85
    min_val_map50: float = 0.50

    def __post_init__(self) -> None:
        if self.max_runs < 1:
            raise BudgetConfigError(f"max_runs must be >= 1, got {self.max_runs}")
        if self.max_total_seconds <= 0:
            raise BudgetConfigError(f"max_total_seconds must be > 0, got {self.max_total_seconds}")
        if self.per_run_timeout_seconds <= 0:
            raise BudgetConfigError(
                f"per_run_timeout_seconds must be > 0, got {self.per_run_timeout_seconds}"
            )
        if self.per_run_timeout_seconds > self.max_total_seconds:
            raise BudgetConfigError(
                f"per_run_timeout_seconds ({self.per_run_timeout_seconds}) exceeds max_total_seconds "
                f"({self.max_total_seconds}); the total budget could never be observed"
            )
        for name, bounds in self._range_fields().items():
            low, high = bounds
            if low > high:
                raise BudgetConfigError(f"{name}_range lower bound {low} exceeds upper bound {high}")

    def _range_fields(self) -> Dict[str, Tuple]:
        return {
            "imgsz": self.imgsz_range,
            "batch": self.batch_range,
            "epochs": self.epochs_range,
            "lr0": self.lr0_range,
            "mosaic": self.mosaic_range,
        }

    def param_violations(self, params: ParamsSpec) -> List[str]:
        """Return human-readable violations for out-of-bounds hyperparameters."""
        violations: List[str] = []
        for name, (low, high) in self._range_fields().items():
            value = getattr(params, name)
            if not (low <= value <= high):
                violations.append(f"param '{name}'={value} outside allowed range [{low}, {high}]")
        if self.fixed_seed is not None and params.seed != self.fixed_seed:
            violations.append(
                f"seed={params.seed} violates the policy fixed seed {self.fixed_seed}"
            )
        return violations

    def allowed_model_ids_for(self, adapter_id: str, registry) -> Set[str]:
        """Delegate to the trainer registry; kept for call-site readability."""
        if not registry.is_adapter_allowed(adapter_id):
            return set()
        return set(registry.allowed_models(adapter_id))

    def describe_budget(self) -> Dict[str, object]:
        return {
            "max_runs": self.max_runs,
            "max_total_seconds": self.max_total_seconds,
            "per_run_timeout_seconds": self.per_run_timeout_seconds,
            "fixed_seed": self.fixed_seed,
            "param_ranges": {k: list(v) for k, v in self._range_fields().items()},
        }


__all__ = ["BudgetConfigError", "ExperimentPolicy"]
