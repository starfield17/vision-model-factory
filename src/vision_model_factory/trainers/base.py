"""Base protocol and configuration types for trainer adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vision_model_factory.contracts.models import ClassMapItem, ParamsSpec, TaskSpec


@dataclass
class TrainerConfig:
    """Configuration for a specific training run."""

    experiment_id: str
    run_id: str
    dataset_dir: Path
    output_dir: Path
    params: ParamsSpec
    class_map: List[ClassMapItem]
    model_id: str
    checkpoint_sha256: str
    mock_mode: bool = False


@dataclass
class TrainerRunResult:
    """Outcome produced by a trainer adapter."""

    status: str
    duration_seconds: float
    checkpoint_path: Optional[Path] = None
    checkpoint_sha256: Optional[str] = None
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)


class BaseTrainerAdapter(ABC):
    """Abstract base class for model trainer adapters."""

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Unique adapter identifier in registry."""
        pass

    @abstractmethod
    def prepare_dataset(
        self,
        dataset_dir: Path,
        work_dir: Path,
        task: TaskSpec,
        class_map: List[ClassMapItem],
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Convert an immutable dataset package into trainer-specific layout.
        Returns:
            (config_path, stats_dict)
        """
        pass

    @abstractmethod
    def train(self, config: TrainerConfig) -> TrainerRunResult:
        """Execute a training run according to config."""
        pass
