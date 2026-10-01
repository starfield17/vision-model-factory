"""Experiment execution policy and budget governance."""

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class ExperimentPolicy:
    """Budget and constraint policy governing experiment iterations."""

    max_runs: int = 5
    max_total_seconds: float = 3600.0
    per_run_timeout_seconds: float = 600.0

    # Whitelisted models per adapter
    allowed_adapters: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "yolo_detection_v1": {
                "yolov8n.pt",
                "yolov8s.pt",
                "yolov8m.pt",
                "mock_yolo_v1",
            }
        }
    )

    # Allowed parameter ranges
    imgsz_range: Tuple[int, int] = (320, 1280)
    batch_range: Tuple[int, int] = (1, 64)
    epochs_range: Tuple[int, int] = (1, 300)
    lr0_range: Tuple[float, float] = (1e-5, 0.1)
    mosaic_range: Tuple[float, float] = (0.0, 1.0)

    # Seed policy
    fixed_seed: int = 42

    # Stopping rules
    target_val_map50: float = 0.85
    min_val_map50: float = 0.50


DEFAULT_EXPERIMENT_POLICY = ExperimentPolicy()
