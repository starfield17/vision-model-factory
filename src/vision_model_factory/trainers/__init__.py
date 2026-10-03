"""Training adapters, registry, and adapter resolution."""

from vision_model_factory.trainers.base import (
    BaseTrainerAdapter,
    TrainerConfig,
    TrainerRunResult,
)
from vision_model_factory.trainers.registry import (
    RegistryError,
    TrainerRegistry,
    load_production_registry,
)
from vision_model_factory.trainers.yolo import TinyYoloMockNet, YoloTrainerAdapter

__all__ = [
    "BaseTrainerAdapter",
    "RegistryError",
    "TinyYoloMockNet",
    "TrainerConfig",
    "TrainerRegistry",
    "TrainerRunResult",
    "YoloTrainerAdapter",
    "load_production_registry",
]
