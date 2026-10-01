"""Training adapters and registry module."""

from vision_model_factory.trainers.base import (
    BaseTrainerAdapter,
    TrainerConfig,
    TrainerRunResult,
)
from vision_model_factory.trainers.registry import DEFAULT_TRAINER_REGISTRY, TrainerRegistry
from vision_model_factory.trainers.yolo import TinyYoloMockNet, YoloTrainerAdapter

__all__ = [
    "BaseTrainerAdapter",
    "DEFAULT_TRAINER_REGISTRY",
    "TinyYoloMockNet",
    "TrainerConfig",
    "TrainerRegistry",
    "TrainerRunResult",
    "YoloTrainerAdapter",
]
