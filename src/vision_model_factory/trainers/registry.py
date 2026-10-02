"""Trainer adapter registry and parameter whitelist enforcement."""

from typing import Dict, Optional, Set


class TrainerRegistry:
    """Registry maintaining whitelisted trainer adapters, model architectures, and checkpoints."""

    def __init__(self) -> None:
        # Whitelisted models per adapter_id -> set of allowed model_ids
        self._allowed_models: Dict[str, Set[str]] = {
            "yolo_detection_v1": {
                "yolov8n.pt",
                "yolov8s.pt",
                "yolov8m.pt",
                "yolo11n.pt",
                "yolo26n.pt",
                "mock_yolo_v1",
            }
        }
        # Whitelisted base checkpoints per model_id -> set of known SHA-256 digests (or wildcard for mocks)
        self._allowed_checkpoints: Dict[str, Set[str]] = {
            "mock_yolo_v1": {"*"},
            "yolo26n.pt": {"9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"},
            "yolo11n.pt": {"0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"},
            "yolov8n.pt": {"f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"},
        }

    def register_adapter_model(
        self, adapter_id: str, model_id: str, allowed_sha256: Optional[str] = None
    ) -> None:
        """Register a new allowed model under an adapter."""
        if adapter_id not in self._allowed_models:
            self._allowed_models[adapter_id] = set()
        self._allowed_models[adapter_id].add(model_id)

        if allowed_sha256:
            if model_id not in self._allowed_checkpoints:
                self._allowed_checkpoints[model_id] = set()
            self._allowed_checkpoints[model_id].add(allowed_sha256)

    def is_adapter_allowed(self, adapter_id: str) -> bool:
        return adapter_id in self._allowed_models

    def is_model_allowed(self, adapter_id: str, model_id: str) -> bool:
        models = self._allowed_models.get(adapter_id)
        return models is not None and model_id in models

    def is_checkpoint_allowed(self, model_id: str, checkpoint_sha256: str) -> bool:
        allowed = self._allowed_checkpoints.get(model_id)
        if not allowed:
            # If no specific checkpoint whitelist registered, require non-empty 64-char hex
            return len(checkpoint_sha256) == 64
        if "*" in allowed:
            return True
        return checkpoint_sha256 in allowed


DEFAULT_TRAINER_REGISTRY = TrainerRegistry()
