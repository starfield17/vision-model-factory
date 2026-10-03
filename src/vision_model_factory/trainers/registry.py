"""Trainer adapter registry, checkpoint pinning and adapter resolution.

This is the single source of truth for which adapters, model ids and base checkpoints
may be executed. The spec requires trainer/model/checkpoint to come from an allowed
registry and to be validated by the *executor*, so the runner resolves adapters here
instead of instantiating one directly, and the CLI assembles the production registry in
one place.

The checkpoint digest is what binds the registry to reality: Ultralytics resolves a bare
`yolov8n.pt` by name and will download it, so a name whitelist alone cannot pin what was
executed. Execution requires a local file whose digest matches a pinned entry.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import TrainerSpec
from vision_model_factory.trainers.base import BaseTrainerAdapter
from vision_model_factory.trainers.yolo import YoloTrainerAdapter


class RegistryError(ValueError):
    """Raised when a trainer request is not covered by the whitelist."""


class TrainerRegistry:
    """Registry maintaining whitelisted trainer adapters, model architectures, and checkpoints."""

    def __init__(self) -> None:
        # adapter_id -> adapter factory. Resolution goes through this table, so an
        # unregistered adapter id can never reach an adapter implementation.
        self._adapter_factories: Dict[str, Callable[[], BaseTrainerAdapter]] = {
            "yolo_detection_v1": YoloTrainerAdapter,
        }

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
        """Whether a checkpoint digest is pinned for this model id.

        Unpinned model ids are rejected rather than accepted on the strength of the
        digest looking like a digest: an unpinned checkpoint is untraceable, and a
        whitelist that admits anything shaped like a hash is not a whitelist. Unknown
        model ids stay false via the empty lookup below.
        """
        allowed = self._allowed_checkpoints.get(model_id)
        if not allowed:
            return False
        if "*" in allowed:
            # Mock adapters produce their own weights per run, so no fixed digest exists.
            return True
        return checkpoint_sha256 in allowed

    def adapter_ids(self) -> List[str]:
        return sorted(self._adapter_factories)

    def allowed_models(self, adapter_id: str) -> List[str]:
        return sorted(self._allowed_models.get(adapter_id, set()))

    def pinned_checkpoints(self, model_id: str) -> List[str]:
        return sorted(self._allowed_checkpoints.get(model_id, set()))

    def resolve_adapter(self, trainer: TrainerSpec) -> BaseTrainerAdapter:
        """Resolve and fully validate a trainer request, returning the adapter instance.

        Raises RegistryError when the adapter, model id or checkpoint digest is not
        whitelisted. Callers that execute a spec go through here, never through an
        adapter constructor directly.
        """
        if not self.is_adapter_allowed(trainer.adapter_id):
            raise RegistryError(
                f"Trainer adapter '{trainer.adapter_id}' is not registered; allowed: {self.adapter_ids()}"
            )
        if not self.is_model_allowed(trainer.adapter_id, trainer.model_id):
            raise RegistryError(
                f"Model '{trainer.model_id}' is not allowed for adapter '{trainer.adapter_id}'; "
                f"allowed: {self.allowed_models(trainer.adapter_id)}"
            )
        if not self.is_checkpoint_allowed(trainer.model_id, trainer.checkpoint_sha256):
            raise RegistryError(
                f"Checkpoint digest {trainer.checkpoint_sha256} is not pinned for model "
                f"'{trainer.model_id}'; pinned: {self.pinned_checkpoints(trainer.model_id)}"
            )
        return self._adapter_factories[trainer.adapter_id]()

    def verify_checkpoint_file(self, trainer: TrainerSpec, checkpoint_path: Path) -> Path:
        """Confirm a local checkpoint file exists and hashes to the pinned digest.

        Returns the resolved path. Mock model ids have no file on disk, so they are
        exempt from the file check and are reported through the registry as mocks.
        """
        if self.is_mock_model(trainer.model_id):
            return checkpoint_path
        if not checkpoint_path.is_file():
            raise RegistryError(
                f"Base checkpoint file for '{trainer.model_id}' not found at {checkpoint_path}. "
                "Execution refuses to fall back to a network download, because an "
                "un-pinned download would execute weights the digest whitelist never approved."
            )
        actual = compute_sha256_file(checkpoint_path)
        if actual != trainer.checkpoint_sha256:
            raise RegistryError(
                f"Base checkpoint digest mismatch for '{trainer.model_id}': "
                f"expected {trainer.checkpoint_sha256}, got {actual}"
            )
        return checkpoint_path

    def is_mock_model(self, model_id: str) -> bool:
        return "*" in self._allowed_checkpoints.get(model_id, set())


def load_production_registry(overrides: Optional[Dict[str, Set[str]]] = None) -> TrainerRegistry:
    """Composition root for the executable trainer whitelist.

    `overrides` maps model_id -> set of additionally pinned digests, letting an operator
    add a locally verified checkpoint without editing source. It is intentionally narrow:
    it can pin additional digests, it cannot remove the `*` wildcard or drop a model.
    """
    registry = TrainerRegistry()
    for model_id, digests in (overrides or {}).items():
        for digest in digests:
            registry.register_adapter_model(
                "yolo_detection_v1", model_id, allowed_sha256=digest
            )
    return registry
