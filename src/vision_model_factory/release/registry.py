"""Model registry for candidate, approval, and activation tracking."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ModelRegistry:
    """
    Tracks lifecycle states (candidate, approved, active) in an external index file.
    Published packages are immutable and never modified by registry operations.
    """

    def __init__(self, registry_file: Path):
        self.registry_file = registry_file.resolve()
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_file.is_file():
            with self.registry_file.open("r", encoding="utf-8") as f:
                self.entries = json.load(f)

    def _save(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_file.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, sort_keys=True)

    def register_candidate(self, package_id: str, package_path: Path, manifest_sha256: str) -> None:
        """Register a freshly published model package as a candidate."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.entries[package_id] = {
            "package_id": package_id,
            "status": "candidate",
            "package_path": str(package_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "registered_at": now,
            "approved_at": None,
            "approval_evidence": None,
            "activated_at": None,
        }
        self._save()

    def approve_model(self, package_id: str, evidence: Dict[str, Any]) -> None:
        """Approve a candidate model based on verification evidence."""
        if package_id not in self.entries:
            raise KeyError(f"Package '{package_id}' not found in registry")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.entries[package_id]["status"] = "approved"
        self.entries[package_id]["approved_at"] = now
        self.entries[package_id]["approval_evidence"] = evidence
        self._save()

    def activate_model(self, package_id: str) -> None:
        """Activate an approved model for runtime deployment."""
        if package_id not in self.entries:
            raise KeyError(f"Package '{package_id}' not found in registry")
        if self.entries[package_id]["status"] != "approved":
            st = self.entries[package_id]["status"]
            raise ValueError(
                f"Cannot activate package '{package_id}' with status '{st}'. Must be approved first."
            )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Demote previous active models to approved
        for item in self.entries.values():
            if item["status"] == "active":
                item["status"] = "approved"
        self.entries[package_id]["status"] = "active"
        self.entries[package_id]["activated_at"] = now
        self._save()

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Return the currently active model record, if any."""
        for item in self.entries.values():
            if item["status"] == "active":
                return item
        return None
