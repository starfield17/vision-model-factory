"""Gate policy contract: the shape of a policy document and the gate derivation.

`contracts` owns the *type* of a publication policy and the pure function that derives a
gate verdict from it. It deliberately owns no threshold values: there is no default
policy constant anywhere in this repository, because a hard-coded bar is precisely what
made the previous gate unaccountable. Concrete thresholds live in operator-supplied
policy documents identified by their own SHA-256, so a gate verdict always names and
pins the policy that produced it.

The architecture spec requires publication thresholds to be configured explicitly
("发布策略显式配置指标上下限、目标 profile、容差") and refuses support for profiles that were
not measured. `evaluation` calls `build_gate` to record a verdict; `release` re-derives
it from the operator's policy before publishing, so neither the measured artifact nor
its author can choose the bar after the fact.

Placement note: this lives in `contracts` rather than `release` because `evaluation` may
only import `contracts` under the declared dependency rules, and the gate verdict must
not be derivable by the layer being measured.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from vision_model_factory.contracts.hashing import compute_sha256_bytes
from vision_model_factory.contracts.models import (
    BaseContractModel,
    ExportParitySection,
    GateCheck,
    GateSection,
    TestEvaluationSection,
)


class ClassQualityFloor(BaseContractModel):
    """Precision/recall floor for one critical class."""

    class_id: str = Field(..., min_length=1)
    min_precision: float = Field(..., ge=0.0, le=1.0)
    min_recall: float = Field(..., ge=0.0, le=1.0)


class TargetProfile(BaseContractModel):
    """A deployment profile that must have been measured before publication."""

    backend: Literal["onnxruntime"] = "onnxruntime"
    provider: Literal["cpu"] = "cpu"
    precision: Literal["fp32", "int8"] = "fp32"
    max_latency_ms_p95: Optional[float] = Field(None, gt=0.0)
    max_peak_memory_bytes: Optional[int] = Field(None, gt=0)


class GatePolicy(BaseContractModel):
    """Explicit publication thresholds for one task/deployment target."""

    policy_id: str = Field(..., min_length=1)
    min_map50: float = Field(..., ge=0.0, le=1.0)
    min_map50_95: float = Field(..., ge=0.0, le=1.0)
    critical_classes: List[ClassQualityFloor] = Field(default_factory=list)
    require_export_parity_passed: bool = True
    required_target_profiles: List[TargetProfile] = Field(default_factory=list)
    extensions: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_map_ordering(self) -> "GatePolicy":
        # mAP50:95 is a strictly harder metric than mAP50; a policy that inverts them
        # would let a model pass a ceiling it cannot actually clear.
        if self.min_map50_95 > self.min_map50:
            raise ValueError(
                f"policy '{self.policy_id}': min_map50_95 ({self.min_map50_95}) must not exceed "
                f"min_map50 ({self.min_map50})"
            )
        seen = set()
        for floor in self.critical_classes:
            if floor.class_id in seen:
                raise ValueError(f"policy '{self.policy_id}': duplicate critical class '{floor.class_id}'")
            seen.add(floor.class_id)
        return self


def policy_sha256(policy: GatePolicy) -> str:
    """Canonical digest of a gate policy document."""
    canonical = json.dumps(
        policy.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":")
    )
    return compute_sha256_bytes(canonical.encode("utf-8"))


def load_gate_policy_file(path: Path) -> GatePolicy:
    """Load and validate a gate policy document."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return GatePolicy.model_validate(raw)


def _check(
    metric: str,
    threshold: float,
    actual: float,
    direction: str = "min",
    class_id: Optional[str] = None,
) -> GateCheck:
    passed = actual >= threshold if direction == "min" else actual <= threshold
    return GateCheck(
        metric=metric,
        threshold=threshold,
        actual=actual,
        passed=passed,
        direction=direction,
        class_id=class_id,
    )


def build_gate(
    policy: GatePolicy,
    test: TestEvaluationSection,
    export_parity: Optional[ExportParitySection] = None,
    target_benchmarks: Optional[List[Any]] = None,
) -> GateSection:
    """Derive the gate verdict for an evaluation report from an explicit policy.

    Every threshold and every required profile comes from the policy document. Nothing
    here has a default verdict: a missing required benchmark or a missing critical class
    is a failed check, not an absent one.
    """
    checks: List[GateCheck] = [
        _check("mAP50", policy.min_map50, test.mAP50),
        _check("mAP50_95", policy.min_map50_95, test.mAP50_95),
    ]

    for floor in policy.critical_classes:
        cls = test.classes.get(floor.class_id)
        if cls is None:
            checks.append(_check("critical_class_reported", 1.0, 0.0, class_id=floor.class_id))
            checks.append(_check("precision", floor.min_precision, 0.0, class_id=floor.class_id))
            checks.append(_check("recall", floor.min_recall, 0.0, class_id=floor.class_id))
            continue
        checks.append(_check("precision", floor.min_precision, cls.precision, class_id=floor.class_id))
        checks.append(_check("recall", floor.min_recall, cls.recall, class_id=floor.class_id))

    if policy.require_export_parity_passed:
        passed = export_parity is not None and export_parity.status == "passed"
        checks.append(_check("export_parity_passed", 1.0, 1.0 if passed else 0.0))

    for profile in policy.required_target_profiles:
        matching = [
            b
            for b in (target_benchmarks or [])
            if getattr(b, "provider", None) == profile.provider
            and getattr(b, "runtime", None) == profile.backend
        ]
        if not matching:
            checks.append(_check(f"target_benchmark[{profile.provider}/{profile.precision}]", 1.0, 0.0))
            continue
        # Best measurement for the profile wins; a profile is supported if any
        # measurement against it clears the declared ceilings.
        best_p95 = min(b.latency_ms_p95 for b in matching)
        if profile.max_latency_ms_p95 is not None:
            checks.append(
                _check(f"latency_ms_p95[{profile.provider}]", profile.max_latency_ms_p95, best_p95, direction="max")
            )
        if profile.max_peak_memory_bytes is not None:
            best_mem = min(b.peak_memory_bytes for b in matching)
            checks.append(
                _check(
                    f"peak_memory_bytes[{profile.provider}]",
                    float(profile.max_peak_memory_bytes),
                    float(best_mem),
                    direction="max",
                )
            )

    status: Literal["passed", "failed"] = "passed" if all(c.passed for c in checks) else "failed"
    return GateSection(
        policy_id=policy.policy_id,
        policy_sha256=policy_sha256(policy),
        status=status,
        checks=checks,
    )


__all__ = [
    "ClassQualityFloor",
    "GatePolicy",
    "TargetProfile",
    "build_gate",
    "load_gate_policy_file",
    "policy_sha256",
]
