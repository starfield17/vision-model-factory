"""Constrained agent proposal validation against policy and trainer registry."""

from typing import Any, Dict, Union

from pydantic import ValidationError as PydanticValidationError

from vision_model_factory.contracts.models import ExperimentSpec
from vision_model_factory.contracts.validators import ValidationError
from vision_model_factory.experiments.policy import ExperimentPolicy
from vision_model_factory.trainers.registry import TrainerRegistry


class AgentProposalViolation(ValidationError):
    """Raised when an external agent proposes parameters violating policy."""


def validate_agent_proposal(
    proposal: Union[Dict[str, Any], str],
    policy: ExperimentPolicy,
    registry: TrainerRegistry,
) -> ExperimentSpec:
    """
    Validate an agent-generated proposal against the ExperimentSpec schema, the budget
    policy, and the trainer registry.

    Rejects:
    - Structurally invalid specs (extra or unknown fields included)
    - Out-of-bounds hyperparameters and seed-policy violations
    - Adapters, model ids or checkpoint digests outside the registry whitelist
    """
    try:
        if isinstance(proposal, str):
            spec = ExperimentSpec.model_validate_json(proposal)
        else:
            spec = ExperimentSpec.model_validate(proposal)
    except PydanticValidationError as e:
        raise AgentProposalViolation(f"Schema validation failed for agent proposal: {e}")

    # Trainer/model/checkpoint whitelist comes from the registry, which is also what the
    # runner enforces at execution time. Validating against a second copy of the list
    # would let a proposal pass review and then fail (or bypass) execution.
    try:
        registry.resolve_adapter(spec.trainer)
    except Exception as exc:  # noqa: BLE001 - normalised into the agent-facing error type
        raise AgentProposalViolation(f"Trainer request rejected by registry: {exc}")

    violations = policy.param_violations(spec.params)
    if violations:
        raise AgentProposalViolation("; ".join(violations))

    return spec
