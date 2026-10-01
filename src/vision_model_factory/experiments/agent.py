"""Constrained agent proposal validator and boundary enforcement."""

from typing import Any, Dict, Union

from pydantic import ValidationError as PydanticValidationError

from vision_model_factory.contracts.models import ExperimentSpec
from vision_model_factory.contracts.validators import ValidationError
from vision_model_factory.experiments.policy import ExperimentPolicy


class AgentProposalViolation(ValidationError):
    """Raised when an external agent proposes parameters violating policy."""


def validate_agent_proposal(
    proposal: Union[Dict[str, Any], str],
    policy: ExperimentPolicy,
) -> ExperimentSpec:
    """
    Validate an agent-generated proposal against the ExperimentSpec schema and ExperimentPolicy.

    Rejects:
    - Out-of-bounds hyperparameters
    - Unwhitelisted trainers or models
    - Unauthorized configuration modifications
    """
    try:
        if isinstance(proposal, str):
            spec = ExperimentSpec.model_validate_json(proposal)
        else:
            spec = ExperimentSpec.model_validate(proposal)
    except PydanticValidationError as e:
        raise AgentProposalViolation(f"Schema validation failed for agent proposal: {e}")

    # Validate trainer adapter
    adapter_id = spec.trainer.adapter_id
    model_id = spec.trainer.model_id

    if adapter_id not in policy.allowed_adapters:
        raise AgentProposalViolation(
            f"Trainer adapter '{adapter_id}' is not in policy whitelist: {list(policy.allowed_adapters.keys())}"
        )

    allowed_models = policy.allowed_adapters[adapter_id]
    if model_id not in allowed_models:
        raise AgentProposalViolation(
            f"Model '{model_id}' is not allowed for adapter '{adapter_id}': {list(allowed_models)}"
        )

    # Validate parameter ranges
    p = spec.params
    min_img, max_img = policy.imgsz_range
    if not (min_img <= p.imgsz <= max_img):
        raise AgentProposalViolation(
            f"Param 'imgsz'={p.imgsz} outside allowed range [{min_img}, {max_img}]"
        )

    min_b, max_b = policy.batch_range
    if not (min_b <= p.batch <= max_b):
        raise AgentProposalViolation(
            f"Param 'batch'={p.batch} outside allowed range [{min_b}, {max_b}]"
        )

    min_ep, max_ep = policy.epochs_range
    if not (min_ep <= p.epochs <= max_ep):
        raise AgentProposalViolation(
            f"Param 'epochs'={p.epochs} outside allowed range [{min_ep}, {max_ep}]"
        )

    min_lr, max_lr = policy.lr0_range
    if not (min_lr <= p.lr0 <= max_lr):
        raise AgentProposalViolation(
            f"Param 'lr0'={p.lr0} outside allowed range [{min_lr}, {max_lr}]"
        )

    min_mos, max_mos = policy.mosaic_range
    if not (min_mos <= p.mosaic <= max_mos):
        raise AgentProposalViolation(
            f"Param 'mosaic'={p.mosaic} outside allowed range [{min_mos}, {max_mos}]"
        )

    return spec
