"""Atomic release of verified model packages into a release directory.

A staged package is validated completely and only then promoted: the promoted directory
either appears complete or does not appear at all. The manifest's statements about the
model file are read back out of the ONNX graph itself, so a package cannot declare a
640x640 interface for a model exported at another size, or claim fp32 for an
integer-quantized graph.
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from vision_model_factory.contracts.gate_policy import GatePolicy, build_gate, policy_sha256
from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
    DecoderSpec,
    EvaluationReport,
    FileRef,
    InputSpec,
    ModelManifest,
    OutputSpec,
    PostprocessSpec,
    PreprocessSpec,
    TargetSpec,
    model_json,
)
from vision_model_factory.contracts.validators import (
    ValidationError,
    validate_model_package,
)
from vision_model_factory.export.graph_inspection import describe_onnx_graph, infer_precision

PACKAGE_MODEL_FILENAME = "model.onnx"
PACKAGE_TASK_FILENAME = "task.json"
PACKAGE_EVALUATION_FILENAME = "evaluation.json"
PACKAGE_MANIFEST_FILENAME = "model.json"


class ReleasePublicationError(Exception):
    """Raised when model package publication or the release gate fails."""


def utc_now_stamp() -> str:
    """RFC 3339 UTC timestamp, the only time form a published artifact may carry."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def publish_model_package(
    package_id: str,
    model_onnx_path: Path,
    task_json_path: Path,
    evaluation_json_path: Path,
    dataset_ref: DatasetRef,
    run_id: str,
    class_map: List[ClassMapItem],
    releases_dir: Path,
    postprocess_spec: Optional[PostprocessSpec] = None,
    package_created_at: Optional[str] = None,
    gate_policy: Optional[GatePolicy] = None,
) -> Tuple[Path, ModelManifest]:
    """Publish an immutable Model Package atomically.

    Publication requires `gate_policy`: the operator's threshold document. The recorded
    gate in `evaluation.json` is re-derived from that policy and must agree exactly - same
    policy id, same policy digest, same verdict - so editing a `status` string in an
    evaluation file cannot turn a failing model into a releasable one. Omitting the policy
    is refused rather than defaulting to a permissive one.

    Steps:
    1. Refuse to overwrite an existing package id.
    2. Stage model bytes, upstream task.json copy, and evaluation.json.
    3. Read the tensor interface back out of the ONNX graph.
    4. Generate model.json from what the artifacts actually are.
    5. Validate the staged package completely.
    6. Promote the staged directory atomically.
    """
    releases_dir = Path(releases_dir).resolve()
    releases_dir.mkdir(parents=True, exist_ok=True)

    dest_dir = releases_dir / package_id
    if dest_dir.exists():
        raise ReleasePublicationError(
            f"Package '{package_id}' already exists at {dest_dir}. Immutability forbids overwriting."
        )

    source_model = Path(model_onnx_path).resolve()
    if not source_model.is_file():
        raise ReleasePublicationError(f"Model file not found: {source_model}")

    # Staging lives next to the destination so the final move stays on one filesystem,
    # which is what makes the promotion a single atomic rename.
    staging_root = Path(tempfile.mkdtemp(prefix=f".staging-{package_id}-", dir=str(releases_dir)))
    staging_pkg = staging_root / package_id
    try:
        staging_pkg.mkdir(parents=True, exist_ok=True)

        dest_model = staging_pkg / PACKAGE_MODEL_FILENAME
        dest_task = staging_pkg / PACKAGE_TASK_FILENAME
        dest_eval = staging_pkg / PACKAGE_EVALUATION_FILENAME

        shutil.copy2(source_model, dest_model)
        shutil.copy2(task_json_path, dest_task)
        shutil.copy2(evaluation_json_path, dest_eval)

        model_sha = compute_sha256_file(dest_model)
        task_sha = compute_sha256_file(dest_task)
        eval_sha = compute_sha256_file(dest_eval)

        with dest_eval.open("r", encoding="utf-8") as f:
            eval_report = EvaluationReport.model_validate(json.load(f))

        if eval_report.run_id != run_id:
            raise ReleasePublicationError(
                f"evaluation.json describes run '{eval_report.run_id}' but package claims run '{run_id}'."
            )
        if eval_report.dataset.dataset_id != dataset_ref.dataset_id:
            raise ReleasePublicationError(
                f"evaluation.json describes dataset '{eval_report.dataset.dataset_id}' but package "
                f"claims dataset '{dataset_ref.dataset_id}'."
            )
        if eval_report.dataset.manifest_sha256 != dataset_ref.manifest_sha256:
            raise ReleasePublicationError(
                "evaluation.json and the package disagree on the dataset manifest digest; the "
                "published package would not say which data produced it."
            )

        if gate_policy is None:
            raise ReleasePublicationError(
                "Publication requires an explicit gate policy so the release verdict is "
                "traceable to declared thresholds. Refusing to publish against an implicit bar."
            )
        _verify_gate(eval_report, gate_policy)

        # Read the interface from the graph rather than declaring it.
        graph = describe_onnx_graph(dest_model)
        actual_precision = infer_precision(dest_model)

        num_classes = len(class_map)
        output_channels = graph["output"]["shape"][1]
        if isinstance(output_channels, int) and output_channels != 4 + num_classes:
            raise ReleasePublicationError(
                f"Graph output has {output_channels} channels but class_map declares {num_classes} "
                f"classes (decoder yolo_xywh_scores_v1 requires 4 + K = {4 + num_classes})."
            )

        manifest = ModelManifest(
            schema_version="1.0.0",
            package_id=package_id,
            created_at=package_created_at or utc_now_stamp(),
            task_type="object_detection",
            dataset=dataset_ref,
            run_id=run_id,
            task=FileRef(path=PACKAGE_TASK_FILENAME, sha256=task_sha),
            model=FileRef(path=PACKAGE_MODEL_FILENAME, sha256=model_sha),
            evaluation=FileRef(path=PACKAGE_EVALUATION_FILENAME, sha256=eval_sha),
            target=TargetSpec(backend="onnxruntime", provider="cpu", precision=actual_precision),
            input=InputSpec(**graph["input"]),
            preprocess=PreprocessSpec(id="letterbox_rgb_u8_v1", pad_value=114),
            output=OutputSpec(**graph["output"]),
            decoder=DecoderSpec(id="yolo_xywh_scores_v1"),
            class_map=class_map,
            postprocess=postprocess_spec or PostprocessSpec(),
        )

        if eval_report.export_parity.inference_config.score_threshold != manifest.postprocess.score_threshold:
            raise ReleasePublicationError(
                "Parity was measured at score_threshold="
                f"{eval_report.export_parity.inference_config.score_threshold} but the package "
                f"declares {manifest.postprocess.score_threshold}; the published thresholds must be "
                "the ones the evidence was gathered under."
            )

        model_json_path = staging_pkg / PACKAGE_MANIFEST_FILENAME
        model_json_path.write_text(model_json(manifest, indent=2), encoding="utf-8")

        try:
            validate_model_package(staging_pkg)
        except ValidationError as e:
            raise ReleasePublicationError(f"Staged model package failed validation: {e}")

        _atomic_promote(staging_pkg, dest_dir)
        return dest_dir, manifest
    finally:
        # On success the staging directory is empty (the package was moved); on any
        # failure this leaves the release directory untouched and keeps nothing behind.
        shutil.rmtree(staging_root, ignore_errors=True)


def _verify_gate(eval_report: EvaluationReport, policy: GatePolicy) -> None:
    """Re-derive the gate verdict from the operator policy and require an exact match.

    A recorded gate is evidence about a policy, not an authorization by itself. Re-deriving
    it here means the publisher cannot be fooled by an evaluation report whose `status`
    field was edited, whose checks were dropped, or which was produced under a different
    policy than the operator currently enforces.
    """
    recorded = eval_report.gate
    expected_digest = policy_sha256(policy)
    if recorded.policy_sha256 != expected_digest:
        raise ReleasePublicationError(
            f"evaluation.json was gated under policy '{recorded.policy_id}' digest "
            f"{recorded.policy_sha256}, but the supplied policy '{policy.policy_id}' hashes to "
            f"{expected_digest}. Re-run the evaluation against the current policy."
        )

    derived = build_gate(
        policy, eval_report.test, eval_report.export_parity, list(eval_report.target_benchmarks)
    )
    if derived.status != recorded.status:
        raise ReleasePublicationError(
            f"Gate verdict mismatch: evaluation.json records '{recorded.status}' but the supplied "
            f"policy '{policy.policy_id}' derives '{derived.status}' from the recorded measurements. "
            f"Failing checks: {[c.metric for c in derived.checks if not c.passed]}"
        )
    if derived.status != "passed":
        raise ReleasePublicationError(
            f"Model fails gate policy '{policy.policy_id}': "
            f"{[c.metric for c in derived.checks if not c.passed]}"
        )


def _atomic_promote(staging_pkg: Path, dest_dir: Path) -> None:
    """Move a fully validated staging package into the release directory.

    A same-filesystem rename is atomic: readers see either no package or the complete
    package, never a partially written one.
    """
    if dest_dir.exists():
        raise ReleasePublicationError(
            f"Package '{dest_dir.name}' appeared at {dest_dir} during staging. "
            "Refusing to overwrite an existing release."
        )
    try:
        shutil.move(str(staging_pkg), str(dest_dir))
    except shutil.Error as exc:
        raise ReleasePublicationError(f"Atomic promotion of {staging_pkg} failed: {exc}")


__all__ = [
    "PACKAGE_EVALUATION_FILENAME",
    "PACKAGE_MANIFEST_FILENAME",
    "PACKAGE_MODEL_FILENAME",
    "PACKAGE_TASK_FILENAME",
    "ReleasePublicationError",
    "publish_model_package",
    "utc_now_stamp",
]
