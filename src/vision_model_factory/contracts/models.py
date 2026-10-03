"""Pydantic data models for Vision Model Factory contracts."""

import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vision_model_factory.contracts.hashing import is_valid_sha256

# RFC 3339 in UTC. Mirrors $defs.utcTimestamp in the bundled JSON Schemas; time stamps in
# published artifacts are required to be UTC by 00-overall-architecture.md 3.6.
UTC_TIMESTAMP_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"


class BaseContractModel(BaseModel):
    """Base model enforcing extra forbidden and strict attribute access."""

    model_config = ConfigDict(extra="forbid")


def model_json(model: BaseContractModel, indent: Optional[int] = None) -> str:
    """Serialize a contract model to JSON, omitting unset optional fields.

    An explicit helper rather than an override of model_dump/model_dump_json: overriding
    those would silently change serialization semantics for every caller, including the
    `effective_config_sha256` digest computed in the runner. The contract declares
    optional fields as "optional object"; an explicit null would be a third, undefined
    state that consumers would have to guess about.
    """

    if indent is None:
        return model.model_dump_json(exclude_none=True)
    return json.dumps(model.model_dump(mode="json", exclude_none=True), indent=indent, ensure_ascii=False)


class FileRef(BaseContractModel):
    """Reference to a file within a package by relative path and SHA-256."""

    path: str = Field(..., description="Relative path within the package bundle")
    sha256: str = Field(..., description="64-character lowercase SHA-256 digest")

    @field_validator("path")
    @classmethod
    def check_relative_path(cls, v: str) -> str:
        if v.startswith("/") or v.startswith("\\"):
            raise ValueError(f"Path must be relative, got absolute: {v}")
        parts = v.replace("\\", "/").split("/")
        if ".." in parts:
            raise ValueError(f"Path must not contain '..': {v}")
        if not v.strip():
            raise ValueError("Path must not be empty")
        return v

    @field_validator("sha256")
    @classmethod
    def check_sha256(cls, v: str) -> str:
        if not is_valid_sha256(v):
            raise ValueError(f"Invalid SHA-256 digest: {v}")
        return v


class DatasetRef(BaseContractModel):
    """Reference to an immutable dataset package."""

    dataset_id: str = Field(..., min_length=1)
    manifest_sha256: str = Field(...)

    @field_validator("manifest_sha256")
    @classmethod
    def check_sha256(cls, v: str) -> str:
        if not is_valid_sha256(v):
            raise ValueError(f"Invalid manifest SHA-256: {v}")
        return v


class CategorySpec(BaseContractModel):
    """Single category specification."""

    class_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)


class TaskSpec(BaseContractModel):
    """Task specification owned by Data Factory and consumed by Model Factory."""

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    task_id: str = Field(..., min_length=1)
    task_type: Literal["object_detection"] = "object_detection"
    categories: List[CategorySpec] = Field(..., min_length=1)
    extensions: Optional[Dict[str, Any]] = None


class DatasetManifest(BaseContractModel):
    """Immutable dataset package manifest owned by Data Factory."""

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    dataset_id: str = Field(..., min_length=1)
    created_at: str = Field(
        ..., pattern=UTC_TIMESTAMP_PATTERN, description="RFC 3339 UTC timestamp (Z suffix required)"
    )
    task: FileRef
    samples: FileRef
    annotations: FileRef
    quality: FileRef
    reviews: Optional[FileRef] = None
    extensions: Optional[Dict[str, Any]] = None


class SampleRecord(BaseContractModel):
    """A single media sample within samples.jsonl."""

    sample_id: str = Field(..., min_length=1)
    file: FileRef
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    group_id: str = Field(..., min_length=1)
    split: Literal["train", "val", "test", "audit"]
    annotation_status: Literal["partial", "complete_pseudo", "complete_verified"]
    source: Optional[Dict[str, Any]] = None
    extensions: Optional[Dict[str, Any]] = None


class AnnotationRecord(BaseContractModel):
    """A single object bounding box annotation in annotations.jsonl."""

    annotation_id: str = Field(..., min_length=1)
    sample_id: str = Field(..., min_length=1)
    class_id: str = Field(..., min_length=1)
    bbox_xyxy: List[float] = Field(..., min_length=4, max_length=4)
    origin: Literal["model", "human", "imported"]
    annotator_run_id: str = Field(..., min_length=1)
    review_state: Literal["unreviewed", "model_passed", "human_verified"]
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    extensions: Optional[Dict[str, Any]] = None

    @field_validator("bbox_xyxy")
    @classmethod
    def check_bbox_coords(cls, v: List[float]) -> List[float]:
        x1, y1, x2, y2 = v
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"Invalid bounding box coordinates (must have x1 < x2 and y1 < y2): {v}")
        return v


class TrainerSpec(BaseContractModel):
    """Specification of the training adapter and base model."""

    adapter_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    checkpoint_sha256: str = Field(...)

    @field_validator("checkpoint_sha256")
    @classmethod
    def check_sha256(cls, v: str) -> str:
        if not is_valid_sha256(v):
            raise ValueError(f"Invalid checkpoint SHA-256: {v}")
        return v


class ParamsSpec(BaseContractModel):
    """Bounded hyperparameter configuration for training experiments."""

    imgsz: int = Field(..., gt=0)
    batch: int = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    seed: int = Field(..., ge=0)
    lr0: float = Field(..., gt=0.0, le=1.0)
    mosaic: float = Field(..., ge=0.0, le=1.0)


class ExperimentSpec(BaseContractModel):
    """Declarative specification for a constrained training experiment."""

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    experiment_id: str = Field(..., min_length=1)
    dataset: DatasetRef
    trainer: TrainerSpec
    params: ParamsSpec
    reason: str = Field(..., min_length=1)
    parent_run_id: Optional[str] = None
    extensions: Optional[Dict[str, Any]] = None


class RunResult(BaseContractModel):
    """Outcome and execution evidence for a single training run."""

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    run_id: str = Field(..., min_length=1)
    experiment_id: str = Field(..., min_length=1)
    status: Literal["succeeded", "failed", "timeout", "cancelled"]
    dataset: DatasetRef
    effective_config_sha256: str = Field(...)
    environment: Dict[str, Any]
    duration_seconds: float = Field(..., ge=0.0)
    artifacts: Dict[str, FileRef]
    diagnostics: List[str] = Field(default_factory=list)
    val_metrics: Optional[Dict[str, Any]] = None
    extensions: Optional[Dict[str, Any]] = None

    @field_validator("effective_config_sha256")
    @classmethod
    def check_sha256(cls, v: str) -> str:
        if not is_valid_sha256(v):
            raise ValueError(f"Invalid effective_config SHA-256: {v}")
        return v


class TargetSpec(BaseContractModel):
    """Target deployment environment specification.

    v1 declares only what this repository can actually produce and verify. `cuda`,
    `tensorrt` and `rknn` backends are deliberately absent: no implementation and no
    target-machine verification exists for them, and a declared-but-unverified profile
    is exactly what the architecture spec forbids ("没有对应实测的 profile 不声明支持").
    """

    backend: Literal["onnxruntime"] = "onnxruntime"
    provider: Literal["cpu"] = "cpu"
    precision: Literal["fp32", "int8"] = "fp32"


class InputSpec(BaseContractModel):
    """Model input tensor declaration."""

    name: str = Field(..., min_length=1)
    dtype: Literal["float32", "uint8"] = "float32"
    shape: List[Union[int, str]] = Field(..., min_length=4, max_length=4)


class PreprocessSpec(BaseContractModel):
    """Declarative input preprocessing pipeline."""

    id: Literal["letterbox_rgb_u8_v1"] = "letterbox_rgb_u8_v1"
    pad_value: int = Field(114, ge=0, le=255)


class OutputSpec(BaseContractModel):
    """Model output tensor declaration."""

    name: str = Field(..., min_length=1)
    dtype: Literal["float32"] = "float32"
    shape: List[Union[int, str]] = Field(..., min_length=3)


class DecoderSpec(BaseContractModel):
    """Declarative output decoder specification."""

    id: Literal["yolo_xywh_scores_v1"] = "yolo_xywh_scores_v1"


class ClassMapItem(BaseContractModel):
    """Mapping from model output channel index to stable class_id."""

    index: int = Field(..., ge=0)
    class_id: str = Field(..., min_length=1)


class PostprocessSpec(BaseContractModel):
    """Thresholds and NMS parameters for candidate filtering."""

    score_threshold: float = Field(0.25, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    max_detections: int = Field(100, gt=0)


class ModelManifest(BaseContractModel):
    """Immutable Model Package manifest (model.json)."""

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    package_id: str = Field(..., min_length=1)
    created_at: str = Field(
        ..., pattern=UTC_TIMESTAMP_PATTERN, description="RFC 3339 UTC timestamp (Z suffix required)"
    )
    task_type: Literal["object_detection"] = "object_detection"
    dataset: DatasetRef
    run_id: str = Field(..., min_length=1)
    task: FileRef
    model: FileRef
    evaluation: FileRef
    target: TargetSpec
    input: InputSpec
    preprocess: PreprocessSpec
    output: OutputSpec
    decoder: DecoderSpec
    class_map: List[ClassMapItem] = Field(..., min_length=1)
    postprocess: PostprocessSpec
    extensions: Optional[Dict[str, Any]] = None


class InferenceConfig(BaseContractModel):
    """The postprocess and geometry a measurement was actually taken under.

    Carried alongside every measured section so numbers from different configurations
    cannot be silently mixed, and so a threshold override is recorded rather than
    implicit. Mirrors `$defs.inferenceConfig` in the evaluation report schema.
    """

    score_threshold: float = Field(..., ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(..., ge=0.0, le=1.0)
    max_detections: int = Field(..., gt=0)
    target_shape: List[int] = Field(..., min_length=2, max_length=2)

    @field_validator("target_shape")
    @classmethod
    def check_positive(cls, v: List[int]) -> List[int]:
        if any(d <= 0 for d in v):
            raise ValueError(f"target_shape dimensions must be positive, got {v}")
        return v

    @classmethod
    def from_manifest(cls, manifest: "ModelManifest") -> "InferenceConfig":
        """Derive the config a model package declares, refusing dynamic spatial input."""
        spatial = manifest.input.shape[2:4]
        target = [d for d in spatial if isinstance(d, int)]
        if len(target) != 2:
            raise ValueError(
                f"Cannot derive a fixed evaluation geometry from input shape "
                f"{manifest.input.shape}; measurement requires static spatial dimensions."
            )
        return cls(
            score_threshold=manifest.postprocess.score_threshold,
            nms_iou_threshold=manifest.postprocess.nms_iou_threshold,
            max_detections=manifest.postprocess.max_detections,
            target_shape=target,
        )


class PerClassDetectionMetrics(BaseContractModel):
    """Per-class detection quality on the evaluated split."""

    class_id: str = Field(..., min_length=1)
    ap50: float = Field(..., ge=0.0, le=1.0)
    ap50_95: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    total_gt: int = Field(..., ge=0)
    total_pred: int = Field(..., ge=0)


class TestEvaluationSection(BaseContractModel):
    """Locked test split quality evidence.

    Carries the evaluation data/protocol reference, sample counts and the actual
    inference configuration, as required by the evaluation contract.
    """

    protocol_id: str = Field(..., min_length=1)
    ground_truth: Literal["human_verified"] = "human_verified"
    sample_count: int = Field(..., gt=0)
    gt_annotation_count: int = Field(..., ge=0)
    excluded_partial_count: int = Field(..., ge=0)
    mAP50: float = Field(..., ge=0.0, le=1.0)
    mAP50_95: float = Field(..., ge=0.0, le=1.0)
    classes: Dict[str, PerClassDetectionMetrics]
    inference_config: InferenceConfig
    extensions: Optional[Dict[str, Any]] = None

    @field_validator("classes")
    @classmethod
    def check_classes_nonempty(cls, v: Dict[str, PerClassDetectionMetrics]):
        if not v:
            raise ValueError("test.classes must report at least one class")
        return v


class ParityTolerances(BaseContractModel):
    """Declared comparison tolerances for export parity."""

    tensor_atol: float = Field(..., ge=0.0)
    score_atol: float = Field(..., ge=0.0)
    box_atol: float = Field(..., ge=0.0)


class ParityTensorComparison(BaseContractModel):
    """Raw output tensor difference between reference and exported model.

    `max_rel_diff` is the absolute difference normalised by max(1.0, |reference|). It is
    recorded because a single absolute tolerance is not scale-coherent across a tensor that
    carries pixel-space coordinates and class scores together: the absolute figure alone can
    fail on two graphs that compute the same function, and the relative figure is what shows
    that. Thresholds are not derived from it here.
    """

    max_abs_diff: float = Field(..., ge=0.0)
    max_rel_diff: float = Field(..., ge=0.0)
    mean_abs_diff: float = Field(..., ge=0.0)
    passed: bool


class ParityDetectionComparison(BaseContractModel):
    """Decoded detection-level difference between reference and exported model."""

    ref_count: int = Field(..., ge=0)
    ort_count: int = Field(..., ge=0)
    matched: bool
    max_score_diff: float = Field(..., ge=0.0)
    max_box_diff: float = Field(..., ge=0.0)


class ParitySelfTest(BaseContractModel):
    """Proof that a parity comparison can actually fail.

    Parity that never saw a difference proves nothing: two empty detection lists "match"
    trivially. The comparison is therefore required to report what it detected when the
    reference detections were deliberately perturbed.
    """

    perturbation_px: float = Field(..., gt=0.0)
    detected_perturbation: bool


class ExportParitySection(BaseContractModel):
    """Export parity evidence.

    A `passed` status is structurally impossible without non-empty decoded detections
    on both sides: an empty-vs-empty comparison proves nothing about coordinate
    recovery and must not be recordable as a pass.
    """

    status: Literal["passed", "failed"]
    method: str = Field(..., min_length=1)
    input_reference: str = Field(..., min_length=1)
    # `test` is absent from the type on purpose. Parity compares a reference
    # implementation against the exported graph, so it needs real image bytes but
    # consumes no annotations and measures no quality. Spending the locked test corpus
    # on it would mean the scored set had also been used to validate the artifact, so a
    # parity pass citing it is made unrepresentable rather than discouraged.
    parity_split: Optional[Literal["train", "val", "audit"]] = None
    matching_method: str = Field(..., min_length=1)
    tolerances: ParityTolerances
    raw_tensor: ParityTensorComparison
    detections: ParityDetectionComparison
    # Optional because parity that was never run has no self-test to report; a `passed`
    # verdict must carry one, enforced below.
    self_test: Optional[ParitySelfTest] = None
    exceptions: List[str] = Field(default_factory=list)
    inference_config: InferenceConfig
    extensions: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_passed_is_substantiated(self) -> "ExportParitySection":
        if self.status != "passed":
            return self
        if not self.raw_tensor.passed:
            raise ValueError("export_parity.status cannot be 'passed' while raw_tensor.passed is false")
        if not self.detections.matched:
            raise ValueError("export_parity.status cannot be 'passed' while detections.matched is false")
        if self.detections.ref_count < 1 or self.detections.ort_count < 1:
            raise ValueError(
                "export_parity.status cannot be 'passed' with empty detections "
                f"(ref_count={self.detections.ref_count}, ort_count={self.detections.ort_count}): "
                "an empty comparison does not verify coordinate recovery"
            )
        if self.detections.ref_count != self.detections.ort_count:
            raise ValueError("export_parity.status cannot be 'passed' with mismatched detection counts")
        if self.self_test is None:
            raise ValueError(
                "export_parity.status cannot be 'passed' without a self_test: a parity result "
                "that never demonstrated it can see a coordinate difference proves nothing"
            )
        if not self.self_test.detected_perturbation:
            raise ValueError(
                "export_parity.status cannot be 'passed' when the self-test failed to detect a "
                f"{self.self_test.perturbation_px}px coordinate perturbation: the comparison "
                "cannot see coordinate errors and is vacuous"
            )
        if self.parity_split is None:
            raise ValueError(
                "export_parity.status cannot be 'passed' without recording which split the "
                "parity corpus came from: an unattributed corpus cannot be audited for "
                "locked-test-set consumption"
            )
        return self


class TargetBenchmark(BaseContractModel):
    """Target-machine performance measurement bound to a concrete environment.

    The environment binding is mandatory: a latency number without OS, architecture,
    runtime version and device cannot be re-derived or compared, and the contract
    forbids declaring support for an unmeasured profile.
    """

    os_name: str = Field(..., min_length=1)
    os_version: str = Field(..., min_length=1)
    architecture: str = Field(..., min_length=1)
    device: str = Field(..., min_length=1)
    runtime: str = Field(..., min_length=1)
    runtime_version: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    providers_used: List[str] = Field(..., min_length=1)
    requested_providers: List[str] = Field(..., min_length=1)
    warmup_runs: int = Field(..., ge=0)
    benchmark_runs: int = Field(..., gt=0)
    latency_ms_p50: float = Field(..., gt=0.0)
    latency_ms_p95: float = Field(..., gt=0.0)
    latency_ms_mean: float = Field(..., gt=0.0)
    peak_memory_bytes: int = Field(..., gt=0)
    input_reference: str = Field(..., min_length=1)
    inference_config: InferenceConfig
    extensions: Optional[Dict[str, Any]] = None


class GateCheck(BaseContractModel):
    """Single threshold evaluation inside a release gate decision.

    `direction` decides whether the metric must meet a lower or an upper bound, so
    latency-style ceilings live in the same explicit policy as quality floors.
    """

    metric: str = Field(..., min_length=1)
    threshold: float
    actual: float
    passed: bool
    direction: Literal["min", "max"] = "min"
    class_id: Optional[str] = None


class GateSection(BaseContractModel):
    """Release gate decision derived from an explicitly configured policy.

    Thresholds live in the policy, never in code defaults, and `status` must agree
    with the recorded checks.
    """

    policy_id: str = Field(..., min_length=1)
    policy_sha256: str = Field(...)
    status: Literal["passed", "failed"]
    checks: List[GateCheck] = Field(..., min_length=1)
    extensions: Optional[Dict[str, Any]] = None

    @field_validator("policy_sha256")
    @classmethod
    def check_sha256(cls, v: str) -> str:
        if not is_valid_sha256(v):
            raise ValueError(f"Invalid gate policy SHA-256: {v}")
        if v == "0" * 64:
            raise ValueError("gate.policy_sha256 must be a real digest of the policy document, not a placeholder")
        return v

    @model_validator(mode="after")
    def check_status_matches_checks(self) -> "GateSection":
        all_passed = all(c.passed for c in self.checks)
        expected = "passed" if all_passed else "failed"
        if self.status != expected:
            raise ValueError(
                f"gate.status is '{self.status}' but recorded checks imply '{expected}'"
            )
        for check in self.checks:
            check_passed = (
                check.actual >= check.threshold if check.direction == "min" else check.actual <= check.threshold
            )
            if check.passed != check_passed:
                raise ValueError(
                    f"gate check '{check.metric}' (class={check.class_id}) records passed="
                    f"{check.passed} but actual={check.actual} vs threshold={check.threshold} "
                    "implies passed=" + str(check_passed)
                )
        return self


class EvaluationReport(BaseContractModel):
    """Independent evaluation report (evaluation.json).

    val / locked test / export parity / target performance stay in separate sections
    and are never collapsed into a single score.
    """

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    run_id: str = Field(..., min_length=1)
    dataset: DatasetRef
    test: TestEvaluationSection
    export_parity: ExportParitySection
    target_benchmarks: List[TargetBenchmark] = Field(default_factory=list)
    gate: GateSection
    extensions: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_gate_actuals_match_measurements(self) -> "EvaluationReport":
        """A gate check must report the number the report actually measured.

        `gate.checks[].actual` is a restatement of a measurement, not an independent
        observation. Allowing the two to differ means a document can keep its headline
        mAP low while its gate records a passing actual (or the reverse), which makes the
        recorded verdict meaningless without anyone noticing. Re-deriving the verdict from
        an operator policy (`release.publisher`) still cannot catch a report whose
        *measurements* were rewritten, so this closes the half of the forgery that is
        detectable from the document alone.
        """
        measured = {"mAP50": self.test.mAP50, "mAP50_95": self.test.mAP50_95}
        for check in self.gate.checks:
            if check.class_id is not None:
                continue
            expected = measured.get(check.metric)
            if expected is not None and abs(check.actual - expected) > 1e-9:
                raise ValueError(
                    f"gate check '{check.metric}' records actual={check.actual} but the test "
                    f"section measured {expected}; a gate must restate the measurement it "
                    "is gating on, not a separate number"
                )
        return self
