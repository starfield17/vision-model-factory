"""Pydantic data models for Vision Model Factory contracts."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vision_model_factory.contracts.hashing import is_valid_sha256


class BaseContractModel(BaseModel):
    """Base model enforcing extra forbidden and strict attribute access."""

    model_config = ConfigDict(extra="forbid")


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
    created_at: str = Field(..., min_length=1)
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
    """Target deployment environment specification."""

    backend: Literal["onnxruntime"] = "onnxruntime"
    provider: Literal["cpu", "cuda", "tensorrt"] = "cpu"
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
    created_at: str = Field(..., min_length=1)
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


class EvaluationReport(BaseContractModel):
    """Independent evaluation report (evaluation.json)."""

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    run_id: str = Field(..., min_length=1)
    dataset: DatasetRef
    test: Dict[str, Any]
    export_parity: Dict[str, Any]
    target_benchmarks: List[Dict[str, Any]]
    gate: Dict[str, Any]
    extensions: Optional[Dict[str, Any]] = None
