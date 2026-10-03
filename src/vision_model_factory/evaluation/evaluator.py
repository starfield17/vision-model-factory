"""Locked test split evaluator for independent model quality assessment."""

import platform
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vision_model_factory.contracts.gate_policy import GatePolicy, build_gate
from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
    EvaluationReport,
    ExportParitySection,
    InferenceConfig,
    PerClassDetectionMetrics,
    SampleRecord,
    TargetBenchmark,
    TestEvaluationSection,
)
from vision_model_factory.contracts.validators import validate_dataset_package
from vision_model_factory.evaluation.metrics import (
    compute_class_detection_metrics,
    mean_average_precision,
)

TEST_PROTOCOL_ID = "locked_test_per_image_ap_v1"


class TestEvaluationError(Exception):
    """Raised when test split evaluation constraints are violated."""

    # The name's "Test" prefix makes pytest try to collect this exception as a test class
    # wherever a test module imports it. This is pytest's own opt-out; renaming the
    # exception would break a published API for a warning.
    __test__ = False


def compute_dataset_manifest_sha(dataset_dir: Path) -> str:
    """Compute SHA-256 digest of dataset.json manifest file."""
    return compute_sha256_file(Path(dataset_dir) / "dataset.json")


def parity_not_measured(config: InferenceConfig) -> ExportParitySection:
    """A structurally valid ExportParitySection that records "parity was not run".

    `parity_split` stays absent because no corpus was consumed: claiming a split for a
    comparison that never happened would attribute evidence to data it never touched.
    """
    return ExportParitySection(
        status="failed",
        method="not_run",
        input_reference="none",
        matching_method="none",
        tolerances={"tensor_atol": 0.0, "score_atol": 0.0, "box_atol": 0.0},
        raw_tensor={"max_abs_diff": 0.0, "max_rel_diff": 0.0, "mean_abs_diff": 0.0, "passed": False},
        detections={
            "ref_count": 0,
            "ort_count": 0,
            "matched": False,
            "max_score_diff": 0.0,
            "max_box_diff": 0.0,
        },
        exceptions=["parity was not attempted: no reference model was supplied"],
        inference_config=config,
    )


def evaluate_test_split(
    dataset_dir: Path,
    predict_fn: Callable[[SampleRecord], List[Dict[str, Any]]],
    class_map: List[ClassMapItem],
    inference_config: InferenceConfig,
    run_id: str,
    gate_policy: GatePolicy,
    export_parity: Optional[ExportParitySection] = None,
    target_benchmarks: Optional[List[TargetBenchmark]] = None,
) -> EvaluationReport:
    """Perform locked evaluation strictly on the test split of a dataset package.

    `class_map` and `inference_config` describe the artifact under evaluation, so this
    function needs nothing beyond the artifact's declared interface. Predictions and
    ground truth are kept per image. A class the test split cannot measure aborts the
    evaluation rather than being averaged in as 0 or 1. The gate verdict is derived from
    the supplied policy document, never from a code default.
    """
    ds_manifest, task, samples, annotations = validate_dataset_package(dataset_dir)

    test_samples: List[SampleRecord] = []
    excluded_partial_count = 0
    for s in samples:
        if s.split != "test":
            continue
        if s.annotation_status == "partial":
            excluded_partial_count += 1
            continue
        test_samples.append(s)

    if not test_samples:
        raise TestEvaluationError("No valid non-partial samples found in test split for evaluation.")

    test_sample_ids = {s.sample_id for s in test_samples}
    ground_truth_by_sample: Dict[str, List[Dict[str, Any]]] = {sid: [] for sid in test_sample_ids}
    for a in annotations:
        if a.sample_id in test_sample_ids:
            ground_truth_by_sample[a.sample_id].append(
                {"class_id": a.class_id, "bbox_xyxy": list(a.bbox_xyxy)}
            )

    predictions_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for s in test_samples:
        predictions_by_sample[s.sample_id] = predict_fn(s)

    config = inference_config
    class_ids = sorted({c.class_id for c in class_map})

    # A class the locked test set cannot measure makes the headline mean ill-defined:
    # scoring it 0 unfairly penalises the model, scoring it 1 fabricates credit. Refuse
    # and require a dataset release with adequate coverage instead.
    gt_class_ids = {g["class_id"] for entries in ground_truth_by_sample.values() for g in entries}
    unmeasurable = [cid for cid in class_ids if cid not in gt_class_ids]
    if unmeasurable:
        raise TestEvaluationError(
            f"Locked test split contains no ground truth for class(es) {unmeasurable}. "
            "mAP over the declared class_map cannot be computed; request a dataset release "
            "with coverage for every class the model declares."
        )

    per_class: Dict[str, PerClassDetectionMetrics] = {}
    metric_dicts = []
    for cid in class_ids:
        res = compute_class_detection_metrics(predictions_by_sample, ground_truth_by_sample, cid)
        metric_dicts.append(res)
        per_class[cid] = PerClassDetectionMetrics(**res)

    map50 = mean_average_precision(metric_dicts, "ap50")
    map50_95 = mean_average_precision(metric_dicts, "ap50_95")

    test_section = TestEvaluationSection(
        protocol_id=TEST_PROTOCOL_ID,
        ground_truth="human_verified",
        sample_count=len(test_samples),
        gt_annotation_count=sum(len(v) for v in ground_truth_by_sample.values()),
        excluded_partial_count=excluded_partial_count,
        mAP50=map50,
        mAP50_95=map50_95,
        classes=per_class,
        inference_config=config,
    )

    # Parity that was never measured is recorded as an explicit failed section. There is
    # no "skipped" state: a package without parity evidence simply fails its gate.
    parity_section = export_parity or parity_not_measured(config)
    if parity_section.inference_config != config:
        raise TestEvaluationError(
            "export parity was measured under a different postprocess configuration than "
            "the test evaluation; publish separate measurements instead of mixing configurations"
        )

    gate = build_gate(gate_policy, test_section, parity_section, target_benchmarks or [])

    return EvaluationReport(
        schema_version="1.0.0",
        run_id=run_id,
        dataset=DatasetRef(
            dataset_id=ds_manifest.dataset_id,
            manifest_sha256=compute_dataset_manifest_sha(dataset_dir),
        ),
        test=test_section,
        export_parity=parity_section,
        target_benchmarks=target_benchmarks or [],
        gate=gate,
    )


def measure_target_benchmark(
    onnx_path: Path,
    inference_config: InferenceConfig,
    input_reference: str,
    requested_providers: Optional[List[str]] = None,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
    device: str = "cpu",
) -> TargetBenchmark:
    """Benchmark an exported model and bind the numbers to this machine and runtime.

    Latency without OS/architecture/runtime/device binding is not publishable evidence,
    and peak memory is a required deployment metric.
    """
    import onnxruntime as ort

    from vision_model_factory.export.quantization import (
        benchmark_inference_session,
        measure_peak_memory_bytes,
    )

    requested = requested_providers or ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=requested)
    active = list(session.get_providers())
    if active != requested:
        raise TestEvaluationError(
            f"Requested providers {requested} fell back to {active}; a benchmark (and any "
            "profile declaration) must describe the providers actually executing."
        )

    input_meta = session.get_inputs()[0]
    target_hw = inference_config.target_shape
    shape = [
        d if isinstance(d, int) and d > 0 else (1 if i == 0 else int(target_hw[i - 2]))
        for i, d in enumerate(input_meta.shape)
    ]
    rng = np.random.default_rng(42)
    dummy = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)

    timing = benchmark_inference_session(session, dummy, warmup_runs=warmup_runs, benchmark_runs=benchmark_runs)
    peak = measure_peak_memory_bytes(session, dummy)

    uname = platform.uname()
    return TargetBenchmark(
        os_name=uname.system,
        os_version=uname.version,
        architecture=uname.machine,
        device=device,
        runtime="onnxruntime",
        runtime_version=ort.__version__,
        provider=active[0],
        providers_used=active,
        requested_providers=requested,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        latency_ms_p50=timing["latency_ms_p50"],
        latency_ms_p95=timing["latency_ms_p95"],
        latency_ms_mean=timing["latency_ms_mean"],
        peak_memory_bytes=peak,
        input_reference=input_reference,
        inference_config=inference_config,
    )


class ModelPackagePredictor:
    """Run inference for evaluation using a published Model Package as the source of truth.

    Target geometry and postprocess thresholds come from the package manifest rather than
    code defaults, so measured numbers describe the artifact that will be published.
    Overrides are only accepted when they match the manifest exactly: an override that
    actually changed behaviour would silently evaluate a different model than the one
    under review.
    """

    def __init__(
        self,
        class_map: List[ClassMapItem],
        inference_config: InferenceConfig,
        model_path: Path,
        dataset_dir: Path,
    ):
        self.class_map = list(class_map)
        self.dataset_dir = Path(dataset_dir)
        self.config = inference_config
        self.model_path = Path(model_path)

    def _reject_override(self, name: str, value: Any) -> None:
        declared = self.descriptor()[name]
        if value is not None and value != declared:
            raise TestEvaluationError(
                f"{name}={value} overrides the package's declared {name}={declared}. Locked "
                "evaluation must measure the artifact as published; publish a new package "
                "with different postprocess thresholds instead."
            )

    def descriptor(self) -> Dict[str, Any]:
        """Declared input interface, for consumers that need the tensor spec."""
        return {
            "target_shape": self.config.target_shape,
            "score_threshold": self.config.score_threshold,
            "nms_iou_threshold": self.config.nms_iou_threshold,
            "max_detections": self.config.max_detections,
        }

    def _decode(self, output: np.ndarray, meta: Dict[str, Any], sample: SampleRecord):
        from vision_model_factory.export.decoder import yolo_xywh_scores_v1

        return yolo_xywh_scores_v1(
            output,
            class_map=self.class_map,
            orig_shape=(sample.height, sample.width),
            preprocess_meta=meta,
            score_threshold=self.config.score_threshold,
            nms_iou_threshold=self.config.nms_iou_threshold,
            max_detections=self.config.max_detections,
        )

    def _preprocess(self, sample: SampleRecord):
        from PIL import Image

        from vision_model_factory.export.preprocessor import letterbox_rgb_u8_v1

        img_path = self.dataset_dir / sample.file.path
        if not img_path.is_file():
            raise TestEvaluationError(
                f"Evaluation sample bytes missing from the package: {sample.sample_id} -> {img_path}"
            )
        with Image.open(img_path) as pil_img:
            img_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        target = (self.config.target_shape[0], self.config.target_shape[1])
        return letterbox_rgb_u8_v1(img_np, target_shape=target)

class OnnxPackagePredictor(ModelPackagePredictor):
    """ONNX Runtime predictor driven by the model package manifest."""

    def __init__(
        self,
        class_map: List[ClassMapItem],
        inference_config: InferenceConfig,
        model_path: Path,
        dataset_dir: Path,
        expected_input_dtype: str = "float32",
        providers: Optional[List[str]] = None,
        score_override: Optional[float] = None,
        nms_override: Optional[float] = None,
        max_detections_override: Optional[int] = None,
    ):
        super().__init__(class_map, inference_config, model_path, dataset_dir)
        # Overrides that would change behaviour are refused; only confirmations accepted.
        self._reject_override("score_threshold", score_override)
        self._reject_override("nms_iou_threshold", nms_override)
        self._reject_override("max_detections", max_detections_override)

        import onnxruntime as ort

        requested = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=requested)
        active = self.session.get_providers()
        if active != requested:
            raise TestEvaluationError(
                f"Requested providers {requested} fell back to {active}; evaluation must run on "
                "the profile the package declares."
            )
        session_dtype = self.session.get_inputs()[0].type
        if expected_input_dtype == "float32" and "tensor(float)" not in session_dtype:
            raise TestEvaluationError(
                f"The package declares input dtype float32 but the graph expects {session_dtype}"
            )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, sample: SampleRecord) -> List[Dict[str, Any]]:
        tensor, meta = self._preprocess(sample)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode(output, meta, sample)


class TorchPackagePredictor(ModelPackagePredictor):
    """PyTorch checkpoint predictor driven by the model package manifest.

    Used to obtain the reference side of export parity: the exported graph must reproduce
    this model's decoded detections on identical preprocessed pixels.
    """

    def __init__(
        self,
        class_map: List[ClassMapItem],
        inference_config: InferenceConfig,
        weights_path: Path,
        dataset_dir: Path,
    ):
        super().__init__(class_map, inference_config, weights_path, dataset_dir)
        from ultralytics import YOLO

        model = YOLO(str(self.model_path))
        self.model = model.model
        self.model.eval()

    def predict(self, sample: SampleRecord) -> List[Dict[str, Any]]:
        import torch

        tensor, meta = self._preprocess(sample)
        with torch.no_grad():
            output = self.model(torch.from_numpy(tensor))
            if isinstance(output, (list, tuple)):
                output = output[0]
            if isinstance(output, (list, tuple)):
                output = output[0]
        return self._decode(output.cpu().numpy(), meta, sample)


__all__ = [
    "TEST_PROTOCOL_ID",
    "ModelPackagePredictor",
    "OnnxPackagePredictor",
    "TestEvaluationError",
    "TorchPackagePredictor",
    "compute_dataset_manifest_sha",
    "evaluate_test_split",
    "measure_target_benchmark",
    "parity_not_measured",
]
