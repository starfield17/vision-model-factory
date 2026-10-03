"""Tests for evaluation metrics and locked test split evaluation."""

from pathlib import Path

import numpy as np
import pytest
from factories import DEFAULT_CONFIG, make_gate_policy, make_parity_section

from vision_model_factory.contracts.models import ClassMapItem, SampleRecord
from vision_model_factory.evaluation.evaluator import (
    TestEvaluationError,
    evaluate_test_split,
    measure_target_benchmark,
    parity_not_measured,
)
from vision_model_factory.evaluation.metrics import (
    compute_ap,
    compute_class_detection_metrics,
    match_detections_to_ground_truth,
)


def test_match_detections_to_ground_truth():
    gt_boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 100.0, 100.0],
    ]
    # Prediction 1: perfect match with gt 0, score 0.9
    # Prediction 2: duplicate match with gt 0, score 0.8 -> should be marked FP (already matched)
    # Prediction 3: partial match with gt 1 (IoU > 0.5), score 0.7 -> marked TP
    # Prediction 4: background false positive, score 0.6 -> marked FP
    pred_boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [52.0, 52.0, 100.0, 100.0],
        [200.0, 200.0, 300.0, 300.0],
    ]
    pred_scores = [0.9, 0.8, 0.7, 0.6]

    tp, fp = match_detections_to_ground_truth(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

    assert tp[0] is np.True_  # matches gt 0
    assert fp[1] is np.True_  # gt 0 already claimed
    assert tp[2] is np.True_  # matches gt 1
    assert fp[3] is np.True_  # background


def test_compute_ap():
    # Perfect detector: precision=1.0 at all recall points
    prec = np.array([1.0, 1.0, 1.0])
    rec = np.array([0.33, 0.66, 1.0])
    ap = compute_ap(prec, rec)
    assert abs(ap - 1.0) < 1e-4


def test_metrics_are_per_image_not_pooled():
    """A box that matches a *different image's* ground truth must not count as a hit.

    This is the regression for the pooled-matching bug, which scored a detector that put
    one perfect box in image B while image A held the only ground truth as a flawless
    1.0. Matching must reset per image, so the same numbers here score zero.
    """
    box = [10.0, 10.0, 50.0, 50.0]

    # GT lives in image A only; the model's only prediction is in image B at the same
    # coordinates and the same class.
    predictions = {"img-A": [], "img-B": [{"class_id": "bottle", "score": 0.9, "bbox_xyxy": box}]}
    ground_truth = {
        "img-A": [{"class_id": "bottle", "bbox_xyxy": box}],
        "img-B": [],
    }

    res = compute_class_detection_metrics(predictions, ground_truth, "bottle")

    assert res["total_gt"] == 1
    assert res["total_pred"] == 1
    assert res["ap50"] == 0.0
    assert res["ap50_95"] == 0.0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0

    # Control: the identical box inside the image that actually owns the ground truth is
    # a perfect detection. Without this, the assertion above could pass for the wrong
    # reason (e.g. a scorer that always returns zero).
    same_image = {
        "img-A": [{"class_id": "bottle", "score": 0.9, "bbox_xyxy": box}],
        "img-B": [],
    }
    control = compute_class_detection_metrics(same_image, ground_truth, "bottle")
    assert control["ap50"] == 1.0
    assert control["recall"] == 1.0


def test_spurious_boxes_in_other_images_penalise_precision():
    """Predictions in images with no ground truth are false positives, not invisible.

    The spurious box is scored *above* the true one on purpose. Under the old flat
    matching it would claim the other image's ground truth first and the class would score
    a flawless 1.0; per-image matching makes it a false positive at rank 1, which halves
    the average precision.
    """
    box = [0.0, 0.0, 20.0, 20.0]
    ground_truth = {"img-A": [{"class_id": "bottle", "bbox_xyxy": box}], "img-B": []}
    predictions = {
        "img-A": [{"class_id": "bottle", "score": 0.90, "bbox_xyxy": box}],
        "img-B": [{"class_id": "bottle", "score": 0.95, "bbox_xyxy": box}],
    }

    res = compute_class_detection_metrics(predictions, ground_truth, "bottle")

    assert res["total_pred"] == 2
    assert res["precision"] == 0.5
    assert res["recall"] == 1.0
    assert res["ap50"] < 1.0


def test_evaluate_test_split(synthetic_dataset_dir: Path):
    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]

    # Predict function returning ground truth prediction for sample s-003
    def mock_predict(sample: SampleRecord):
        if sample.sample_id == "s-003":
            return [
                {
                    "class_id": "bottle",
                    "score": 0.95,
                    "bbox_xyxy": [80.0, 90.0, 220.0, 310.0],
                }
            ]
        return []

    # Run locked test split evaluation
    policy = make_gate_policy(require_export_parity_passed=False)
    report = evaluate_test_split(
        dataset_dir=synthetic_dataset_dir,
        predict_fn=mock_predict,
        class_map=class_map,
        inference_config=DEFAULT_CONFIG,
        run_id="run-test-eval-01",
        gate_policy=policy,
        export_parity=parity_not_measured(DEFAULT_CONFIG),
    )

    assert report.schema_version == "1.0.0"
    # The fixture carries two test-split images so that both declared classes have
    # locked-test ground truth.
    assert report.test.sample_count == 2
    assert report.test.gt_annotation_count == 2
    # Only s-003's bottle is predicted; the can in s-004 is missed, so the macro mean
    # sits at roughly 0.5 rather than a flawless 1.0.
    assert 0.4 < report.test.mAP50 < 0.6
    # The one predicted box coincides exactly with its ground truth, so bottle's AP is 1.0
    # at every IoU threshold and the two headline means coincide here.
    assert report.test.mAP50_95 <= report.test.mAP50
    assert report.test.classes["bottle"].recall == 1.0
    assert report.test.classes["can"].recall == 0.0
    # Parity was never measured and this policy does not require it, so the verdict is
    # derived from the quality floors alone.
    assert report.gate.status == "passed"
    assert report.gate.policy_sha256 != "0" * 64
    assert [c.passed for c in report.gate.checks] == [True, True]


def test_parity_requirement_makes_unmeasured_parity_fail(synthetic_dataset_dir: Path):
    """The same measurements, the same numbers, a policy that demands parity -> failed."""
    class_map = [ClassMapItem(index=0, class_id="bottle"), ClassMapItem(index=1, class_id="can")]

    def mock_predict(sample: SampleRecord):
        return []

    report = evaluate_test_split(
        dataset_dir=synthetic_dataset_dir,
        predict_fn=mock_predict,
        class_map=class_map,
        inference_config=DEFAULT_CONFIG,
        run_id="run-test-eval-02",
        gate_policy=make_gate_policy(min_map50=0.0, min_map50_95=0.0, require_export_parity_passed=True),
        export_parity=parity_not_measured(DEFAULT_CONFIG),
    )

    assert report.gate.status == "failed"
    assert "export_parity_passed" in [c.metric for c in report.gate.checks]


def test_parity_measured_under_other_config_is_refused(synthetic_dataset_dir: Path):
    """Parity under a different postprocess config is a different measurement."""
    class_map = [ClassMapItem(index=0, class_id="bottle"), ClassMapItem(index=1, class_id="can")]
    other = DEFAULT_CONFIG.model_copy(update={"score_threshold": 0.4})

    with pytest.raises(TestEvaluationError, match="different postprocess configuration"):
        evaluate_test_split(
            dataset_dir=synthetic_dataset_dir,
            predict_fn=lambda s: [],
            class_map=class_map,
            inference_config=DEFAULT_CONFIG,
            run_id="run-test-eval-03",
            gate_policy=make_gate_policy(),
            export_parity=make_parity_section(inference_config=other),
        )


def test_unmeasurable_class_aborts_evaluation(synthetic_dataset_dir: Path):
    """A declared class with no locked-test ground truth cannot be averaged in."""
    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
        ClassMapItem(index=2, class_id="unsupervised_class"),
    ]

    with pytest.raises(TestEvaluationError, match="no ground truth for class"):
        evaluate_test_split(
            dataset_dir=synthetic_dataset_dir,
            predict_fn=lambda s: [],
            class_map=class_map,
            inference_config=DEFAULT_CONFIG,
            run_id="run-test-eval-04",
            gate_policy=make_gate_policy(),
        )


def test_critical_class_floor_drives_the_gate(synthetic_dataset_dir: Path):
    """A per-class recall floor is enforced even when headline mAP clears its bar."""
    from vision_model_factory.contracts.gate_policy import ClassQualityFloor

    class_map = [ClassMapItem(index=0, class_id="bottle"), ClassMapItem(index=1, class_id="can")]
    policy = make_gate_policy(
        min_map50=0.0,
        min_map50_95=0.0,
        require_export_parity_passed=False,
        critical_classes=[ClassQualityFloor(class_id="can", min_precision=0.5, min_recall=0.5)],
    )

    report = evaluate_test_split(
        dataset_dir=synthetic_dataset_dir,
        predict_fn=lambda s: [],
        class_map=class_map,
        inference_config=DEFAULT_CONFIG,
        run_id="run-test-eval-05",
        gate_policy=policy,
    )

    assert report.gate.status == "failed"
    failed = {(c.metric, c.class_id) for c in report.gate.checks if not c.passed}
    assert ("recall", "can") in failed


def test_target_benchmark_is_bound_to_the_measuring_environment(tmp_path: Path):
    """Latency without OS/arch/runtime/device binding is not publishable evidence."""
    from vision_model_factory.export.exporter import export_torch_model_to_onnx
    from vision_model_factory.trainers.yolo import TinyYoloMockNet

    onnx_path, _ = export_torch_model_to_onnx(TinyYoloMockNet(num_classes=2, num_anchors=5),
                                             tmp_path / "bench.onnx")
    bench = measure_target_benchmark(
        onnx_path=onnx_path,
        inference_config=DEFAULT_CONFIG,
        input_reference=f"random:seed42:{onnx_path.name}",
        warmup_runs=1,
        benchmark_runs=3,
    )

    # Every field the deployment decision depends on is present and non-empty.
    for field in ("os_name", "os_version", "architecture", "device", "runtime",
                  "runtime_version", "provider", "input_reference"):
        value = getattr(bench, field)
        assert isinstance(value, str) and value.strip(), f"{field} missing from benchmark"
    assert bench.runtime == "onnxruntime"
    assert bench.provider == "CPUExecutionProvider"
    assert bench.providers_used == bench.requested_providers, (
        "a provider fallback must never be published as the measured profile"
    )
    assert bench.benchmark_runs == 3 and bench.warmup_runs == 1
    assert bench.latency_ms_p95 >= bench.latency_ms_p50 > 0.0
    assert bench.peak_memory_bytes > 0
    assert bench.inference_config == DEFAULT_CONFIG


def test_benchmark_refuses_a_provider_fallback(tmp_path: Path):
    """Asking for a GPU provider and silently getting CPU must fail, not report CPU as GPU."""
    from vision_model_factory.export.exporter import export_torch_model_to_onnx
    from vision_model_factory.trainers.yolo import TinyYoloMockNet

    onnx_path, _ = export_torch_model_to_onnx(TinyYoloMockNet(num_classes=2, num_anchors=5),
                                             tmp_path / "fallback.onnx")
    with pytest.raises(TestEvaluationError, match="fell back"):
        measure_target_benchmark(
            onnx_path=onnx_path,
            inference_config=DEFAULT_CONFIG,
            input_reference="probe",
            requested_providers=["CUDAExecutionProvider"],
            warmup_runs=0,
            benchmark_runs=1,
        )


def test_graph_inspection_refuses_a_graph_it_cannot_describe(tmp_path: Path):
    """One input, one output: anything else cannot use the single-input contract."""
    import onnx
    from onnx import helper, numpy_helper

    from vision_model_factory.export.exporter import export_torch_model_to_onnx
    from vision_model_factory.export.graph_inspection import (
        GraphInspectionError,
        describe_onnx_graph,
        graph_uses_integer_quantization,
        infer_precision,
    )
    from vision_model_factory.trainers.yolo import TinyYoloMockNet

    good, _ = export_torch_model_to_onnx(TinyYoloMockNet(num_classes=2, num_anchors=5),
                                        tmp_path / "good.onnx")
    described = describe_onnx_graph(good)
    assert described["input"]["name"] == "images"
    assert described["output"]["shape"] == [1, 6, 5]
    assert described["quantized"] is False
    assert infer_precision(good) == "fp32"

    # Two inputs.
    two_in = tmp_path / "two_inputs.onnx"
    node = helper.make_node("Add", ["a", "b"], ["c"])
    graph = helper.make_graph(
        [node], "two", [helper.make_tensor_value_info("a", 1, [1]),
                        helper.make_tensor_value_info("b", 1, [1])],
        [helper.make_tensor_value_info("c", 1, [1])],
    )
    two_in.write_bytes(onnx.ModelProto(graph=graph, ir_version=9,
                                       opset_import=[helper.make_opsetid("", 17)]).SerializeToString())
    with pytest.raises(GraphInspectionError, match="exactly one non-initializer graph input"):
        describe_onnx_graph(two_in)

    # Quantized detection: an initializer of int8 element type is an integer graph.
    quant = tmp_path / "quant.onnx"
    init = numpy_helper.from_array(np.zeros((1,), dtype=np.int8), "w")
    g2 = helper.make_graph([node], "q", [helper.make_tensor_value_info("a", 1, [1])],
                           [helper.make_tensor_value_info("c", 1, [1])], initializer=[init])
    quant.write_bytes(onnx.ModelProto(graph=g2, ir_version=9,
                                      opset_import=[helper.make_opsetid("", 17)]).SerializeToString())
    assert graph_uses_integer_quantization(onnx.load(str(quant))) is True
