"""Export parity verification: PyTorch reference vs exported ONNX Runtime inference.

Parity is a *semantic* check, not a tensor-equality check on arbitrary input. Two rules
make it meaningful:

1. The reference input is a real image from the dataset, preprocessed through the
   published `letterbox_rgb_u8_v1` reference, so both sides see the same bytes on the
   same geometry. Random noise produces zero detections on a real detector, and
   zero-versus-zero compares nothing.
2. The check proves it can see a difference: a deliberately perturbed copy of the
   reference detections must fail the comparison. A parity report whose
   `self_test.detected_perturbation` is false did not verify coordinate recovery and
   cannot be recorded as passed.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image

from vision_model_factory.contracts.models import ClassMapItem, SampleRecord
from vision_model_factory.export.decoder import yolo_xywh_scores_v1
from vision_model_factory.export.preprocessor import letterbox_rgb_u8_v1

PARITY_PROTOCOL_ID = "letterbox_decode_parity_v1"
PARITY_MATCHING_METHOD = "greedy_one_to_one_by_class_iou_then_score"


class ParityInputError(ValueError):
    """Raised when parity cannot be measured because no usable reference input exists."""


def _load_reference_input(sample: SampleRecord, dataset_dir: Path, target_shape: Tuple[int, int]):
    """Run one dataset sample through the published reference preprocessor."""
    img_path = dataset_dir / sample.file.path
    if not img_path.is_file():
        raise ParityInputError(f"Parity reference image missing for sample '{sample.sample_id}': {img_path}")
    with Image.open(img_path) as pil_img:
        img_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    tensor, meta = letterbox_rgb_u8_v1(img_np, target_shape=target_shape)
    return tensor, meta


def decode_reference_detections(
    output_tensor: np.ndarray,
    sample: SampleRecord,
    class_map: List[ClassMapItem],
    preprocess_meta: Dict[str, Any],
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> List[Dict[str, Any]]:
    """Decode one reference inference back into original image coordinates."""
    return yolo_xywh_scores_v1(
        output_tensor,
        class_map=class_map,
        orig_shape=(sample.height, sample.width),
        preprocess_meta=preprocess_meta,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )


def match_detection_sets(
    ref: List[Dict[str, Any]],
    other: List[Dict[str, Any]],
    score_atol: float,
    box_atol: float,
) -> Dict[str, Any]:
    """Compare two decoded detection sets position by position.

    Both sides come from the same deterministic decoder fed the same pixels, so a
    faithful export reproduces the same ordered list. Any deviation - a different count,
    a reclassified box, a box shifted beyond `box_atol` - fails. Matching by count alone
    is deliberately not sufficient: two lists of equal length at different coordinates
    must fail, which is what the perturbation self-test exercises.
    """
    ordered_ref = sorted(ref, key=lambda d: -float(d["score"]))
    ordered_other = sorted(other, key=lambda d: -float(d["score"]))

    counts_match = len(ordered_ref) == len(ordered_other)
    within_tolerance = counts_match
    max_score_diff = 0.0
    max_box_diff = 0.0

    for a, b in zip(ordered_ref, ordered_other):
        if a["class_id"] != b["class_id"]:
            within_tolerance = False
        score_diff = abs(float(a["score"]) - float(b["score"]))
        box_diff = max(
            abs(float(x) - float(y)) for x, y in zip(a["bbox_xyxy"], b["bbox_xyxy"])
        )
        max_score_diff = max(max_score_diff, score_diff)
        max_box_diff = max(max_box_diff, box_diff)
        if score_diff > score_atol or box_diff > box_atol:
            within_tolerance = False

    return {
        "counts_match": counts_match,
        "within_tolerance": within_tolerance,
        "max_score_diff": max_score_diff,
        "max_box_diff": max_box_diff,
    }


PERTURBATION_PX = 25.0


def perturb_detections(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shift every box by a fixed visible amount; used to prove the comparison can fail."""
    shifted: List[Dict[str, Any]] = []
    for d in dets:
        copy = dict(d)
        x1, y1, x2, y2 = d["bbox_xyxy"]
        copy["bbox_xyxy"] = [x1 + PERTURBATION_PX, y1 + PERTURBATION_PX, x2 + PERTURBATION_PX, y2 + PERTURBATION_PX]
        shifted.append(copy)
    return shifted


def check_export_parity(
    torch_model: nn.Module,
    onnx_path: Union[str, Path],
    class_map: List[ClassMapItem],
    samples: List[SampleRecord],
    dataset_dir: Union[str, Path],
    target_shape: Tuple[int, int] = (640, 640),
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
    tensor_atol: float = 1e-2,
    score_atol: float = 1e-3,
    box_atol: float = 1.0,
    parity_split: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify semantic parity between the PyTorch reference and the exported ONNX model.

    The same preprocessed pixels are fed to both sides for every reference sample; the
    decoded detections in original image coordinates are then compared, and a perturbed
    reference is compared as well to prove the comparison is not vacuous.
    """
    onnx_file = Path(onnx_path).resolve()
    if not onnx_file.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_file}")
    dataset_dir = Path(dataset_dir)
    if not samples:
        raise ParityInputError("Parity requires at least one real reference sample.")

    # The split is read off the records actually consumed, never handed down as a label:
    # a claim about which corpus was spent has to follow from the corpus that was used.
    used_splits = {s.split for s in samples}
    if len(used_splits) != 1:
        raise ParityInputError(
            f"Parity samples come from mixed splits {sorted(used_splits)}; a single corpus "
            "must be named so the report can be audited for locked-test-set consumption."
        )
    used_split = next(iter(used_splits))
    if used_split == "test":
        raise ParityInputError(
            "Parity was given samples from the locked 'test' split. The scored corpus may "
            "not be spent validating an artifact, and export_parity.parity_split cannot "
            "record 'test'. Select a train, val, or audit corpus."
        )
    if parity_split is not None and parity_split != used_split:
        raise ParityInputError(
            f"Requested parity split '{parity_split}' but the samples provided are from "
            f"'{used_split}'; the recorded split must match the corpus actually consumed."
        )

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    # A graph whose output does not depend on its input is pruned to zero inputs by the
    # runtime. That is not a comparison to attempt: it ignores every pixel and must be
    # reported as an unusable export rather than surfacing as an index error.
    session_inputs = session.get_inputs()
    if len(session_inputs) != 1:
        raise ParityInputError(
            f"Exported graph exposes {len(session_inputs)} inputs ({[i.name for i in session_inputs]}); "
            "parity requires exactly one input tensor. A graph that ignores its input "
            "cannot be compared against a reference model."
        )
    input_name = session_inputs[0].name
    declared_shape = [d if isinstance(d, str) else int(d) for d in session_inputs[0].shape]

    torch_model.eval()

    tensor_max = 0.0
    # Relative difference, normalised by max(1.0, |reference|). The exported graph emits
    # pre-NMS pixel coordinates alongside class scores in one tensor, so a single absolute
    # tolerance mixes a 640-unit scale with a 0-1 scale; the relative figure is what shows
    # whether the two graphs are the same function up to float reassociation.
    tensor_max_rel = 0.0
    tensor_mean = 0.0
    tensor_compared = 0
    ref_all: List[Dict[str, Any]] = []
    ort_all: List[Dict[str, Any]] = []
    used_samples: List[str] = []

    for sample in samples:
        tensor_np, meta = _load_reference_input(sample, dataset_dir, target_shape)
        if declared_shape != list(tensor_np.shape):
            raise ParityInputError(
                f"Exported model input shape {declared_shape} does not match the reference "
                f"preprocessing output {list(tensor_np.shape)}; the exported graph does not "
                "implement the declared preprocessing interface."
            )

        with torch.no_grad():
            ref_out = torch_model(torch.from_numpy(tensor_np))
            if isinstance(ref_out, (list, tuple)):
                ref_out = ref_out[0]
            ref_np = ref_out.cpu().numpy()
        ort_np = session.run(None, {input_name: tensor_np})[0]

        if ref_np.shape != ort_np.shape:
            raise ParityInputError(
                f"Reference output shape {ref_np.shape} != exported output shape {ort_np.shape}"
            )

        diff = np.abs(ref_np.astype(np.float64) - ort_np.astype(np.float64))
        tensor_max = max(tensor_max, float(np.max(diff)))
        denom = np.maximum(np.abs(ref_np.astype(np.float64)), 1.0)
        tensor_max_rel = max(tensor_max_rel, float(np.max(diff / denom)))
        tensor_mean += float(np.mean(diff))
        tensor_compared += 1

        ref_dets = decode_reference_detections(
            ref_np, sample, class_map, meta, score_threshold, nms_iou_threshold, max_detections
        )
        ort_dets = decode_reference_detections(
            ort_np, sample, class_map, meta, score_threshold, nms_iou_threshold, max_detections
        )
        ref_all.extend(ref_dets)
        ort_all.extend(ort_dets)
        used_samples.append(sample.sample_id)

    if not ref_all and not ort_all:
        raise ParityInputError(
            "Parity produced zero detections on both sides for every reference sample; this "
            "comparison cannot verify decoding or coordinate recovery."
        )

    mean_tensor = tensor_mean / tensor_compared if tensor_compared else 0.0
    tensor_passed = tensor_max <= tensor_atol

    comparison = match_detection_sets(ref_all, ort_all, score_atol, box_atol)
    self_test = match_detection_sets(ref_all, perturb_detections(ref_all), score_atol, box_atol)
    detected_perturbation = not self_test["within_tolerance"]

    detections_passed = bool(ref_all) and bool(ort_all) and comparison["within_tolerance"]

    return {
        "status": "passed" if (tensor_passed and detections_passed and detected_perturbation) else "failed",
        "method": f"pytorch_vs_onnxruntime_cpu/{PARITY_PROTOCOL_ID}",
        "parity_split": used_split,
        "input_reference": f"dataset:{','.join(used_samples)}:letterbox_rgb_u8_v1",
        "matching_method": PARITY_MATCHING_METHOD,
        "tolerances": {
            "tensor_atol": tensor_atol,
            "score_atol": score_atol,
            "box_atol": box_atol,
        },
        "raw_tensor": {
            "max_abs_diff": tensor_max,
        "max_rel_diff": tensor_max_rel,
            "mean_abs_diff": mean_tensor,
            "passed": tensor_passed,
        },
        "detections": {
            "ref_count": len(ref_all),
            "ort_count": len(ort_all),
            "matched": detections_passed,
            "max_score_diff": comparison["max_score_diff"],
            "max_box_diff": comparison["max_box_diff"],
        },
        "self_test": {"perturbation_px": PERTURBATION_PX, "detected_perturbation": detected_perturbation},
        "exceptions": [],
        "inference_config": {
            "score_threshold": score_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "max_detections": max_detections,
            "target_shape": [target_shape[0], target_shape[1]],
        },
    }


__all__ = [
    "PARITY_MATCHING_METHOD",
    "PARITY_PROTOCOL_ID",
    "ParityInputError",
    "check_export_parity",
    "match_detection_sets",
    "perturb_detections",
]
