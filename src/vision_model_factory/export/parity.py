"""Export parity verification comparing PyTorch reference to ONNX Runtime inference."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from vision_model_factory.contracts.models import ClassMapItem
from vision_model_factory.export.decoder import yolo_xywh_scores_v1


def check_export_parity(
    torch_model: nn.Module,
    onnx_path: Union[str, Path],
    class_map: List[ClassMapItem],
    test_input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    orig_shape: Tuple[int, int] = (720, 1280),
    preprocess_meta: Optional[Dict[str, Any]] = None,
    tensor_atol: float = 1e-2,
    score_atol: float = 1e-3,
    box_atol: float = 1.0,
) -> Dict[str, Any]:
    """
    Verify parity between PyTorch model and exported ONNX model on CPU.

    Returns:
        Dict with status, metrics, and parity evidence.
    """
    onnx_file = Path(onnx_path).resolve()
    if not onnx_file.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_file}")

    # Set up deterministic synthetic input
    np.random.seed(42)
    sample_np = np.random.uniform(0.0, 1.0, size=test_input_shape).astype(np.float32)
    sample_torch = torch.from_numpy(sample_np)

    # 1. PyTorch inference
    torch_model.eval()
    with torch.no_grad():
        ref_out = torch_model(sample_torch)
        if isinstance(ref_out, (list, tuple)):
            ref_out = ref_out[0]
        ref_out_np = ref_out.cpu().numpy()

    # 2. ONNX Runtime inference
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_file),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    ort_outs = session.run(None, {input_name: sample_np})
    ort_out_np = ort_outs[0]

    # Compare raw tensor outputs
    tensor_diff = np.abs(ref_out_np - ort_out_np)
    max_tensor_diff = float(np.max(tensor_diff))
    mean_tensor_diff = float(np.mean(tensor_diff))

    tensor_passed = max_tensor_diff <= tensor_atol

    if preprocess_meta is None:
        preprocess_meta = {
            "scale": 0.5,
            "pad_left": 0.0,
            "pad_top": 140.0,
        }

    # 3. Decode both and compare detection lists
    ref_detections = yolo_xywh_scores_v1(
        ref_out_np,
        class_map=class_map,
        orig_shape=orig_shape,
        preprocess_meta=preprocess_meta,
        score_threshold=0.1,
    )
    ort_detections = yolo_xywh_scores_v1(
        ort_out_np,
        class_map=class_map,
        orig_shape=orig_shape,
        preprocess_meta=preprocess_meta,
        score_threshold=0.1,
    )

    detections_matched = len(ref_detections) == len(ort_detections)
    max_box_diff = 0.0
    max_score_diff = 0.0

    if detections_matched:
        for d_ref, d_ort in zip(ref_detections, ort_detections):
            if d_ref["class_id"] != d_ort["class_id"]:
                detections_matched = False
                break
            score_diff = abs(d_ref["score"] - d_ort["score"])
            max_score_diff = max(max_score_diff, score_diff)
            box_diff = max(
                abs(c_ref - c_ort)
                for c_ref, c_ort in zip(d_ref["bbox_xyxy"], d_ort["bbox_xyxy"])
            )
            max_box_diff = max(max_box_diff, box_diff)

            if score_diff > score_atol or box_diff > box_atol:
                detections_matched = False
                break

    passed = tensor_passed and detections_matched

    return {
        "status": "passed" if passed else "failed",
        "method": "pytorch_vs_onnxruntime_cpu",
        "tolerances": {
            "tensor_atol": tensor_atol,
            "score_atol": score_atol,
            "box_atol": box_atol,
        },
        "raw_tensor": {
            "max_abs_diff": max_tensor_diff,
            "mean_abs_diff": mean_tensor_diff,
            "passed": tensor_passed,
        },
        "detections": {
            "ref_count": len(ref_detections),
            "ort_count": len(ort_detections),
            "matched": detections_matched,
            "max_score_diff": max_score_diff,
            "max_box_diff": max_box_diff,
        },
    }
