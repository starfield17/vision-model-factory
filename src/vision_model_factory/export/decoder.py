"""Reference YOLO output decoder implementation for yolo_xywh_scores_v1."""

from typing import Any, Dict, List, Tuple

import numpy as np

from vision_model_factory.contracts.models import ClassMapItem


def compute_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def nms_per_class(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    """
    Perform greedy non-maximum suppression for a single class.
    Args:
        boxes: (M, 4) array of [x1, y1, x2, y2]
        scores: (M,) array of confidence scores
        iou_threshold: float
    Returns:
        List of selected integer indices sorted by score descending.
    """
    if len(boxes) == 0:
        return []

    # Stable sort descending by score
    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []

    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break

        remaining_indices = order[1:]
        ious = np.array([compute_iou_xyxy(boxes[idx], boxes[rem]) for rem in remaining_indices])
        order = remaining_indices[ious < iou_threshold]

    return keep


def yolo_xywh_scores_v1(
    output_tensor: np.ndarray,
    class_map: List[ClassMapItem],
    orig_shape: Tuple[int, int],
    preprocess_meta: Dict[str, Any],
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> List[Dict[str, Any]]:
    """
    Decode yolo_xywh_scores_v1 model output into restored original coordinate detections.

    Args:
        output_tensor: Tensor of shape (1, 4 + K, N) or (4 + K, N)
        class_map: List of ClassMapItem mapping index to class_id
        orig_shape: (orig_h, orig_w)
        preprocess_meta: Metadata from letterbox_rgb_u8_v1 containing scale and pads
        score_threshold: Minimum confidence score [0, 1]
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections to return

    Returns:
        List of detection dictionaries:
        [{"class_id": str, "score": float, "bbox_xyxy": [x1, y1, x2, y2]}]
    """
    if output_tensor.ndim == 3:
        if output_tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {output_tensor.shape}")
        arr = output_tensor[0]
    elif output_tensor.ndim == 2:
        arr = output_tensor
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {output_tensor.shape}")

    num_classes = len(class_map)
    expected_channels = 4 + num_classes
    if arr.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels (4 + {num_classes}), got {arr.shape[0]}"
        )

    num_anchors = arr.shape[1]
    if num_anchors == 0:
        return []

    # Map class index to class_id
    index_to_class = {item.index: item.class_id for item in class_map}

    boxes_cxcywh = arr[:4, :]  # Shape: (4, N)
    class_scores = arr[4:, :]  # Shape: (K, N)

    # Highest score per anchor; tie broken by lowest index (numpy argmax behavior)
    best_class_indices = np.argmax(class_scores, axis=0)  # Shape: (N,)
    best_scores = class_scores[best_class_indices, np.arange(num_anchors)]  # Shape: (N,)

    # Score threshold filtering
    valid_mask = best_scores >= score_threshold
    if not np.any(valid_mask):
        return []

    valid_indices = np.where(valid_mask)[0]
    boxes_filtered = boxes_cxcywh[:, valid_indices]
    scores_filtered = best_scores[valid_indices]
    classes_filtered = best_class_indices[valid_indices]

    # Convert cx, cy, w, h -> x1, y1, x2, y2 in model input space
    cx = boxes_filtered[0, :]
    cy = boxes_filtered[1, :]
    w = boxes_filtered[2, :]
    h = boxes_filtered[3, :]

    x1_model = cx - w / 2.0
    y1_model = cy - h / 2.0
    x2_model = cx + w / 2.0
    y2_model = cy + h / 2.0

    # Invert letterbox
    scale = float(preprocess_meta["scale"])
    pad_left = float(preprocess_meta["pad_left"])
    pad_top = float(preprocess_meta["pad_top"])
    orig_h, orig_w = orig_shape

    x1_orig = (x1_model - pad_left) / scale
    y1_orig = (y1_model - pad_top) / scale
    x2_orig = (x2_model - pad_left) / scale
    y2_orig = (y2_model - pad_top) / scale

    # Clip to image boundary
    x1_clipped = np.clip(x1_orig, 0.0, float(orig_w))
    y1_clipped = np.clip(y1_orig, 0.0, float(orig_h))
    x2_clipped = np.clip(x2_orig, 0.0, float(orig_w))
    y2_clipped = np.clip(y2_orig, 0.0, float(orig_h))

    # Discard invalid boxes
    valid_box_mask = (x2_clipped > x1_clipped) & (y2_clipped > y1_clipped)
    if not np.any(valid_box_mask):
        return []

    x1_clipped = x1_clipped[valid_box_mask]
    y1_clipped = y1_clipped[valid_box_mask]
    x2_clipped = x2_clipped[valid_box_mask]
    y2_clipped = y2_clipped[valid_box_mask]
    scores_filtered = scores_filtered[valid_box_mask]
    classes_filtered = classes_filtered[valid_box_mask]

    boxes_xyxy = np.stack([x1_clipped, y1_clipped, x2_clipped, y2_clipped], axis=1)

    # Per-class NMS
    all_kept_detections: List[Dict[str, Any]] = []
    unique_classes = np.unique(classes_filtered)

    for c_idx in unique_classes:
        c_mask = classes_filtered == c_idx
        c_boxes = boxes_xyxy[c_mask]
        c_scores = scores_filtered[c_mask]

        kept_local_indices = nms_per_class(c_boxes, c_scores, nms_iou_threshold)
        class_id = index_to_class[int(c_idx)]

        for k_idx in kept_local_indices:
            box = c_boxes[k_idx]
            all_kept_detections.append(
                {
                    "class_id": class_id,
                    "score": round(float(c_scores[k_idx]), 4),
                    "bbox_xyxy": [
                        round(float(box[0]), 3),
                        round(float(box[1]), 3),
                        round(float(box[2]), 3),
                        round(float(box[3]), 3),
                    ],
                }
            )

    # Final sort descending by score
    all_kept_detections.sort(key=lambda d: d["score"], reverse=True)

    # Truncate to max_detections
    return all_kept_detections[:max_detections]
