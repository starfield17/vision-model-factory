"""Object detection evaluation metrics: IoU matching, Precision, Recall, mAP.

Matching is performed **per image**. Detection AP is defined over per-image one-to-one
matching between predictions and ground truth; pooling every prediction and every ground
truth box of a class into one flat list lets a box predicted on one image match a ground
truth box on a different image, which inflates mAP and can fabricate a perfect score out
of an unrelated false positive.
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from vision_model_factory.export.decoder import compute_iou_xyxy

DEFAULT_IOU_THRESHOLDS: Tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

Detection = Dict[str, Any]


def match_detections_to_ground_truth(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match predictions to ground truth inside a single image deterministically.

    Predictions are considered in descending score order and each ground truth box can be
    claimed by at most one prediction, so a duplicate detection of the same object is a
    false positive rather than a second true positive.

    Returns:
        (tp_array, fp_array) binary arrays of length len(pred_boxes).
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    if num_preds == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    if num_gts == 0:
        return np.zeros(num_preds, dtype=bool), np.ones(num_preds, dtype=bool)

    sorted_indices = np.argsort(-np.asarray(pred_scores, dtype=np.float64), kind="stable")
    tp = np.zeros(num_preds, dtype=bool)
    fp = np.zeros(num_preds, dtype=bool)
    gt_matched = np.zeros(num_gts, dtype=bool)

    for p_idx in sorted_indices:
        p_box = np.asarray(pred_boxes[p_idx], dtype=np.float64)
        best_iou = -1.0
        best_gt_idx = -1
        for g_idx in range(num_gts):
            if gt_matched[g_idx]:
                continue
            iou = compute_iou_xyxy(p_box, np.asarray(gt_boxes[g_idx], dtype=np.float64))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp[p_idx] = True
            gt_matched[best_gt_idx] = True
        else:
            fp[p_idx] = True

    return tp, fp


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using the standard 101-point interpolation."""
    if len(recall) == 0 or len(precision) == 0:
        return 0.0

    rec_thresholds = np.linspace(0.0, 1.0, 101)
    prec_interpolated: List[float] = []
    for r_th in rec_thresholds:
        prec_at_r = precision[recall >= r_th]
        prec_interpolated.append(float(np.max(prec_at_r)) if len(prec_at_r) > 0 else 0.0)
    return float(np.mean(prec_interpolated))


def _per_image_tp_fp(
    predictions_by_sample: Dict[str, List[Detection]],
    ground_truth_by_sample: Dict[str, List[Detection]],
    class_id: str,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten per-image matches into aligned (scores, tp) arrays in prediction order."""
    scores: List[float] = []
    tp_flags: List[bool] = []

    for sample_id in sorted(set(predictions_by_sample) | set(ground_truth_by_sample)):
        preds = [p for p in predictions_by_sample.get(sample_id, []) if p["class_id"] == class_id]
        gts = [g for g in ground_truth_by_sample.get(sample_id, []) if g["class_id"] == class_id]
        if not preds:
            continue
        tp, _fp = match_detections_to_ground_truth(
            [p["bbox_xyxy"] for p in preds], [float(p["score"]) for p in preds],
            [g["bbox_xyxy"] for g in gts], iou_threshold=iou_threshold,
        )
        scores.extend(float(p["score"]) for p in preds)
        tp_flags.extend(bool(t) for t in tp)

    return np.asarray(scores, dtype=np.float64), np.asarray(tp_flags, dtype=bool)


def compute_class_detection_metrics(
    predictions_by_sample: Dict[str, List[Detection]],
    ground_truth_by_sample: Dict[str, List[Detection]],
    class_id: str,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> Dict[str, Any]:
    """Compute AP50, AP50:95, precision and recall for one class across a split.

    Precision and recall are reported at the operating point actually used, i.e. over all
    predictions that survived the declared score threshold, at IoU 0.5.
    """
    gts_by_sample = {
        sid: [g for g in gts if g["class_id"] == class_id]
        for sid, gts in ground_truth_by_sample.items()
    }
    total_gts = sum(len(v) for v in gts_by_sample.values())
    total_preds = sum(
        len([p for p in preds if p["class_id"] == class_id]) for preds in predictions_by_sample.values()
    )

    if total_gts == 0:
        # Unreachable through evaluate_test_split, which refuses a class without ground
        # truth before reaching here. Kept as a defensive zero rather than a fabricated 1.0.
        return {
            "class_id": class_id,
            "ap50": 0.0,
            "ap50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "total_gt": 0,
            "total_pred": total_preds,
        }

    if total_preds == 0:
        return {
            "class_id": class_id,
            "ap50": 0.0,
            "ap50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "total_gt": total_gts,
            "total_pred": 0,
        }

    aps: List[float] = []
    ap50 = 0.0
    tp_count_at_50 = 0

    for iou_th in iou_thresholds:
        scores, tp = _per_image_tp_fp(predictions_by_sample, ground_truth_by_sample, class_id, iou_th)
        order = np.argsort(-scores, kind="stable")
        tp_sorted = tp[order]
        fp_sorted = ~tp_sorted

        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(fp_sorted)
        prec = cum_tp / (cum_tp + cum_fp)
        rec = cum_tp / total_gts

        ap = compute_ap(prec, rec)
        aps.append(ap)

        if abs(iou_th - 0.5) < 1e-9:
            ap50 = ap
            tp_count_at_50 = int(tp_sorted.sum())

    return {
        "class_id": class_id,
        "ap50": round(ap50, 4),
        "ap50_95": round(float(np.mean(aps)), 4),
        "precision": round(tp_count_at_50 / total_preds, 4),
        "recall": round(tp_count_at_50 / total_gts, 4),
        "total_gt": total_gts,
        "total_pred": total_preds,
    }


def mean_average_precision(per_class_metrics: Iterable[Dict[str, Any]], key: str) -> float:
    """Macro-average one metric over the evaluated classes.

    Every class handed in must have been measurable; `evaluate_test_split` enforces that
    precondition, so the headline mean never mixes measured and unmeasured classes.
    """
    values = [float(m[key]) for m in per_class_metrics]
    if not values:
        return 0.0
    return round(float(np.mean(values)), 4)


__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "compute_ap",
    "compute_class_detection_metrics",
    "match_detections_to_ground_truth",
    "mean_average_precision",
]
