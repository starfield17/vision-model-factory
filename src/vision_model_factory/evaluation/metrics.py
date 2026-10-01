"""Object detection evaluation metrics: IoU matching, Precision, Recall, mAP."""

from typing import Any, Dict, List, Tuple

import numpy as np

from vision_model_factory.export.decoder import compute_iou_xyxy


def match_detections_to_ground_truth(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match predictions to ground truth bounding boxes deterministically.
    Each ground truth box can match at most one prediction (highest score).

    Returns:
        (tp_array, fp_array) binary arrays of length len(pred_boxes).
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    if num_preds == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    if num_gts == 0:
        return np.zeros(num_preds, dtype=bool), np.ones(num_preds, dtype=bool)

    # Sort predictions by score descending
    sorted_indices = np.argsort(-np.array(pred_scores), kind="stable")
    tp = np.zeros(num_preds, dtype=bool)
    fp = np.zeros(num_preds, dtype=bool)
    gt_matched = np.zeros(num_gts, dtype=bool)

    for p_idx in sorted_indices:
        p_box = np.array(pred_boxes[p_idx])
        best_iou = -1.0
        best_gt_idx = -1

        for g_idx in range(num_gts):
            if gt_matched[g_idx]:
                continue
            g_box = np.array(gt_boxes[g_idx])
            iou = compute_iou_xyxy(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[p_idx] = True
            gt_matched[best_gt_idx] = True
        else:
            fp[p_idx] = True

    return tp, fp


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using standard 101-point interpolation."""
    if len(recall) == 0 or len(precision) == 0:
        return 0.0

    # 101-point interpolation
    rec_thresholds = np.linspace(0.0, 1.0, 101)
    prec_interpolated = []

    for r_th in rec_thresholds:
        prec_at_r = precision[recall >= r_th]
        if len(prec_at_r) > 0:
            prec_interpolated.append(float(np.max(prec_at_r)))
        else:
            prec_interpolated.append(0.0)

    return float(np.mean(prec_interpolated))


def compute_class_detection_metrics(
    all_preds: List[Dict[str, Any]],
    all_gts: List[Dict[str, Any]],
    class_id: str,
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
) -> Dict[str, Any]:
    """
    Compute AP50, AP50-95, precision, recall for a specific class across all images.
    """
    # Filter preds and gts for class_id
    preds_for_class = [p for p in all_preds if p["class_id"] == class_id]
    gts_for_class = [g for g in all_gts if g["class_id"] == class_id]

    total_gts = len(gts_for_class)
    total_preds = len(preds_for_class)

    if total_gts == 0:
        return {
            "class_id": class_id,
            "ap50": 0.0,
            "ap50_95": 0.0,
            "precision": 0.0 if total_preds > 0 else 1.0,
            "recall": 1.0 if total_preds == 0 else 0.0,
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

    pred_boxes = [p["bbox_xyxy"] for p in preds_for_class]
    pred_scores = [p["score"] for p in preds_for_class]
    gt_boxes = [g["bbox_xyxy"] for g in gts_for_class]

    # Compute AP across multiple IoU thresholds
    aps: List[float] = []
    ap50 = 0.0
    p50 = 0.0
    r50 = 0.0

    for iou_th in iou_thresholds:
        tp, fp = match_detections_to_ground_truth(pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_th)
        order = np.argsort(-np.array(pred_scores), kind="stable")
        tp_sorted = tp[order]
        fp_sorted = fp[order]

        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(fp_sorted)

        prec = cum_tp / (cum_tp + cum_fp)
        rec = cum_tp / total_gts

        ap = compute_ap(prec, rec)
        aps.append(ap)

        if abs(iou_th - 0.5) < 1e-5:
            ap50 = ap
            p50 = float(prec[-1]) if len(prec) > 0 else 0.0
            r50 = float(rec[-1]) if len(rec) > 0 else 0.0

    return {
        "class_id": class_id,
        "ap50": round(ap50, 4),
        "ap50_95": round(float(np.mean(aps)), 4),
        "precision": round(p50, 4),
        "recall": round(r50, 4),
        "total_gt": total_gts,
        "total_pred": total_preds,
    }
