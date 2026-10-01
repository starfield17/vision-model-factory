"""Tests for reference letterbox preprocessor and YOLO output decoder."""

import numpy as np

from vision_model_factory.contracts.models import ClassMapItem
from vision_model_factory.export.decoder import compute_iou_xyxy, yolo_xywh_scores_v1
from vision_model_factory.export.preprocessor import letterbox_rgb_u8_v1


def test_letterbox_rgb_u8_v1():
    # Input image 400x800 (H=400, W=800) -> aspect ratio 2:1
    img = np.full((400, 800, 3), 200, dtype=np.uint8)
    target_shape = (640, 640)

    tensor, meta = letterbox_rgb_u8_v1(img, target_shape=target_shape, pad_value=114)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32

    # Scale: min(640/400, 640/800) = 640/800 = 0.8
    assert meta["scale"] == 0.8
    # New dimensions: new_w = floor(800 * 0.8 + 0.5) = 640, new_h = floor(400 * 0.8 + 0.5) = 320
    assert meta["new_shape"] == (320, 640)
    # Pad width: 640 - 640 = 0 -> pad_left = 0, pad_right = 0
    assert meta["pad_left"] == 0
    assert meta["pad_right"] == 0
    # Pad height: 640 - 320 = 320 -> pad_top = 160, pad_bottom = 160
    assert meta["pad_top"] == 160
    assert meta["pad_bottom"] == 160

    # Check that padded top region contains pad_value (114/255.0)
    expected_pad_val = 114.0 / 255.0
    np.testing.assert_allclose(tensor[0, :, :160, :], expected_pad_val, atol=1e-5)

    # Check that image region contains 200/255.0
    expected_img_val = 200.0 / 255.0
    np.testing.assert_allclose(tensor[0, :, 160:480, :], expected_img_val, atol=1e-5)


def test_compute_iou_xyxy():
    box1 = np.array([0.0, 0.0, 10.0, 10.0])
    box2 = np.array([0.0, 0.0, 10.0, 10.0])
    assert compute_iou_xyxy(box1, box2) == 1.0

    box3 = np.array([10.0, 10.0, 20.0, 20.0])
    assert compute_iou_xyxy(box1, box3) == 0.0

    box4 = np.array([5.0, 0.0, 15.0, 10.0])
    # Intersection: [5, 0, 10, 10] -> w=5, h=10, area=50. Union = 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
    assert abs(compute_iou_xyxy(box1, box4) - (1.0 / 3.0)) < 1e-5


def test_yolo_xywh_scores_v1_decoding():
    # 2 classes: bottle (0), can (1)
    class_map = [
        ClassMapItem(index=0, class_id="bottle"),
        ClassMapItem(index=1, class_id="can"),
    ]

    # Model input target 640x640, original image 400x800
    orig_shape = (400, 800)
    preprocess_meta = {
        "scale": 0.8,
        "pad_left": 0.0,
        "pad_top": 160.0,
    }

    # Construct output tensor: shape (1, 6, 2) -> 2 candidate anchors
    # Anchor 0: cx=320, cy=320, w=160, h=160 (in model space). Bottle score=0.9, Can score=0.1
    # Anchor 1: duplicate box with lower score for NMS suppression test (Bottle score=0.8)
    output = np.zeros((1, 6, 2), dtype=np.float32)

    # Box 1
    output[0, 0, 0] = 320.0  # cx
    output[0, 1, 0] = 320.0  # cy
    output[0, 2, 0] = 160.0  # w
    output[0, 3, 0] = 160.0  # h
    output[0, 4, 0] = 0.90   # bottle
    output[0, 5, 0] = 0.10   # can

    # Box 2 (overlapping box for bottle)
    output[0, 0, 1] = 322.0  # cx
    output[0, 1, 1] = 322.0  # cy
    output[0, 2, 1] = 160.0  # w
    output[0, 3, 1] = 160.0  # h
    output[0, 4, 1] = 0.80   # bottle
    output[0, 5, 1] = 0.05   # can

    detections = yolo_xywh_scores_v1(
        output,
        class_map=class_map,
        orig_shape=orig_shape,
        preprocess_meta=preprocess_meta,
        score_threshold=0.25,
        nms_iou_threshold=0.45,
    )

    # Box 2 should be suppressed by Box 1 due to NMS (same class 'bottle' and high IoU)
    assert len(detections) == 1
    det = detections[0]
    assert det["class_id"] == "bottle"
    assert det["score"] == 0.90

    # Invert letterbox:
    # Model box: [320 - 80, 320 - 80, 320 + 80, 320 + 80] = [240, 240, 400, 400]
    # x_orig = (x_model - pad_left) / scale = (240 - 0) / 0.8 = 300, (400 - 0) / 0.8 = 500
    # y_orig = (y_model - pad_top) / scale = (240 - 160) / 0.8 = 100, (400 - 160) / 0.8 = 300
    expected_bbox = [300.0, 100.0, 500.0, 300.0]
    np.testing.assert_allclose(det["bbox_xyxy"], expected_bbox, atol=1e-1)
