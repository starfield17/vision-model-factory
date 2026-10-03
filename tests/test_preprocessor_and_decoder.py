"""Tests for reference letterbox preprocessor and YOLO output decoder."""

import numpy as np
import pytest

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


def test_letterbox_odd_padding_puts_smaller_half_on_left_and_top():
    """The one-pixel rule a consumer must reproduce exactly.

    02-model-factory.md 3: "leftover padding's smaller half goes left/top". With an odd
    leftover the two halves differ by a pixel, and a decoder that assumes the other split
    maps every box back one pixel off — invisible in metrics, visible in the overlay.
    """
    import math

    # 480x640 source into 640x640: scale = 640/640 = 1.0 in width, so height scales to
    # 480 and pad_h = 160 (even). Force an odd leftover with a 481x640 source.
    img = np.zeros((481, 640, 3), dtype=np.uint8)
    img[:, :] = 77
    tensor, meta = letterbox_rgb_u8_v1(img, target_shape=(640, 640), pad_value=114)

    pad_h = 640 - meta["new_shape"][0]
    assert pad_h % 2 == 1, "fixture must produce an odd leftover padding to test the split"
    assert meta["pad_top"] == pad_h // 2
    assert meta["pad_bottom"] == pad_h - meta["pad_top"]
    assert meta["pad_top"] < meta["pad_bottom"]

    # The rule as declared in the docstring, and the pixel evidence that it holds.
    assert meta["pad_top"] == math.floor(pad_h / 2)
    pad_row = meta["pad_top"] - 1
    content_row = meta["pad_top"]
    np.testing.assert_allclose(tensor[0, :, pad_row, :], 114.0 / 255.0, atol=1e-6)
    np.testing.assert_allclose(tensor[0, :, content_row, :], 77.0 / 255.0, atol=1e-6)

    # Width is flush here (scale limited by width), so the horizontal pads stay zero.
    assert meta["pad_left"] == 0 and meta["pad_right"] == 0


def test_letterbox_odd_width_padding_is_also_smaller_on_the_left():
    """Same rule on the horizontal axis, where the source is height-limited."""
    img = np.zeros((640, 481, 3), dtype=np.uint8)
    img[:, :] = 200
    _tensor, meta = letterbox_rgb_u8_v1(img, target_shape=(640, 640), pad_value=114)

    pad_w = 640 - meta["new_shape"][1]
    assert pad_w % 2 == 1
    assert meta["pad_left"] == pad_w // 2 < meta["pad_right"]


def test_letterbox_reports_the_metadata_a_decoder_needs():
    """Every field `yolo_xywh_scores_v1` inverts through must be present and consistent."""
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    tensor, meta = letterbox_rgb_u8_v1(img, target_shape=(640, 640), pad_value=114)

    assert meta["id"] == "letterbox_rgb_u8_v1"
    assert meta["orig_shape"] == (300, 500)
    assert meta["target_shape"] == (640, 640)
    assert meta["pad_left"] + meta["pad_right"] + meta["new_shape"][1] == 640
    assert meta["pad_top"] + meta["pad_bottom"] + meta["new_shape"][0] == 640
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_letterbox_rejects_input_it_cannot_honestly_process():
    """A dtype or shape error must not be silently coerced into a wrong tensor."""
    with pytest.raises(ValueError, match="dtype must be uint8"):
        letterbox_rgb_u8_v1(np.zeros((10, 10, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="must be a numpy ndarray"):
        letterbox_rgb_u8_v1([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="Expected 3-channel image"):
        letterbox_rgb_u8_v1(np.zeros((10, 10, 2), dtype=np.uint8))

    # Grayscale and RGBA are normalised to RGB rather than refused: the contract is RGB.
    gray, _ = letterbox_rgb_u8_v1(np.zeros((20, 20), dtype=np.uint8), target_shape=(32, 32))
    assert gray.shape == (1, 3, 32, 32)
    rgba, _ = letterbox_rgb_u8_v1(np.zeros((20, 20, 4), dtype=np.uint8), target_shape=(32, 32))
    assert rgba.shape == (1, 3, 32, 32)
