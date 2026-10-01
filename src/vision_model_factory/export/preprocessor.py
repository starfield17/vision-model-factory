"""Reference letterbox preprocessor implementation for letterbox_rgb_u8_v1."""

import math
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def letterbox_rgb_u8_v1(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    pad_value: int = 114,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Implements letterbox_rgb_u8_v1 preprocessing contract:
    - Input: uint8 image (H, W, 3) or (H, W)
    - Convert to RGB uint8
    - Preserve aspect ratio: scale = min(target_h / H, target_w / W)
    - Target dimensions: floor(value + 0.5)
    - Bilinear resize
    - Place smaller half of remaining padding on left/top, filled with pad_value
    - Convert to NCHW float32 divided by 255.0

    Returns:
        (tensor_nchw, metadata)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be a numpy ndarray, got {type(image)}")
    if image.dtype != np.uint8:
        raise ValueError(f"Image dtype must be uint8, got {image.dtype}")

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {image.shape}")

    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_shape

    scale = min(target_h / orig_h, target_w / orig_w)
    new_w = int(math.floor(orig_w * scale + 0.5))
    new_h = int(math.floor(orig_h * scale + 0.5))

    # Resize using PIL Bilinear resampling
    pil_img = Image.fromarray(image, mode="RGB")
    resized_pil = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    resized = np.array(resized_pil, dtype=np.uint8)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    # Smaller half of remaining padding on left / top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    # Convert to NCHW float32 / 255.0
    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # Shape: (1, 3, target_h, target_w)

    metadata: Dict[str, Any] = {
        "id": "letterbox_rgb_u8_v1",
        "orig_shape": (orig_h, orig_w),
        "target_shape": (target_h, target_w),
        "scale": scale,
        "new_shape": (new_h, new_w),
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_value": pad_value,
    }

    return tensor, metadata
