"""ONNX model exporter for Vision Model Factory."""

from pathlib import Path
from typing import Optional, Tuple, Union

import onnx
import torch
import torch.nn as nn

from vision_model_factory.contracts.hashing import compute_sha256_file


def export_torch_model_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    input_name: str = "images",
    output_name: str = "output0",
    opset_version: int = 17,
    dynamic_axes: Optional[dict] = None,
) -> Tuple[Path, str]:
    """
    Export a PyTorch nn.Module to ONNX format and verify validity.

    Args:
        model: PyTorch model in eval mode
        output_path: Target path for the .onnx file
        input_shape: Input tensor shape (batch, channels, height, width)
        input_name: Input tensor name (contract default: 'images')
        output_name: Output tensor name (contract default: 'output0')
        opset_version: ONNX opset version
        dynamic_axes: Optional dynamic axes specification

    Returns:
        (saved_path, sha256_digest)
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.zeros(input_shape, dtype=torch.float32)

    with torch.no_grad():
        export_kwargs = {
            "export_params": True,
            "opset_version": opset_version,
            "do_constant_folding": True,
            "input_names": [input_name],
            "output_names": [output_name],
            "dynamic_axes": dynamic_axes,
        }
        # Avoid onnxscript dependency requirement in PyTorch 2.14+
        try:
            torch.onnx.export(model, dummy_input, str(out_file), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy_input, str(out_file), **export_kwargs)

    # Verify ONNX model structure
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)

    sha256 = compute_sha256_file(out_file)
    return out_file, sha256


def export_yolo_checkpoint_to_onnx(
    checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
    imgsz: int = 640,
    opset_version: int = 18,
) -> Tuple[Path, str]:
    """
    Export a trained YOLO checkpoint (.pt) to standard ONNX.
    Ensures input tensor is 'images' [1, 3, imgsz, imgsz]
    and output tensor is 'output0' [1, 4 + K, N].

    Args:
        checkpoint_path: Path to YOLO PyTorch checkpoint (.pt)
        output_path: Target path for the .onnx file
        imgsz: Image dimension for model input
        opset_version: ONNX opset version

    Returns:
        (saved_path, sha256_digest)
    """
    import shutil

    from ultralytics import YOLO

    ckpt_file = Path(checkpoint_path).resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO(str(ckpt_file))
    exported_raw = yolo_model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
        opset=opset_version,
    )
    exported_path = Path(exported_raw).resolve()
    if exported_path != out_file:
        shutil.move(str(exported_path), str(out_file))

    # Verify ONNX model structure
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)

    sha256 = compute_sha256_file(out_file)
    return out_file, sha256
