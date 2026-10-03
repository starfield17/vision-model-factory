"""ONNX graph introspection: read the declared contract out of the real model file.

The publication rule is that input/output names, dtypes and shapes are read from the
exported graph rather than assumed from the checkpoint name or a code default
(02-model-factory.md 3: "实际名称、shape 由导出器检查，不凭模型文件名推断"). Everything the
publisher writes into `model.json` about the tensor interface comes from here.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import onnx
import onnxruntime as ort
from onnx import TensorProto

_ONNX_DTYPE_TO_NAME = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.FLOAT16: "float16",
}

class GraphInspectionError(ValueError):
    """Raised when an ONNX file cannot be interpreted as a declared model interface."""


def _dim_to_declaration(dim: Any) -> Union[int, str]:
    """Map an ONNX tensor dimension to the manifest declaration form.

    A static dimension is published as a positive integer; a symbolic/dynamic one is
    published under the name the graph itself declares, never collapsed to a number.
    """
    if dim.HasField("dim_param") and dim.dim_param:
        return str(dim.dim_param)
    if dim.HasField("dim_value") and dim.dim_value > 0:
        return int(dim.dim_value)
    raise GraphInspectionError(
        "Tensor dimension is neither a positive static value nor a named dynamic axis; "
        "the manifest cannot declare it."
    )


def _value_info_declaration(value_info: Any) -> Dict[str, Any]:
    elem = value_info.type.tensor_type
    dtype = _ONNX_DTYPE_TO_NAME.get(elem.elem_type)
    if dtype is None:
        raise GraphInspectionError(
            f"Unsupported ONNX element type {elem.elem_type} for tensor '{value_info.name}'"
        )
    shape = [_dim_to_declaration(d) for d in elem.shape.dim]
    return {"name": value_info.name, "dtype": dtype, "shape": shape}


def describe_onnx_graph(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    """Return the declared tensor interface and dtype summary of an ONNX model file."""
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {path}")

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    graph_inputs = [i for i in model.graph.input if not _is_initializer(model, i.name)]
    if len(graph_inputs) != 1:
        raise GraphInspectionError(
            f"Expected exactly one non-initializer graph input, found {len(graph_inputs)} "
            f"({[i.name for i in graph_inputs]}). A model with extra inputs cannot be "
            "described by the single-input deployment contract."
        )
    if len(model.graph.output) != 1:
        raise GraphInspectionError(
            f"Expected exactly one graph output, found {len(model.graph.output)} "
            f"({[o.name for o in model.graph.output]}). NMS-fused or multi-output exports "
            "require a different decoder id and cannot use this path."
        )

    opset = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}

    return {
        "path": str(path.resolve()),
        "input": _value_info_declaration(graph_inputs[0]),
        "output": _value_info_declaration(model.graph.output[0]),
        "opset": opset,
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "initializer_count": len(model.graph.initializer),
        "quantized": graph_uses_integer_quantization(model),
    }


def _is_initializer(model: Any, name: str) -> bool:
    return any(init.name == name for init in model.graph.initializer)


def graph_uses_integer_quantization(model: Any) -> bool:
    """Detect an integer-quantized graph from node types and tensor element types."""
    for node in model.graph.node:
        op = node.op_type.upper()
        if op in ("QUANTIZE_LINEAR", "DEQUANTIZE_LINEAR", "QLINEARCONV", "QLINEARMATMUL"):
            return True
    for init in model.graph.initializer:
        if _ONNX_DTYPE_TO_NAME.get(init.data_type, "") in ("int8", "uint8"):
            return True
    return False


def infer_precision(onnx_path: Union[str, Path]) -> str:
    """Return the precision the graph actually declares: 'int8' or 'fp32'."""
    path = Path(onnx_path)
    model = onnx.load(str(path))
    return "int8" if graph_uses_integer_quantization(model) else "fp32"


def active_providers(session: "ort.InferenceSession") -> List[str]:
    """Return the providers actually active on a session.

    A requested provider that silently fell back must never be published as the
    deployment profile: the package would declare support it was never measured against.
    """
    return list(session.get_providers())


__all__ = [
    "GraphInspectionError",
    "active_providers",
    "describe_onnx_graph",
    "graph_uses_integer_quantization",
    "infer_precision",
]
