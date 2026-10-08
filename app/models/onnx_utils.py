"""ONNX artifact helpers."""

from pathlib import Path

import onnx

_INT8_OPS = frozenset(
    {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul", "ConvInteger"}
)


def onnx_precision(path: str | Path) -> str:
    """int8, fp16 or fp32, read from the graph."""
    model = onnx.load(str(path), load_external_data=False)
    if any(node.op_type in _INT8_OPS for node in model.graph.node):
        return "int8"
    elem_type = model.graph.input[0].type.tensor_type.elem_type
    if elem_type == onnx.TensorProto.FLOAT16:
        return "fp16"
    return "fp32"
