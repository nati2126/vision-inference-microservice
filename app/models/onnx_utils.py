"""Helpers for reading facts out of an ONNX artifact."""

from pathlib import Path

import onnx

# Operators that only appear in a quantised graph. ONNX Runtime's static
# quantiser emits QuantizeLinear/DequantizeLinear pairs (QDQ format); the
# older operator-oriented format uses the QLinear* / ConvInteger family.
_INT8_OPS = frozenset(
    {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul", "ConvInteger"}
)


def onnx_precision(path: str | Path) -> str:
    """Infer ``int8`` / ``fp16`` / ``fp32`` from the graph itself."""
    model = onnx.load(str(path), load_external_data=False)
    if any(node.op_type in _INT8_OPS for node in model.graph.node):
        return "int8"
    elem_type = model.graph.input[0].type.tensor_type.elem_type
    if elem_type == onnx.TensorProto.FLOAT16:
        return "fp16"
    return "fp32"
