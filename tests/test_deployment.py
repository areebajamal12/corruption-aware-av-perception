from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper

from av_perception.deployment.export_onnx import validate_onnx


def test_validate_onnx_executes_graph(tmp_path: Path) -> None:
    node = helper.make_node("Identity", ["images"], ["output"])
    graph = helper.make_graph(
        [node],
        "identity",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    path = tmp_path / "identity.onnx"
    onnx.save(model, path)
    result = validate_onnx(path, "images", tuple(np.zeros((1, 3, 2, 2)).shape))
    assert result["output_shapes"] == [[1, 3, 2, 2]]
