import onnx
from onnx import helper, TensorProto
import numpy as np

def create_mlp_model(output_path="models/model.onnx"):
    # Input tensor: [batch_size, 8] with dynamic batch dimension
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch_size', 8])
    # Output tensor: [batch_size, 2]
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch_size', 2])

    # Weights and Biases for Layer 1: [8, 4] and [4]
    np.random.seed(42)
    w1_val = np.random.randn(8, 4).astype(np.float32)
    b1_val = np.random.randn(4).astype(np.float32)

    # Weights and Biases for Layer 2: [4, 2] and [2]
    w2_val = np.random.randn(4, 2).astype(np.float32)
    b2_val = np.random.randn(2).astype(np.float32)

    w1_init = helper.make_tensor('W1', TensorProto.FLOAT, [8, 4], w1_val.flatten())
    b1_init = helper.make_tensor('B1', TensorProto.FLOAT, [4], b1_val.flatten())
    w2_init = helper.make_tensor('W2', TensorProto.FLOAT, [4, 2], w2_val.flatten())
    b2_init = helper.make_tensor('B2', TensorProto.FLOAT, [2], b2_val.flatten())

    # Nodes: Gemm1 -> Relu -> Gemm2
    gemm1_node = helper.make_node(
        'Gemm',
        inputs=['input', 'W1', 'B1'],
        outputs=['gemm1_out'],
        alpha=1.0,
        beta=1.0,
        transB=0
    )

    relu_node = helper.make_node(
        'Relu',
        inputs=['gemm1_out'],
        outputs=['relu_out']
    )

    gemm2_node = helper.make_node(
        'Gemm',
        inputs=['relu_out', 'W2', 'B2'],
        outputs=['output'],
        alpha=1.0,
        beta=1.0,
        transB=0
    )

    # Graph
    graph_def = helper.make_graph(
        [gemm1_node, relu_node, gemm2_node],
        'DemoMLP',
        [input_tensor],
        [output_tensor],
        [w1_init, b1_init, w2_init, b2_init]
    )

    # Model with ir_version=10 and opset 17 for ONNX Runtime 1.18 compatibility
    model_def = helper.make_model(
        graph_def,
        producer_name='neuroserve',
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_path)
    print(f"Generated ONNX model with ir_version=10 saved to: {output_path}")

if __name__ == '__main__':
    create_mlp_model()
