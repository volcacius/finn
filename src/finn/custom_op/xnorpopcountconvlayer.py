import onnxruntime as rt
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.custom_op import CustomOp


def conv_node_no_pad(x, W, k, ofm_ch, ofm_dim):
    x_exp = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    W_exp = helper.make_tensor_value_info("W", TensorProto.FLOAT, W.shape)
    y = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, ofm_ch, ofm_dim, ofm_dim]
    )
    conv_node_no_pad = helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        kernel_shape=[k, k],
        # Default values for other attributes:
        # strides=[1, 1], dilations=[1, 1], groups=1
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        nodes=[conv_node_no_pad], name="conv_graph", inputs=[x_exp, W_exp], outputs=[y]
    )

    model = helper.make_model(graph, producer_name="conv-model")
    input_dict = {"x": x, "W": W}
    print(input_dict)
    sess = rt.InferenceSession(model.SerializeToString())
    output_dict = sess.run(None, input_dict)

    return output_dict[0]


def xnorpopcount_convolution(inp, W, k):
    ofm_ch = W.shape[0]
    ifm_ch = inp.shape[1]
    ifm_dim = inp.shape[2]
    ofm_dim = ifm_dim - k + 1
    output = conv_node_no_pad(inp, W, k, ofm_ch, ofm_dim)

    return (output + (k * k * ifm_ch)) * 0.5


class XnorPopcountConvLayer(CustomOp):
    def get_nodeattr_types(self):
        return {
            "ConvKernelDim": ("i", True, 0),
        }

    def make_shape_compatible_op(self):
        node = self.onnx_node
        k = self.get_nodeattr("ConvKernelDim")
        return helper.make_node(
            "Conv",
            [node.input[0], node.input[1]],
            [node.output[0]],
            kernel_shape=[k, k],
            # Default values for other attributes:
            # strides=[1, 1], dilations=[1, 1], groups=1
            pads=[0, 0, 0, 0],
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # ensure inputs are binary
        assert model.get_tensor_datatype(node.input[0]) == DataType["BINARY"]
        assert model.get_tensor_datatype(node.input[1]) == DataType["BINARY"]
        # bipolar convlayer should have bipolar output
        model.set_tensor_datatype(node.output[0], DataType["BIPOLAR"])

    def execute_node(self, context, graph):
        node = self.onnx_node
        k = self.get_nodeattr("ConvKernelDim")
        # assume that first input is data and second is weights
        inp = context[node.input[0]]
        W = context[node.input[1]]
        # calculate output
        output = xnorpopcount_convolution(inp, W, k)
        # set context according to output name
        context[node.output[0]] = output

    def verify_node(self):
        pass
