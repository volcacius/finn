import pytest

import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.utils import calculate_signed_dot_prod_range, gen_finn_dt_tensor
from finn.custom_op.multithreshold import multithreshold
from finn.transformation.fpgadataflow.codegen import CodeGen
from finn.transformation.fpgadataflow.compile import Compile


def make_single_fclayer_modelwrapper(
    W, k, ifm_ch, ifm_dim, ofm_ch, ofm_dim, pe, simd, wdt, idt, odt, T=None, tdt=None
):

    # there are two ways to implement bipolar weights and inputs for
    # StreamingFC:
    # - specify their datatypes as such
    # - specify their datatypes as BINARY as use binaryXnorMode
    if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
        # we'll internally convert weights/inputs to binary and specify the
        # datatypes as such, and also set the binaryXnorMode attribute to 1
        export_wdt = DataType.BINARY
        export_idt = DataType.BINARY
        binary_xnor_mode = 1
    else:
        export_wdt = wdt
        export_idt = idt
        binary_xnor_mode = 0

    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [ofm_ch, ofm_dim * ofm_dim]
    )
    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType.BIPOLAR:
            actval = 0
        else:
            actval = odt.min()
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
        actval = 0
        no_act = 1
    ConvLayer_node = helper.make_node(
        "ConvLayer_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        ConvKernelDim=k,
        IFMChannels=ifm_ch,
        IFMDim=ifm_dim,
        OFMChannels=ofm_ch,
        OFMDim=ofm_dim,
        SIMD=simd,
        PE=pe,
        inputDataType=export_idt.name,
        weightDataType=export_wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )
    graph = helper.make_graph(
        nodes=[ConvLayer_node], name="convlayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="convlayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    if binary_xnor_mode:
        # convert bipolar to binary
        model.set_initializer("weights", (W + 1) / 2)
    else:
        model.set_initializer("weights", W)
    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)
    return model


def prepare_inputs(input_tensor, idt, wdt):
    if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
        # convert bipolar to binary
        return {"inp": (input_tensor + 1) / 2}
    else:
        return {"inp": input_tensor}


def expected_conv_output_no_pad(x, W, k, idt, wdt, ofm_ch, ofm_dim):

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


# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType.INT2])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType.INT2])
# input datatype
@pytest.mark.parametrize("idt", [DataType.INT2])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("ifm_dim", [3, 6, 9])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("k", [2, 3])
# HLS matrix width (input features)
@pytest.mark.parametrize("pe", [1])
# HLS matrix height (output features)
@pytest.mark.parametrize("simd", [1])
def test_fpgadataflow_convlayer(act, wdt, idt, ifm_dim, k, pe, simd):

    ifm_ch = 1
    ofm_dim = ifm_dim - k + 1
    ofm_ch = 1
    mw = k * k * ifm_ch
    mh = ofm_ch
    W = gen_finn_dt_tensor(wdt, (ofm_ch, ifm_ch, k, k))

    if act is None:
        # no activation, produce accumulators
        T = None
        tdt = None
        if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
            odt = DataType.UINT32
        else:
            odt = DataType.INT32
    else:
        odt = act
        (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
        # provide non-decreasing thresholds
        T = np.sort(T, axis=1)
        # generate thresholds for activation
        if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
            tdt = DataType.UINT32
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType.INT32

    model = make_single_fclayer_modelwrapper(
        W, k, ifm_ch, ifm_dim, ofm_ch, ofm_dim, pe, simd, wdt, idt, odt, T, tdt
    )

    x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_dim, ifm_dim))

    model = model.transform(CodeGen())
    model = model.transform(Compile())

    # prepare input data
    input_dict = prepare_inputs(x, idt, wdt)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    # to calculate the expected output the ONNX convolution node is used
    oshape = model.get_tensor_shape("outp")
    expected_conv_output = expected_conv_output_no_pad(
        x, W, k, idt, wdt, ofm_ch, ofm_dim
    )
    y_expected = expected_conv_output.reshape(oshape)

    if T is not None:
        y_expected = multithreshold(y_expected, T)
        if act == DataType.BIPOLAR:
            # binary to bipolar
            y_expected = 2 * y_expected - 1
        else:
            # signed offset
            y_expected += act.min()

    assert (y_produced.reshape(y_expected.shape) == y_expected).all()
