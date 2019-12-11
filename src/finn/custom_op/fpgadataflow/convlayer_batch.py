import os

import numpy as np

# from finn.backend.fpgadataflow.utils import numpy_to_hls_code
from finn.core.datatype import DataType
# from finn.core.utils import interleave_matrix_outer_dim_from_partitions
from finn.custom_op.fpgadataflow import HLSCustomOp


class ConvLayer_Batch(HLSCustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "backend": ("s", True, "fpgadataflow"),
            "code_gen_dir": ("s", True, ""),
            "executable_path": ("s", True, ""),
            "ConvKernelDim": ("i", True, 0),
            "IFMChannels": ("i", True, 0),
            "IFMDim": ("i", True, 0),
            "OFMChannels": ("i", True, 0),
            "OFMDim": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "PE": ("i", True, 0),
            "resType": ("s", True, ""),
            "ActVal": ("i", False, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            "binaryXnorMode": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def calc_wmem(self):
        ifmdim = self.get_nodeattr("IFMDim")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        assert ifmdim % pe == 0
        assert ifmdim % simd == 0
        wmem = ifmdim * ifmdim // (pe * simd)
        return wmem

    def calc_tmem(self):
        ifmdim = self.get_nodeattr("IFMDIM")
        pe = self.get_nodeattr("PE")
        return ifmdim // pe

    def make_shape_compatible_op(self):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        pass

    def get_input_datatype(self):
        return DataType[self.get_nodeattr("inputDataType")]

    def get_weight_datatype(self):
        return DataType[self.get_nodeattr("weightDataType")]

    def get_output_datatype(self):
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self):
        i_bits = self.get_input_datatype().bitwidth()
        return i_bits * self.get_nodeattr("SIMD")

    def get_outstream_width(self):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def execute_node(self, context, graph):
        node = self.onnx_node
        ifmdim = self.get_nodeattr("IFMDim")
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        sf = ifmdim // simd
        nf = ifmdim // pe

        # TODO ensure codegen dir exists
        code_gen_dir = self.get_nodeattr("code_gen_dir")
        # create a npy file fore each input of the node (in_ind is input index)
        in_ind = 0
        for inputs in node.input:
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
            if in_ind == 0:
                assert str(context[inputs].dtype) == "float32"
                expected_inp_shape = (1, sf, simd)
                reshaped_input = context[inputs].reshape(expected_inp_shape)
                # flip SIMD (innermost) dimension of input tensor, there's some reversal
                # going on somewhere with a mistmatch between npy and hls...
                reshaped_input = np.flip(reshaped_input, -1)
                if self.get_input_datatype() == DataType.BIPOLAR:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for ConvLayer")
            in_ind += 1
        # execute the precompiled model
        super().exec_precompiled_singlenode_model()
        # load output npy file
        super().npy_to_dynamic_output(context)
        # reinterpret binary output as bipolar where needed
        if self.get_output_datatype() == DataType.BIPOLAR:
            out = context[node.output[0]]
            out = 2 * out - 1
            context[node.output[0]] = out
        assert context[node.output[0]].shape == (1, nf, pe)
        # reshape output to have expected shape
        context[node.output[0]] = context[node.output[0]].reshape(1, ifmdim)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "params.h"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self):
        numReps = 1
        self.code_gen_dict["$DEFINES$"] = [
            """#define ConvKernelDim1 {}\n #define IFMChannels1 {}
            #define IFMDim1 {}\n #define OFMChannels1 {}\n #define OFMDim1 {}
            #define SIMD1 \n #define PE1 {}\n #define WMEM1 {}\n #define TMEM1 {}
            #define numReps {}""".format(
                self.get_nodeattr("ConvKernelDim"),
                self.get_nodeattr("IFMChannels"),
                self.get_nodeattr("IFMDim"),
                self.get_nodeattr("OFMChannels"),
                self.get_nodeattr("OFMDim"),
                self.get_nodeattr("SIMD"),
                self.get_nodeattr("PE"),
                self.calc_wmem(),
                self.calc_tmem(),
                numReps,
            )
        ]

    def read_npy_data(self):
        pass

    def strm_decl(self):
        pass

    def docompute(self):
        pass

    def dataoutstrm(self):
        pass

    def save_as_npy(self):
        pass
