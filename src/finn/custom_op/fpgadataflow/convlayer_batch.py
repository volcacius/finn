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
            "ConvKernelDim": ("i", True, 0),
            "IFMChannels": ("i", True, 0),
            "IFMDim": ("i", True, 0),
            "OFMChannels": ("i", True, 0),
            "OFMDim": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "PE": ("i", True, 0),
            "resType": ("s", True, ""),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            "binaryXnorMode": ("i", False, 0),
            "noActivation": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def calc_wmem(self):
        k = self.get_nodeattr("ConvKernelDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ofm_ch = self.get_nodeattr("OFMChannels")
        mw = k * k * ifm_ch
        mh = ofm_ch
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        assert mh % pe == 0
        assert mw % simd == 0
        wmem = mw * mh // (pe * simd)
        return wmem


    def calc_tmem(self):
        if self.get_nodeattr("noActivation") == 1:
            return 0
        else:
            ofm_ch = self.get_nodeattr("OFMChannels")
            mh = ofm_ch
            pe = self.get_nodeattr("PE")
            return mh // pe


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

    def get_template_param_values(self):
        ret = dict()
        inp_hls_str = self.get_input_datatype().get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype() == DataType.BINARY
        out_is_binary = self.get_output_datatype() == DataType.BINARY
        wt_is_binary = self.get_weight_datatype() == DataType.BINARY
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise Exception("True binary (non-bipolar) inputs not yet supported")
        inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
        out_is_bipolar = self.get_output_datatype() == DataType.BIPOLAR
        wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # fill in TSrcI and TWeightI
        # TODO check these with Giulio
        # TODO handle non-bipolar binary inputs
        if inp_is_bipolar and wt_is_bipolar:
            ret["TSrcI"] = "Recast<XnorMul>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and wt_is_bipolar:
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Recast<Binary>"
        elif inp_is_bipolar and (not wt_is_bipolar):
            ret["TSrcI"] = "Recast<Binary>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and (not wt_is_bipolar):
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Identity"
        # fill in TDstI
        if out_is_bipolar or out_is_binary:
            ret["TDstI"] = "Identity"
        else:
            ret["TDstI"] = "Slice<%s>" % out_hls_str
        return ret

    def get_hls_compatible_weight_tensor(self, orig_weight_matrix):
        pass

    def get_hls_compatible_threshold_tensor(self, orig_thres_matrix):
        pass 
    
    def generate_params(self, model):
        pass

    def execute_node(self, context, graph):

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
        code_gen_dir = self.get_nodeattr("code_gen_dir")
        dtype = self.get_input_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        ) 

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def docompute(self):
        node = self.onnx_node
        tmpl_args = self.get_template_param_values()
        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            threshs = "PassThroughActivation<%s>()" % odtype_hls_str
        else:
            threshs = "threshs"
        
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<ConvKernelDim1, IFMChannels1, IFMDim1, OFMChannels1, 
            OFMDim1, SIMD1, PE1, {}, {}, {}>
            (in0, out, weights, {}, numReps, {});""".format(
                node.op_type,
                tmpl_args["TSrcI"],
                tmpl_args["TDstI"],
                tmpl_args["TWeightI"],
                threshs,
                self.get_nodeattr("resType"),
            )
        ]


    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir")
        dtype = self.get_output_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        nf = int(self.get_nodeattr("MH") / self.get_nodeattr("PE"))
        shape = (1, nf, self.get_nodeattr("PE"))
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s");'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                shape_cpp_str,
                npy_out,
            )
        ]


    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []
