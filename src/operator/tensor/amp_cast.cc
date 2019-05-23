/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file amp_cast.cc
 * \brief Casts used by AMP
 */

#include "./amp_cast.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(AMPCastParam);
DMLC_REGISTER_PARAMETER(AMPMultiCastParam);

#if MXNET_USE_MKLDNN == 1
static void AMPCastExCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const AMPCastParam& param = nnvm::get<AMPCastParam>(attrs.parsed);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  auto data = inputs[0];
  if (data.IsView() && data.IsMKLDNNData())
    data = data.Reorder2Default();
  const auto i_mem = data.GetMKLDNNData();
  auto i_mpd = i_mem->get_primitive_desc();
  auto i_desc = i_mpd.desc();
  const mkldnn::memory::format i_fmt = static_cast<mkldnn::memory::format>(i_desc.data.format);
  const size_t i_ndim = data.shape().ndim();
  mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
  for (size_t i = 0; i < i_ndim; i++) {
    i_dims[i] = static_cast<int>(data.shape()[i]);
  }
  const auto o_desc = mkldnn::memory::desc(i_dims, get_mkldnn_type(param.dtype), i_fmt);
  const auto o_mpd = memory::primitive_desc(o_desc, cpu_engine);
  const auto out_mem = CreateMKLDNNMem(outputs[0], o_mpd, req[0]);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(*i_mem, *out_mem.second));
  MKLDNNStream::Get()->Submit();
}

inline static bool AMPCastStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                                      DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
                                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

static void AMPMultiCastExCPU(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                              const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  const AMPMultiCastParam& param = nnvm::get<AMPMultiCastParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), param.num_outputs);
  CHECK_EQ(outputs.size(), param.num_outputs);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  for (size_t i = 0; i < param.num_outputs; ++i) {
    auto data = inputs[i];
    if (data.IsView() && data.IsMKLDNNData()) data = data.Reorder2Default();
    const auto i_mem = data.GetMKLDNNData();
    auto i_mpd = i_mem->get_primitive_desc();
    auto i_desc = i_mpd.desc();
    const mkldnn::memory::format i_fmt = static_cast<mkldnn::memory::format>(i_desc.data.format);
    const size_t i_ndim = data.shape().ndim();
    mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
    for (size_t i = 0; i < i_ndim; i++) {
      i_dims[i] = static_cast<int>(data.shape()[i]);
    }
    const auto o_desc = mkldnn::memory::desc(i_dims, get_mkldnn_type(outputs[i].dtype()), i_fmt);
    const auto o_mpd = memory::primitive_desc(o_desc, cpu_engine);
    const auto out_mem = CreateMKLDNNMem(outputs[i], o_mpd, req[0]);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(*i_mem, *out_mem.second));
  }
  MKLDNNStream::Get()->Submit();
}

inline static bool AMPMultiCastStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                                           DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
                                           std::vector<int>* out_attrs) {
  const AMPMultiCastParam& param = nnvm::get<AMPMultiCastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_outputs);
  CHECK_EQ(out_attrs->size(), param.num_outputs);
  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

#endif  // MXNET_USE_MKLDNN == 1

NNVM_REGISTER_OP(amp_cast)
.describe(R"code(Cast function between low precision float/FP32 used by AMP.

It casts only between low precision float/FP32 and does not do anything for other types.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<AMPCastParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", AMPCastType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", AMPCastCompute<cpu>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FInferStorageType>("FInferStorageType", AMPCastStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", AMPCastExCPU)
#endif
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_amp_cast"})
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(AMPCastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_amp_cast)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", AMPCastCompute<cpu>);

NNVM_REGISTER_OP(amp_multicast)
    .describe(R"code(Cast function used by AMP, that casts its inputs to the common widest type.

It casts only between low precision float/FP32 and does not do anything for other types.

)code" ADD_FILELINE)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_outputs);
    })
    .set_num_outputs([](const nnvm::NodeAttrs& attrs) {
      const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_outputs);
    })
    .set_attr_parser(ParamParser<AMPMultiCastParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", AMPMultiCastShape)
    .set_attr<nnvm::FInferType>("FInferType", AMPMultiCastType)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       uint32_t num_args =
                                           dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                       std::vector<std::string> ret;
                                       for (uint32_t i = 0; i < num_args; ++i) {
                                         ret.push_back(std::string("data_") + std::to_string(i));
                                       }
                                       return ret;
                                     })
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      int num_args =
                                          dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                      std::vector<std::pair<int, int>> ret;
                                      for (int i = 0; i < num_args; ++i) {
                                        ret.emplace_back(i, i);
                                      }
                                      return ret;
                                    })
    .set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
                                      [](const NodeAttrs& attrs) {
                                        int num_args =
                                            dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                        return std::vector<bool>(num_args, true);
                                      })
    .set_attr<FCompute>("FCompute<cpu>", AMPMultiCastCompute<cpu>)
#if MXNET_USE_MKLDNN == 1
    .set_attr<bool>("TIsMKLDNN", true)
    .set_attr<FInferStorageType>("FInferStorageType", AMPMultiCastStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", AMPMultiCastExCPU)
#endif
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_amp_multicast"})
    .add_argument("data", "NDArray-or-Symbol[]", "Weights")
    .add_arguments(AMPMultiCastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_amp_multicast)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_outputs);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_outputs);
  })
.set_attr_parser(ParamParser<AMPMultiCastParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("grad_") + std::to_string(i));
    }
    return ret;
  })
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    int num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    std::vector<std::pair<int, int>> ret;
    for (int i = 0; i < num_args; ++i) {
      ret.emplace_back(i, i);
    }
    return ret;
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    int num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    return std::vector<bool>(num_args, true);
  })
.set_attr<FCompute>("FCompute<cpu>", AMPMultiCastCompute<cpu>)
.add_argument("grad", "NDArray-or-Symbol[]", "Gradients")
.add_arguments(AMPMultiCastParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
