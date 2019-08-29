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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_fat_fc.cc
 * \brief MKLDNN (Quantized) fat FullyConnected operator based on subgraph
*/

#if MXNET_USE_MKLDNN == 1

#include <utility>
#include <vector>
#include <string>
#include "../common.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

struct MKLDNNFCFatFullParam {
  std::vector<FullyConnectedParam> default_params;
  MKLDNNFCParam mkldnn_param;
  std::vector<float> output_scales = {0.0};
  std::vector<float> requantize_scales = {0.0};
};

class SgMKLDNNFatFCOp {
 public:
  explicit SgMKLDNNFatFCOp(const nnvm::NodeAttrs &attrs)
      : initialized_(false),
        subgraph_sym_(*attrs.subgraphs[0]),
        full_param_(nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx, const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req, const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx, const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req, const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn fully connected only supports "
                  "inference computation.";
  }

 private:
  bool initialized_;
  nnvm::Symbol subgraph_sym_;
  MKLDNNFCFatFullParam full_param_;
  std::vector<std::shared_ptr<mkldnn::inner_product_forward>> fwd_;
  std::vector<std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>> fc_pd_;
  std::shared_ptr<mkldnn::memory> cached_data_;
  std::vector<std::shared_ptr<mkldnn::memory>> cached_weight_;
  std::vector<std::shared_ptr<mkldnn::memory>> cached_bias_;
  std::vector<std::shared_ptr<mkldnn::memory>> cached_output_;
  float cached_min_data_;
  float cached_max_data_;
  float cached_min_weight_;
  float cached_max_weight_;
  float cached_min_bias_;
  float cached_max_bias_;
  float cached_min_output_;
  float cached_max_output_;
};

void SgMKLDNNFatFCOp::Forward(const OpContext &ctx, const std::vector<NDArray> &in_data,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &out_data) {
  auto &mkldnn_param = full_param_.mkldnn_param;
  auto &default_param = full_param_.default_params[0];
  const auto num_fc = full_param_.default_params.size();
  bool has_bias = !default_param.no_bias;
  size_t base_num_inputs = has_bias ? 3 : 2;
  auto data = in_data[fullc::kData];
  if (!initialized_) {
    initialized_ = true;
    omp_set_max_active_levels(10);
    omp_set_nested(1);
    fwd_.resize(num_fc);
    fc_pd_.resize(num_fc);
    cached_weight_.resize(num_fc);
    cached_bias_.resize(num_fc);
    cached_output_.resize(num_fc);
    const mxnet::TShape ishape = data.shape();
    if (ishape.ndim() != 2) {
      if (!default_param.flatten) {
        data = NDArray(Shape2(ishape.ProdShape(0, ishape.ndim() - 1), ishape[ishape.ndim() - 1]),
                       data.ctx(), true, data.dtype());
      } else {
        data = NDArray(Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), data.ctx(), true,
                       data.dtype());
      }
    }
    cached_data_ =
        std::make_shared<mkldnn::memory>(data.GetMKLDNNData()->get_primitive_desc(), nullptr);
    for (size_t i = 0; i < num_fc; i++) {
      const auto& weight = in_data[i * base_num_inputs + fullc::kWeight];
      const auto &out = out_data[i + fullc::kOut];
      mkldnn::memory::desc out_md = GetMemDesc(out);

      if (ishape.ndim() != 2) {
        const mxnet::TShape oshape = out.shape();
        if (!default_param.flatten) {
          mkldnn::memory::dims out_dims{static_cast<int>(oshape.ProdShape(0, oshape.ndim() - 1)),
                                        static_cast<int>(oshape[ishape.ndim() - 1])};
          out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(out.dtype()),
                                        mkldnn::memory::format::any);
        } else {
          mkldnn::memory::dims out_dims{static_cast<int>(oshape[0]),
                                        static_cast<int>(oshape.ProdShape(1, oshape.ndim()))};
          out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(out.dtype()),
                                        mkldnn::memory::format::any);
        }
      }
      MKLDNNFCFullParam param;
      param.default_param = full_param_.default_params[i];
      param.mkldnn_param = mkldnn_param;
      param.output_scales = full_param_.output_scales;
      param.requantize_scales = full_param_.requantize_scales;
      fc_pd_[i] = std::make_shared<mkldnn::inner_product_forward::primitive_desc>(GetFCFwdImpl(
          param, false, data, weight,
          (has_bias ? &in_data[i * base_num_inputs + fullc::kBias] : nullptr), out_md));
      cached_output_[i] =
          std::make_shared<mkldnn::memory>(fc_pd_[i]->dst_primitive_desc(), nullptr);
      cached_weight_[i] =
          std::make_shared<mkldnn::memory>(fc_pd_[i]->weights_primitive_desc(), nullptr);
      if (has_bias) {
        cached_bias_[i] = std::make_shared<mkldnn::memory>(fc_pd_[i]->bias_primitive_desc(), nullptr);
        fwd_[i] = std::make_shared<mkldnn::inner_product_forward>(
            *fc_pd_[i], mkldnn::primitive::at(*cached_data_),
            mkldnn::primitive::at(*cached_weight_[i]), mkldnn::primitive::at(*cached_bias_[i]),
            *cached_output_[i]);
      } else {
        fwd_[i] = std::make_shared<mkldnn::inner_product_forward>(
            *fc_pd_[i], mkldnn::primitive::at(*cached_data_),
            mkldnn::primitive::at(*cached_weight_[i]), *cached_output_[i]);
      }
    }
  }
  data = in_data[fullc::kData];
  cached_data_->set_data_handle(data.GetMKLDNNData()->get_data_handle());
  int num_omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
#pragma omp parallel for num_threads(num_fc)
  for (int i = 0; i < static_cast<int>(num_fc); i++) {
    omp_set_num_threads(num_omp_threads / num_fc);
    const auto& weight = in_data[i * base_num_inputs + fullc::kWeight];
    cached_weight_[i]->set_data_handle(weight.GetMKLDNNData()->get_data_handle());
    if (has_bias) {
      const auto& bias = in_data[i * base_num_inputs + fullc::kBias];
      cached_bias_[i]->set_data_handle(bias.GetMKLDNNData()->get_data_handle());
    }
    auto out_mem = CreateMKLDNNMem(out_data[i + fullc::kOut], fc_pd_[i]->dst_primitive_desc(),
                                   req[i + fullc::kOut], &data);
    cached_output_[i]->set_data_handle(out_mem.second->get_data_handle());
    MKLDNNStream::Get()->RegisterPrim(*fwd_[i]);
    CommitOutput(out_data[i + fullc::kOut], out_mem);
    MKLDNNStream::Get()->Submit();
  }
  omp_set_num_threads(num_omp_threads);
}

static void SgMKLDNNFatFCParamParser(nnvm::NodeAttrs *attrs) {
  MKLDNNFCFatFullParam full_param;
  try {
    full_param.mkldnn_param.Init(attrs->dict);
  } catch (const dmlc::ParamError &e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto &k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  auto subgraph_sym = attrs->subgraphs[0];
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::NodePtr &node) {
    if (node->is_variable()) return;
    auto &node_name = node->op()->name;
    if (node_name == "FullyConnected") {
      full_param.default_params.push_back(nnvm::get<FullyConnectedParam>(node->attrs.parsed));
    }
  });
  attrs->parsed = std::move(full_param);
}

static bool SgMKLDNNFatFCInferShape(const nnvm::NodeAttrs &attrs, mxnet::ShapeVector *in_shapes,
                                    mxnet::ShapeVector *out_shapes) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    const auto num_fc = full_param.default_params.size();
    const auto base_num_inputs = full_param.default_params[0].no_bias ? 2 : 3;
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes;
    for (size_t i = 0; i < num_fc * base_num_inputs; i++) {
      base_in_shapes.push_back(in_shapes->at(i));
    }
    for (size_t i = 0; i < num_fc; i++) {
      base_out_shapes.push_back(out_shapes->at(i));
    }
    bool ret = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);
    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (i < base_in_shapes.size())
        in_shapes->at(i) = base_in_shapes[i];
      else
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
    }
    for (size_t i = 0; i < out_shapes->size(); ++i) {
      if (i < base_out_shapes.size())
        out_shapes->at(i) = base_out_shapes[i];
      else
        SHAPE_ASSIGN_CHECK(*out_shapes, i, Shape1(1));
    }
    return ret;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgMKLDNNFatFCInferType(const nnvm::NodeAttrs &attrs, std::vector<int> *in_types,
                                   std::vector<int> *out_types) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    const auto base_num_inputs = full_param.default_params[0].no_bias ? 2 : 3;
    const auto num_fc = full_param.default_params.size();

    CHECK(in_types->at(0) == mshadow::kInt8 || in_types->at(0) == mshadow::kUint8)
        << "Quantized Fat FullyConnected only supports int8/uint8 input, while " << in_types->at(0)
        << " is given.";
    for (size_t i = 1; i < in_types->size(); ++i) {
      if (i < base_num_inputs * num_fc) {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
      }
    }

    if (full_param.mkldnn_param.enable_float_output) {
      for (size_t i = 0; i < out_types->size(); ++i) {
        TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kFloat32);
      }
    } else {
      if (full_param.mkldnn_param.min_calib_range.has_value() &&
          full_param.mkldnn_param.max_calib_range.has_value()) {
        if (full_param.mkldnn_param.with_eltwise) {
          for (size_t i = 0; i < num_fc; ++i) {
            TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kUint8);
          }
        } else {
          for (size_t i = 0; i < num_fc; ++i) {
            TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kInt8);
          }
        }
      } else {
        for (size_t i = 0; i < num_fc; ++i) {
          TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kInt32);
        }
        for (size_t i = num_fc; i < out_types->size(); ++i) {
          TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kFloat32);
        }
      }
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

static bool SgMKLDNNFatFCStorageType(const nnvm::NodeAttrs &attrs, const int dev_mask,
                                     DispatchMode *dispatch_mode, std::vector<int> *in_attrs,
                                     std::vector<int> *out_attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    const auto num_fc = full_param.default_params.size();
    const auto base_num_inputs = full_param.default_params[0].no_bias ? 2 : 3;
    std::vector<int> base_in_attrs;
    std::vector<int> base_out_attrs;
    for (size_t i = 0; i < num_fc * base_num_inputs; i++) {
      base_in_attrs.push_back(in_attrs->at(i));
    }
    for (size_t i = 0; i < num_fc; i++) {
      base_out_attrs.push_back(out_attrs->at(i));
    }
    bool ret = DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode, &base_in_attrs,
                                            &base_out_attrs);

    for (size_t i = 0; i < in_attrs->size(); ++i) {
      if (i < base_in_attrs.size())
        in_attrs->at(i) = base_in_attrs[i];
      else
        type_assign(&in_attrs->at(i), mxnet::kDefaultStorage);
    }
    for (size_t i = 0; i < out_attrs->size(); ++i) {
      if (i < base_out_attrs.size())
        out_attrs->at(i) = base_out_attrs[0];
      else
        type_assign(&out_attrs->at(i), mxnet::kDefaultStorage);
    }
    return ret;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
  }
}

static OpStatePtr CreateSgMKLDNNFatFCState(const nnvm::NodeAttrs &attrs, Context ctx,
                                           const mxnet::ShapeVector &in_shapes,
                                           const std::vector<int> &in_types) {
  return OpStatePtr::Create<SgMKLDNNFatFCOp>(attrs);
}

static void SgMKLDNNFatFCForward(const OpStatePtr &state_pointer, const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  SgMKLDNNFatFCOp &op = state_pointer.get_state<SgMKLDNNFatFCOp>();
  op.Forward(ctx, inputs, req, outputs);
}

nnvm::NodePtr SgMKLDNNFatFCQuantizedOp(const NodeAttrs &attrs) {
  nnvm::NodePtr node = nnvm::Node::Create();
  node->attrs.op = Op::Get("_sg_mkldnn_fat_fully_connected");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

int SgMKLDNNFatFCNumOutputs(const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  auto num_fc = full_param.default_params.size();
  return ((full_param.mkldnn_param.quantized &&
          !full_param.mkldnn_param.enable_float_output) ? 3 : 1) * num_fc;
}

NNVM_REGISTER_OP(_sg_mkldnn_fat_fully_connected)
.describe(R"code(_sg_mkldnn_fat_fully_connected)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  auto num_fc = full_param.default_params.size();
  auto num_inputs = (full_param.default_params[0].no_bias ? 2 : 3) * num_fc;
  if (full_param.mkldnn_param.quantized)
    return num_inputs * 3;
  else
    return num_inputs;
})
.set_num_outputs(SgMKLDNNFatFCNumOutputs)
.set_attr_parser(SgMKLDNNFatFCParamParser)
.set_attr<mxnet::FInferShape>("FInferShape", SgMKLDNNFatFCInferShape)
.set_attr<nnvm::FInferType>("FInferType", SgMKLDNNFatFCInferType)
.set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNFatFCStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNFatFCState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNFatFCForward)
.set_attr<bool>("TIsMKLDNN", true)
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
// .set_attr<FDynamicOutput>("FDynamicOutput", [](const NodeAttrs& n) {
//   const auto num_outputs = SgMKLDNNFatFCNumOutputs(n);
//   return std::vector<bool>(num_outputs, true);
// })
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNFatFCQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; });

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
