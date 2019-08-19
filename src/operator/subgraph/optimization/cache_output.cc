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
 * \file cache_output.cc
 * \brief Do calculation only once and cache the result
 */

#include <string>
#include <utility>
#include <vector>
#include "../../../imperative/cached_op.h"
#include "../common.h"

namespace mxnet {
namespace op {

struct CachedOutputState {
  bool initalized_;
  OpStatePtr cached_op_state_;
  std::vector<size_t> input_versions_;
  std::vector<NDArray> cached_outputs_;
  explicit CachedOutputState(std::shared_ptr<CachedOp> op) {
    cached_op_state_ = OpStatePtr::Create<CachedOutputState>(op);
    initalized_ = false;
  }
};

void CachedOutputForward(const OpStatePtr& state_ptr, const OpContext& ctx,
                         const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  CachedOutputState& s = state_ptr.get_state<CachedOutputState>();
  if (s.initalized_) {
    // Check if inputs are all the same with cached result
    for (size_t i = 0; i < inputs.size(); i++) {
      if (s.input_versions_[i] != inputs[i].version()) {
        s.initalized_ = false;
        break;
      }
    }
  }
  // Copy cached result to output
  if (s.initalized_) {
    for (size_t i = 0; i < outputs.size(); i++) {
      CopyFromTo(s.cached_outputs_[i], outputs[i]);
    }
  } else {
    // Cache isn't created or dirty, need to create new one.
    CachedOpForward(s.cached_op_state_, ctx, inputs, req, outputs);
    s.input_versions_.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      s.input_versions_[i] = inputs[i].version();
    }
    // Cache output
    s.cached_outputs_.clear();
    s.cached_outputs_.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      s.cached_outputs_[i] =
          NDArray(outputs[i].shape(), outputs[i].ctx(), false, outputs[i].dtype());
      CopyFromTo(outputs[i], s.cached_outputs_[i]);
    }
    s.initalized_ = true;
  }
}

OpStatePtr CreateCachedOutputState(const NodeAttrs& attrs, Context ctx,
                                   const mxnet::ShapeVector& in_shapes,
                                   const std::vector<int>& in_types) {
  const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
  return OpStatePtr::Create<CachedOutputState>(op);
}

NNVM_REGISTER_OP(_CacheOutput)
.set_num_inputs([](const NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_outputs();
  })
.set_attr_parser(CachedOpParamParser)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(n->attrs.parsed);
    return op->Gradient(n, ograds);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->ListForwardInputNames();
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->ListForwardOutputNames();
  })
.set_attr<FCreateOpState>("FCreateOpState", CreateCachedOutputState)
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     mxnet::ShapeVector *in_shapes,
     mxnet::ShapeVector *out_shapes) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return DefaultSubgraphOpShapeHelper(op->GetForwardSym(), in_shapes, out_shapes);
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int> *in_types,
     std::vector<int> *out_types) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return DefaultSubgraphOpTypeHelper(op->GetForwardSym(), in_types, out_types);
  })
.set_attr<FInferStorageType>("FInferStorageType",
  [](const nnvm::NodeAttrs& attrs,
     const int dev_mask,
     DispatchMode* dispatch_mode,
     std::vector<int>* in_stypes,
     std::vector<int>* out_stypes) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return DefaultSubgraphOpStorageTypeHelper(op->GetForwardSym(),
                                                  dev_mask, dispatch_mode,
                                                  in_stypes, out_stypes);
  })
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", CachedOutputForward)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", CachedOutputForward)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return DefaultSubgraphOpMutableInputsHelper(op->GetForwardSym());
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return DefaultSubgraphOpResourceRequestHelper(op->GetForwardSym());
  })
.set_attr<FExecType>("FExecType", DefaultSubgraphOpExecType)
.add_argument("data", "NDArray-or-Symbol[]", "input data list");

}  // namespace op
}  // namespace mxnet
