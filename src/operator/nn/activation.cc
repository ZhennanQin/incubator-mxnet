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
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu, Da Zheng
*/
#include "./activation-inl.h"
#include "../mshadow_op.h"
#include "../tensor/elemwise_unary_op.h"
#include "../operator_common.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

namespace activation {

int GradNumInputs(int act_type) {
    // check activation.cu \sa ActivationGradCompute
    switch (act_type) {
        case kReLU:
            return 2;
        case kSoftReLU:
        case kSoftSign:
        case kTanh:
        case kSigmoid:
            return 3;
        default:
            CHECK(false) << "missing activation type";
    }
    // unreachable
    return -1;
}

}  // namespace activation

DMLC_REGISTER_PARAMETER(ActivationParam);

// This will determine the order of the inputs for backward computation.
struct ActivationGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    // ograds, output...
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.emplace_back(nnvm::NodeEntry{n, activation::kOut, 0});

    const NodeAttrs& attrs = n->attrs;
    using namespace activation;
    int act_type = dmlc::get<ActivationParam>(attrs.parsed).act_type;
    // for ReLU, no need to pass input data. This enables inplace optimization during the
    // forward pass.
    // check activation.cu \sa ActivationGradCompute
    switch (act_type) {
        case kReLU:
            break;
        case kSoftReLU:
        case kSoftSign:
        case kTanh:
        case kSigmoid:
            heads.push_back(n->inputs[activation::kData]);
            break;
        default:
            CHECK(false) << "missing activation type";
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};


MXNET_OPERATOR_REGISTER_UNARY(Activation)
.describe(R"code(Applies an activation function element-wise to the input.

The following activation functions are supported:

- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
- `softsign`: :math:`y = \frac{x}{1 + abs(x)}`

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<FCompute>("FCompute<cpu>", ActivationCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ActivationGrad{"_backward_Activation"})
.add_arguments(ActivationParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Activation)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const int act_type = dmlc::get<ActivationParam>(attrs.parsed).act_type;
    return activation::GradNumInputs(act_type);
})
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<-1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<FCompute>("FCompute<cpu>", ActivationGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
