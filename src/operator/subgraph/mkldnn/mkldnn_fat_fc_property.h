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
 * \file mkldnn_fat_fc_property.h
 * \brief Partition gragph property for fat FullyConnected operator
*/

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FAT_FC_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FAT_FC_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../common.h"
#include "../subgraph_property.h"

namespace mxnet {
namespace op {

class SgMKLDNNFatFCSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

 private:
  const nnvm::Node *data_ = nullptr;
  std::vector<const nnvm::Node *> matched_fc_;

 public:
  explicit SgMKLDNNFatFCSelector() {}

  bool Select(const nnvm::Node &n) override {
    if (n.op() == Op::Get("FullyConnected")) {
      matched_fc_.clear();
      matched_fc_.push_back(&n);
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (&n == matched_fc_[0] && !data_) {
      data_ = &new_node;
      return true;
    }
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (&n == data_ && new_node.op() == Op::Get("FullyConnected")) {
      FullyConnectedParam param = nnvm::get<FullyConnectedParam>(matched_fc_[0]->attrs.parsed);
      FullyConnectedParam new_param = nnvm::get<FullyConnectedParam>(new_node.attrs.parsed);
      if (param.flatten == new_param.flatten && param.no_bias == new_param.no_bias) {
        matched_fc_.push_back(&new_node);
        return true;
      }
    }
    return false;
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (matched_fc_.size() > 1) {
      std::vector<nnvm::Node *> ret;
      for (auto i : matched_fc_) {
        auto non_const_i = const_cast<nnvm::Node *>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
            candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return ret;
    }
    return std::vector<nnvm::Node *>(0);
  }

  void Reset() override {
    CHECK_GE(matched_fc_.size(), 1);
    auto new_selector = SgMKLDNNFatFCSelector();
    new_selector.Select(*matched_fc_[0]);
    *this = new_selector;
  }

  const std::vector<const nnvm::Node *>& GetMatchedNode() { return matched_fc_; }
};

class SgMKLDNNFatFCProperty : public SubgraphProperty {
 public:
  SgMKLDNNFatFCProperty() {
  }

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN Fat FullyConnected optimization pass";
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FC_OPT", 0)) {
      LOG(INFO) << name << " is disabled.";
      return nullptr;
    }
    auto property = std::make_shared<SgMKLDNNFatFCProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    return property;
  }

  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const SubgraphSelectorPtr& subgraph_selector,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();
    // Remove duplicated output.
    nnvm::Symbol new_sym;
    const auto &selector_ptr = static_cast<SgMKLDNNFatFCSelector*>(subgraph_selector.get());
    const auto &matched_fc = selector_ptr->GetMatchedNode();
    for (const auto& sub_node : matched_fc) {
      bool find = false;
      for (const auto sym_out : sym.outputs) {
        if (sym_out.node.get() == sub_node) {
          new_sym.outputs.emplace_back(sym_out);
          find = true;
          break;
        }
      }
      CHECK_EQ(find, true);
    }
    std::ostringstream node_name;
    node_name << "sg_mkldnn_fat_fc_" << std::to_string(subgraph_id);;
    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_sg_mkldnn_fat_fully_connected");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(sym));
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNFatFCSelector>();
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      const auto &sym = n->attrs.subgraphs[0];
      bool find = false;
      for (size_t i = 0; i < sym->outputs.size(); i++) {
        if (entry_ptr->node.get() == sym->outputs[i].node.get()) {
          *entry_ptr = nnvm::NodeEntry{n, static_cast<uint32_t>(i), 0};
          find = true;
          break;
        }
      }
      CHECK_EQ(find, true);
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
