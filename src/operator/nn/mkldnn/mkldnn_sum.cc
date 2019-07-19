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
 * \file mkldnn_sum.cc
 * \brief
 * \author Da Zheng
*/
#include <iostream>

#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void MKLDNNSum(const mkldnn::memory &arr1,
               const mkldnn::memory &arr2,
               const mkldnn::memory &out) {
  std::vector<mkldnn::memory::desc> input_pds(2);
  std::vector<float> scales(2, 1);
  input_pds[0] = arr1.get_desc();
  input_pds[1] = arr2.get_desc();
  CHECK(input_pds[0] == input_pds[0]);
  const mkldnn::memory *in_mem1 = &arr1;
  const mkldnn::memory *in_mem2 = &arr2;
  auto output_pd = out.get_desc();
  if (input_pds[0] != output_pd) {
    auto tmp_memory1 = TmpMemMgr::Get()->Alloc(output_pd);
    auto tmp_memory2 = TmpMemMgr::Get()->Alloc(output_pd);
    mxnet::MKLDNNCopy(arr1, tmp_memory1);
    mxnet::MKLDNNCopy(arr2, tmp_memory2);
    input_pds[0] = tmp_memory1->get_desc();
    input_pds[1] = tmp_memory2->get_desc();
    in_mem1 = tmp_memory1;
    in_mem2 = tmp_memory2;
  }
  mkldnn::sum::primitive_desc sum_pd(output_pd, scales, input_pds, CpuEngine::Get()->get_engine());
  std::unordered_map<int, mkldnn::memory> args = {
    { MKLDNN_ARG_MULTIPLE_SRC, *in_mem1 },
    { MKLDNN_ARG_MULTIPLE_SRC + 1, *in_mem2 },
    { MKLDNN_ARG_DST, out },
  };
  MKLDNNStream::Get()->RegisterPrimArgs(mkldnn::sum(sum_pd), args);
}

}  // namespace op
}  // namespace mxnet
#endif
