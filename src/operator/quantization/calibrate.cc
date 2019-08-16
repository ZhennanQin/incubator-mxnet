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
 *  Copyright (c) 2019 by Contributors
 * \file calibrate.cc
 * \brief
 */

#include "./calibrate-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(CalibrateParam);

static void GetMinMax(const TBlob& data, float* min, float* max) {
  auto in_ptr = data.dptr<float>();
  float data_min = mshadow::red::limits::MaxValue<float>();
  float data_max = mshadow::red::limits::MinValue<float>();
  auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  std::vector<float> data_maxs(nthreads, data_max);
  std::vector<float> data_mins(nthreads, data_min);
#pragma omp parallel for num_threads(nthreads)
  for (index_t i = 0; i < static_cast<index_t>(data.Size()); i++) {
    int tid = omp_get_thread_num();
    if (in_ptr[i] > data_maxs[tid]) data_maxs[tid] = in_ptr[i];
    if (in_ptr[i] < data_mins[tid]) data_mins[tid] = in_ptr[i];
  }
  for (index_t i = 0; i < nthreads; i++) {
    if (data_maxs[i] > data_max) data_max = data_maxs[i];
    if (data_mins[i] < data_min) data_min = data_mins[i];
  }
  *min = data_min;
  *max = data_max;
}

void NaiveCalibrate(const CalibrateParam& param, const TBlob& data, const TBlob& calib_min,
                    const TBlob& calib_max) {
  float data_min, data_max;
  GetMinMax(data, &data_min, &data_max);
  *calib_min.dptr<float>() = data_min;
  *calib_max.dptr<float>() = data_max;
}

// Given a discrete distribution (may have not been normalized to 1),
// smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
// corresponding amount off the non-zero values.
template<typename T>
std::vector<float> SmoothDistribution(const std::vector<T>& p, const float eps = 0.0001) {
  std::vector<size_t> is_zeros(p.size());
  std::vector<size_t> is_nonzeros(p.size());
  {
    auto it = p.begin();
    std::generate(is_zeros.begin(), is_zeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) == 0.f); });
  }
  {
    auto it = p.begin();
    std::generate(is_nonzeros.begin(), is_nonzeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) != 0.f); });
  }

  size_t n_zeros = std::accumulate(is_zeros.begin(), is_zeros.end(), 0);
  size_t n_nonzeros = p.size() - n_zeros;
  if (!n_nonzeros) {
    // The discrete probability distribution is malformed. All entries are 0.
    return std::vector<float>();
  }
  float eps1 = eps * static_cast<float>(n_zeros) / static_cast<float>(n_nonzeros);
  std::vector<float> ret(p.size());
  for (size_t i = 0; i < p.size(); i++) {
    ret[i] = p[i] + eps * is_zeros[i] - eps1 * is_nonzeros[i];
    if (ret[i] < 0) return std::vector<float>();
  }
  return ret;
}
static float ComputeEntropy(std::vector<float>& p, std::vector<float>& q) {
  CHECK_EQ(p.size(), q.size());
  float p_sum = std::accumulate(p.begin(), p.end(), 0.f);
  float q_sum = std::accumulate(q.begin(), q.end(), 0.f);
  for (auto& it : p) {
    it = it / p_sum;
  }

  for (auto& it : q) {
    it = it / q_sum;
  }
  float ret = 0;
  for (size_t i = 0; i < p.size(); i++) {
    CHECK(p[i] > 0 && q[i] > 0);
    if (p[i] && q[i]) ret += p[i] * std::log(p[i] / q[i]);
  }
  return ret;
}

template <typename T>
static inline void print_vec(std::string title, std::vector<T> vec) {
    std::cout << title;
  for (auto it = vec.begin(); it != vec.begin() + 10; ++it) {
    std::cout << static_cast<int>(*it) << " ";
  }
  std::cout << " ... ";
  for (auto it = vec.end() - 10; it != vec.end(); ++it) {
    std::cout << static_cast<int>(*it) << " ";
  }
  std::cout << std::endl;
}

void EntropyCalibrate(const CalibrateParam& param, const TBlob& data, const TBlob& calib_min,
                      const TBlob& calib_max) {
  auto in_ptr = data.dptr<float>();

  float data_min, data_max;
  GetMinMax(data, &data_min, &data_max);
  const float th = std::max(std::abs(data_min), std::abs(data_max));
  bool is_int = true;
  if (data_min >= 0 && param.quantized_dtype != QuantizeOutType::kInt8) {
    is_int = false;
  }
  const float search_begin = th / 3;
  const int ref_preci_ratio = 4;
  const int search_count = 128;
  std::vector<float> thresholds(search_count, 0.f);
  std::vector<float> divergence(search_count, 0.f);
  if (is_int) {
    const int num_quantized_bins = 255;
    const float search_step = (th - search_begin) / search_count;
    #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (index_t i = 0; i < search_count; i++) {
      const float threshold = search_begin + i * search_step;
      thresholds[i] = threshold;
      const float quantized_step = threshold * 2 / num_quantized_bins;
      const float step = quantized_step / ref_preci_ratio;
      const float new_th = std::ceil(th / quantized_step) * quantized_step;
      const int num_bins = new_th * 2 / step;
      std::vector<size_t> hist(num_bins, 0);
      for (index_t j = 0; j < static_cast<index_t>(data.Size()); j++) {
        const int bin_idx = (th + in_ptr[j]) / step;
        if (bin_idx >= num_bins)
          hist.back()++;
        else
          hist[bin_idx]++;
      }
      const int p_bin_idx_start = (num_bins - num_quantized_bins * ref_preci_ratio) / 2;
      const int p_bin_idx_stop = (num_bins + num_quantized_bins * ref_preci_ratio) / 2;

      std::vector<size_t> sliced_nd_hist(hist.begin() + p_bin_idx_start,
                                         hist.begin() + p_bin_idx_stop);
      CHECK_EQ(sliced_nd_hist.size(), num_quantized_bins * ref_preci_ratio);
      sliced_nd_hist[0] =
          std::accumulate(hist.begin(), hist.begin() + p_bin_idx_start, sliced_nd_hist[0]);
      sliced_nd_hist.back() =
          std::accumulate(hist.begin() + p_bin_idx_stop, hist.end(), sliced_nd_hist.back());

      std::vector<float> quantized_bins(num_quantized_bins, 0);
      for (index_t j = 0; j < num_quantized_bins; j++) {
        const int start = j * ref_preci_ratio;
        const int stop = (j + 1) * ref_preci_ratio;
        quantized_bins[j] =
            std::accumulate(sliced_nd_hist.begin() + start, sliced_nd_hist.begin() + stop, 0);
      }

      // expand quantized_bins into num_bins
      std::vector<float> q(num_bins, 0);
      for (index_t j = 0; j < num_quantized_bins; j++) {
        const int start = j * ref_preci_ratio;
        const int stop = (j + 1) * ref_preci_ratio;
        for (index_t k = p_bin_idx_start + start; k < p_bin_idx_start + stop; k++) {
          q[k] = quantized_bins[j] / ref_preci_ratio;
        }
      }
      q = SmoothDistribution(q);
      std::vector<float> p = SmoothDistribution(hist);
      if (!q.size()) {
        divergence[i] = std::numeric_limits<float>::infinity();
      } else {
        divergence[i] = ComputeEntropy(p, q);
      }
      // LOG(INFO) << "i: " << i << "  divergence: " << divergence[i];
    }
  } else {
    const float num_quantized_bins = 255.5;
    const float search_step = (th - search_begin) / search_count;
    const int p_bin_idx_stop = num_quantized_bins * ref_preci_ratio;
    #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (index_t i = 0; i < search_count; i++) {
      const float threshold = search_begin + i * search_step;
      thresholds[i] = threshold;
      const float quantized_step = threshold / num_quantized_bins;
      const float new_th = std::ceil(th / quantized_step) * quantized_step;
      const float step = quantized_step / ref_preci_ratio;
      const int num_bins = new_th / step;
      std::vector<size_t> hist(num_bins, 0);
      for (index_t j = 0; j < static_cast<index_t>(data.Size()); j++) {
        const int bin_idx = std::round(in_ptr[j] / step);
        if (bin_idx >= num_bins)
          hist.back()++;
        else
          hist[bin_idx]++;
      }
      CHECK(p_bin_idx_stop <= hist.size());
      std::vector<size_t> sliced_nd_hist(hist.begin(), hist.begin() + p_bin_idx_stop);
      sliced_nd_hist.back() =
          std::accumulate(hist.begin() + p_bin_idx_stop, hist.end(), sliced_nd_hist.back());

      std::vector<float> quantized_bins(std::ceil(num_quantized_bins), 0);
      for (index_t j = 0; j < std::ceil(num_quantized_bins); j++) {
        int start, stop;
        if (j == 0) {
          start = 0;
          stop = ref_preci_ratio / 2;
        } else {
          start = ref_preci_ratio / 2 + (j - 1) * ref_preci_ratio;
          stop = ref_preci_ratio / 2 + j * ref_preci_ratio;
        }
        CHECK(stop <= sliced_nd_hist.size());
        quantized_bins[j] =
            std::accumulate(sliced_nd_hist.begin() + start, sliced_nd_hist.begin() + stop, 0);
      }
      // expand quantized_bins into num_bins
      std::vector<float> q(num_bins, 0);
      for (index_t j = 0; j < num_quantized_bins; j++) {
        int start, stop, ratio;
        if (j == 0) {
          start = 0;
          stop = ref_preci_ratio / 2;
          ratio = ref_preci_ratio / 2;
        } else {
          start = ref_preci_ratio / 2 + (j - 1) * ref_preci_ratio;
          stop = ref_preci_ratio / 2 + j * ref_preci_ratio;
          ratio = ref_preci_ratio;
        }
        CHECK(stop <= q.size());
        for (index_t k = start; k < stop; k++) {
          q[k] = quantized_bins[j] / ratio;
        }
      }

      q = SmoothDistribution(q);
      std::vector<float> p = SmoothDistribution(hist);
      if (!q.size()) {
        divergence[i] = std::numeric_limits<float>::infinity();
      } else {
        divergence[i] = ComputeEntropy(p, q);
      }
      // LOG(INFO) << "i: " << i << "  divergence: " << divergence[i];
    }
  }

  size_t min_divergence_idx = 0;
  float min_divergence = mshadow::red::limits::MaxValue<float>();
  for (size_t i = 0; i < divergence.size(); i++) {
    if (divergence[i] < min_divergence) {
      min_divergence = divergence[i];
      min_divergence_idx = i;
    }
  }
  if (data_min >= 0 && param.quantized_dtype != QuantizeOutType::kInt8) {
    *calib_min.dptr<float>() = 0;
  } else {
    *calib_min.dptr<float>() = -thresholds[min_divergence_idx];
  }
  *calib_max.dptr<float>() = thresholds[min_divergence_idx];
  LOG(INFO) << "min: " << data_min << " max: " << data_max
            << " th: " << thresholds[min_divergence_idx] << " divergence: " << min_divergence;
}

void CalibrateComputeCPU(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                         const std::vector<TBlob>& inputs, const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  const auto& param = nnvm::get<CalibrateParam>(attrs.parsed);
  if (param.mode == kNaive) {
    NaiveCalibrate(param, inputs[0], outputs[0], outputs[1]);
  } else if (param.mode == kEntropy) {
    EntropyCalibrate(param, inputs[0], outputs[0], outputs[1]);
  } else {
    LOG(FATAL) << "unrecognized calibration mode: " << param.mode;
  }
}

static inline bool CalibrateShape(const nnvm::NodeAttrs& attrs, std::vector<TShape>* in_attrs,
                                  std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  mxnet::TShape dshape = (*in_attrs)[0];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape(1, 1));
  return !shape_is_none(in_attrs->at(0));
}

static inline bool CalibrateType(const nnvm::NodeAttrs& attrs, std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  CHECK(in_attrs->at(0) == mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  return true;
}

NNVM_REGISTER_OP(_contrib_calibrate)
.describe(R"code(Provide calibrated min/max for input.

.. Note::
    This operator only supports forward propagation. DO NOT use it in training.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<CalibrateParam>)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"calib_min", "calib_max"};
})
.set_attr<mxnet::FInferShape>("FInferShape", CalibrateShape)
.set_attr<nnvm::FInferType>("FInferType", CalibrateType)
.set_attr<FCompute>("FCompute<cpu>", CalibrateComputeCPU)
.add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_arguments(CalibrateParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
