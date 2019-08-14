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
 * \file calibraite-inl.h
 * \brief Implementation of calibrate operator
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_CALIBRATE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_CALIBRATE_INL_H_

#include <limits>
#include <vector>
#include "../mxnet_op.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

enum CalibrationMode {
  kNaive = 0,
  kEntropy,
};

struct CalibrateParam : public dmlc::Parameter<CalibrateParam> {
  int quantized_dtype;
  int mode;
  DMLC_DECLARE_PARAMETER(CalibrateParam) {
    DMLC_DECLARE_FIELD(quantized_dtype)
      .add_enum("auto", QuantizeOutType::kAuto)
      .add_enum("int8", QuantizeOutType::kInt8)
      .add_enum("uint8", QuantizeOutType::kUint8)
      .set_default(QuantizeOutType::kInt8)
      .describe(
          "Quantized data type. `auto` can be specified to automatically determine quantized "
          "type according to min_calib_range.");
    DMLC_DECLARE_FIELD(mode)
      .add_enum("naive", CalibrationMode::kNaive)
      .add_enum("entropy", CalibrationMode::kEntropy)
      .set_default(CalibrationMode::kNaive)
      .describe(
          "Calibration mode. `naive` is to collect acutal min/max value. `entropy` is to collect "
          "with KL-divergence");
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_CALIBRATE_INL_H_
