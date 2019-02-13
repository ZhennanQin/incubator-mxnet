# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""tools for analyzing quantized models."""

from __future__ import absolute_import

try:
    from scipy import stats
except ImportError:
    stats = None

import ctypes
import logging
import os
import numpy as np
import mxnet as mx
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import NDArray
from ..io import DataIter
from ..context import cpu, Context
from ..module import Module


def compare_model_result(target_sym,
                         target_arg_params,
                         target_aux_params,
                         ref_sym,
                         ref_arg_params,
                         ref_aux_params,
                         data,
                         devs,
                         label_name,
                         num_iterations,
                         metrics,
                         logger=None):
    ref_mod = mx.mod.Module(
        symbol=ref_sym, context=devs, label_names=[
            label_name,
        ])
    ref_mod.bind(
        for_training=False,
        data_shapes=data.provide_data,
        label_shapes=data.provide_label)
    ref_mod.set_params(ref_arg_params, ref_aux_params)

    target_mod = mx.mod.Module(
        symbol=target_sym, context=devs, label_names=[
            label_name,
        ])
    target_mod.bind(
        for_training=False,
        data_shapes=data.provide_data,
        label_shapes=data.provide_label)
    target_mod.set_params(target_arg_params, target_aux_params)
    diff = []
    num = 0
    for batch in data:
        ref_mod.forward(batch, is_train=False)
        ref_output = ref_mod.get_outputs()[0]
        ref_label = ref_output.argmax(axis=1)
        target_mod.forward(batch, is_train=False)
        target_output = target_mod.get_outputs()[0]
        target_label = target_output.argmax(axis=1)
        ref_label_np = ref_label.asnumpy()
        target_label_np = target_label.asnumpy()
        eq_np = np.equal(ref_label_np, target_label_np)
        for idx, val in enumerate(eq_np):
            if val == False:
                real_label = batch.label[0].asnumpy()
                diff.append([
                    batch.index[idx], ref_label_np[idx], target_label_np[idx],
                    real_label[idx]
                ])
                logger.info(
                    "result mismatch: image_index: %d, ref_label:%d, target_label:%d, real_label:%d"
                    % (batch.index[idx], int(ref_label_np[idx]),
                       int(target_label_np[idx]), int(real_label[idx])))

        for m in metrics:
            target_mod.update_metric(m, [ref_label])
        num += 1
        if num_iterations is not None and num >= num_iterations:
            break
    return diff
