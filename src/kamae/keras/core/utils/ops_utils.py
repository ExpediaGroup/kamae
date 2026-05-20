# Copyright [2024] Expedia, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-backend operation utilities for backend-agnostic layers.

Provides common operations that aren't directly available in keras.ops.
"""

import math

import keras
from keras import KerasTensor, ops


def divide_no_nan(x: KerasTensor, y: KerasTensor) -> KerasTensor:
    """
    Multi-backend safe division that returns 0 where y == 0.

    This is a backend-agnostic equivalent of tf.math.divide_no_nan.
    Instead of returning NaN or Inf when dividing by zero, returns 0.

    :param x: Numerator tensor
    :param y: Denominator tensor
    :returns: Result of x / y, with 0 where y == 0
    """
    is_zero = ops.equal(y, ops.convert_to_tensor(0.0, dtype=y.dtype))
    return ops.where(is_zero, ops.zeros_like(x), ops.divide(x, y))


def get_radians(degrees: KerasTensor) -> KerasTensor:
    """
    Converts degrees tensor to radians. We need to cast to float64 otherwise
    pi / 180 will lose precision.

    :param degrees: Tensor of degrees.
    :returns: Tensor of radians.
    """
    return ops.cast(degrees, dtype="float64") * ops.convert_to_tensor(
        math.pi / 180, dtype="float64"
    )


def get_degrees(radians: KerasTensor) -> KerasTensor:
    """
    Converts radians tensor to degrees.

    :param radians: Tensor of radians.
    :returns: Tensor of degrees.
    """
    return ops.cast(radians, dtype="float64") * ops.convert_to_tensor(
        180 / math.pi, dtype="float64"
    )


def l2_normalize(x: KerasTensor, axis: int, epsilon: float = 1e-12) -> KerasTensor:
    """
    L2 normalize a tensor along a specified axis.

    This is a backend-agnostic implementation of L2 normalization:
    normalized = x / sqrt(sum(x^2))

    :param x: Input tensor to normalize.
    :param axis: Axis along which to normalize.
    :param epsilon: Small constant to avoid division by zero.
    :returns: L2-normalized tensor.
    """
    square_sum = ops.sum(ops.square(x), axis=axis, keepdims=True)
    norm = ops.sqrt(
        ops.maximum(square_sum, ops.convert_to_tensor(epsilon, dtype=x.dtype))
    )
    return x / norm
