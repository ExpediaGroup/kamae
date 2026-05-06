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

from keras import ops

from kamae.keras.core.typing import Tensor


def divide_no_nan(x: Tensor, y: Tensor) -> Tensor:
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
