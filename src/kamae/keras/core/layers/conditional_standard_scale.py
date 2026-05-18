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

from typing import Any

import keras
from keras import ops

from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.keras.core.utils.normalize_layer import NormalizeLayer
from kamae.keras.core.utils.ops_utils import divide_no_nan
from kamae.params import ParamSpec


class ConditionalStandardScaleLayer(NormalizeLayer):
    """
    Performs the standard scaling of the input with a masking condition.

    This layer will shift and scale inputs into a distribution centered around
    0 with standard deviation 1. It accomplishes this by precomputing the mean
    and variance of the data, and calling `(input - mean) / sqrt(var)` at
    runtime.

    The skip_zeros parameter allows to apply the standard scaling process
    only when input is not equal to zero. If equal to zero, it will remain zero in
    the output value as it was in the input value.
    """

    jit_compatible = True

    _params = {
        "skip_zeros": ParamSpec(
            default=False,
            doc="If True, do not apply the scaling when the values to scale are equal to zero",
        ),
        "epsilon": ParamSpec(
            default=0,
            doc="Small value to add to conditional check of zeros. Valid only when skipZeros is True",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs normalization on the input tensor(s).

        It applies the scaling only to values matching the mask condition, if set.
        It applies the scaling only to values not equal to zero, if skip_zeros is set.

        :param inputs: Input tensor to perform the normalization on.
        :returns: The input tensor with the normalization applied.
        """
        input_dtype_str = keras.backend.standardize_dtype(inputs.dtype)
        mean = self._cast(self.mean, input_dtype_str)
        variance = self._cast(self.variance, input_dtype_str)

        numerator = ops.subtract(inputs, mean)
        denominator = ops.maximum(
            ops.sqrt(variance), ops.convert_to_tensor(self.epsilon, dtype=inputs.dtype)
        )
        normalized_outputs = divide_no_nan(numerator, denominator)

        normalized_outputs = ops.where(
            ops.equal(variance, 0),
            ops.zeros_like(normalized_outputs),
            normalized_outputs,
        )

        if self.skip_zeros:
            eps = ops.convert_to_tensor(self.epsilon, dtype=inputs.dtype)
            normalized_outputs = ops.where(
                ops.less_equal(ops.abs(inputs), eps),
                ops.zeros_like(normalized_outputs),
                normalized_outputs,
            )
        return normalized_outputs
