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
from kamae.params.shared_specs import MASK_VALUE_PARAMS


class StandardScaleLayer(NormalizeLayer):
    """
    Performs the standard scaling of the input.

    This layer will shift and scale inputs into a distribution centered around
    0 with standard deviation 1. It accomplishes this by precomputing the mean
    and variance of the data, and calling `(input - mean) / sqrt(var)` at
    runtime. mask_value is used to ignore certain values in the standard scaling
    process. They will remain the same value in the output value as they were in
    the input value.
    """

    jit_compatible = True

    _params = {**MASK_VALUE_PARAMS}

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs normalization on the input tensor(s). It ignores values which
        are equal to the mask_value.

        :param inputs: Input tensor to perform the normalization on.
        :returns: The input tensor with the normalization applied.
        """
        input_dtype_str = keras.backend.standardize_dtype(inputs.dtype)
        mean = self._cast(self.mean, input_dtype_str)
        variance = self._cast(self.variance, input_dtype_str)

        numerator = ops.subtract(inputs, mean)
        denominator = ops.maximum(
            ops.sqrt(variance), ops.convert_to_tensor(1e-8, dtype=inputs.dtype)
        )
        normalized_outputs = divide_no_nan(numerator, denominator)

        if self.mask_value is not None:
            mask = ops.equal(inputs, self.mask_value)
            normalized_outputs = ops.where(
                mask, inputs, self._cast(normalized_outputs, input_dtype_str)
            )
        return normalized_outputs
