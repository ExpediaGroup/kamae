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
import numpy as np
from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.keras.core.utils.tensor_utils import get_dtype_max
from kamae.params import ParamSpec


def _validate_decimals(value):
    if value < 0:
        raise ValueError("decimals must be greater than or equal to 0")
    return value


class RoundToDecimalLayer(BaseLayer):
    """
    Performs a rounding to the nearest decimal operation on the input tensor.

    If the specified number of decimals is too large for the input precision type,
    this operation can result in overflow. This is because the operation is performed by
    multiplying the input tensor by 10 to the power of the number of decimals, rounding
    the result to the nearest integer, and then dividing by 10 to the power of the
    number of decimals.
    """

    jit_compatible = True

    _compatible_dtypes = ["float16", "float32", "float64", "int32", "int64"]
    _params = {
        "decimals": ParamSpec(
            default=1,
            doc="The number of decimal places to round to",
            validator=_validate_decimals,
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the rounding operation on the input tensor.

        :param inputs: Input tensor to perform the rounding on.
        :returns: The input tensor with the rounding applied.
        """
        # WARNING: Depending on the type of the input and the number of decimals,
        # this multiplier could overflow.
        dtype_str = keras.backend.standardize_dtype(inputs.dtype)
        max_val = get_dtype_max(dtype_str)

        if 10**self.decimals > max_val:
            raise ValueError(
                """The number of decimals is too large for the input dtype.
                Overflow expected."""
            )
        multiplier = ops.cast(10**self.decimals, dtype=inputs.dtype)
        return ops.divide(ops.round(ops.multiply(inputs, multiplier)), multiplier)
