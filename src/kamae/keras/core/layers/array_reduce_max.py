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

from typing import Any, Dict, List, Optional

import keras
from keras import ops

import kamae
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import ParamSpec


class ArrayReduceMaxLayer(BaseLayer):
    """
    Reduces the last dimension of a tensor by taking the maximum.

    Input:  (..., N)
    Output: (...)

    NaN values in the result are replaced with the configured default_value.
    """

    jit_compatible = True

    _compatible_dtypes = [
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ]

    _params = {
        "default_value": ParamSpec(
            default=0.0,
            doc="Value to use when result is NaN",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        result = ops.max(inputs, axis=-1)
        return ops.where(
            ops.isnan(result),
            ops.cast(self.default_value, dtype=result.dtype),
            result,
        )
