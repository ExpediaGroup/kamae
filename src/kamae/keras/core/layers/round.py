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

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import ParamSpec


class RoundLayer(BaseLayer):
    """
    Performs a standard rounding operation on the input tensor.
    Supported rounding types are 'ceil', 'floor' and 'round'.

    - 'ceil' rounds up to the nearest integer.
    - 'floor' rounds down to the nearest integer.
    - 'round' rounds to the nearest integer.
    """

    jit_compatible = True

    _compatible_dtypes = ["float16", "float32", "float64"]
    _params = {
        "round_type": ParamSpec(
            default="round",
            doc="The type of rounding to perform: 'ceil', 'floor', or 'round'",
        ),
    }

    def _post_init(self):
        if self.round_type not in ["ceil", "floor", "round"]:
            raise ValueError("roundType must be one of 'ceil', 'floor' or 'round'.")

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the rounding operation on the input tensor.

        :param inputs: Input tensor to perform the rounding on.
        :returns: The input tensor with the rounding applied.
        """
        if self.round_type == "ceil":
            return ops.ceil(inputs)
        elif self.round_type == "floor":
            return ops.floor(inputs)
        elif self.round_type == "round":
            return ops.round(inputs)
        else:
            raise ValueError("roundType must be one of 'ceil', 'floor' or 'round'.")
