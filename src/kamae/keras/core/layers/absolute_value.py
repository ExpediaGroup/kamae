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


class AbsoluteValueLayer(BaseLayer):
    """
    Performs the abs(x) operation on a given input tensor.
    """

    jit_compatible = True

    _compatible_dtypes = [
        "float16",
        "float32",
        "float64",
        "int32",
        "int64",
        "complex64",
        "complex128",
    ]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the abs(x) operation on a given input tensor.


        :param inputs: Tensor to perform the abs(x) operation on.
        :returns: The absolute value of the input tensor.
        """
        return ops.absolute(inputs)
