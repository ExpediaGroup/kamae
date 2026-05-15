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


class LogLayer(BaseLayer):
    """
    Performs the log(alpha + x) operation on a given input tensor.
    """

    jit_compatible = True

    _compatible_dtypes = [
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]
    _params = {
        "alpha": ParamSpec(default=0.0, doc="Alpha value in log(alpha + x)"),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the log(alpha + x) operation on a given input tensor


        :param inputs: Input tensor to perform the log(alpha + x) operation on.
        :returns: The input tensor with the log(alpha + x) operation applied.
        """
        return ops.log(ops.add(inputs, self.alpha))
