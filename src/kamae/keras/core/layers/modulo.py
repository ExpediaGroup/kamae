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

from typing import Any, Iterable, Union

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input
from kamae.params import ParamSpec


class ModuloLayer(BaseLayer):
    """
    Performs the modulo(x, y) operation on a given input tensor.
    If divisor is not set, inputs are assumed to be a list of two tensors and the
    first tensor is modulo'd by the second.
    If divisor is set, inputs must be a tensor.
    """

    jit_compatible = True

    _compatible_dtypes = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ]
    _params = {
        "divisor": ParamSpec(default=None, doc="The divisor to modulo by"),
    }

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs: Any) -> Tensor:
        """
        Performs the modulo(x, y) operation on either an iterable of input tensors or
        a single input tensor and a constant.


        :param inputs: Single tensor or iterable of tensors to perform the
        modulo(x, y) operation on.
        :returns: The tensor resulting from the modulo(x, y) operation.
        """
        if self.divisor is not None:
            if len(inputs) > 1:
                raise ValueError("If divisor is set, cannot have multiple inputs")
            cast_input, cast_divisor = self._force_cast_to_compatible_numeric_type(
                inputs[0], self.divisor
            )
            return ops.mod(cast_input, cast_divisor)
        else:
            if len(inputs) != 2:
                raise ValueError("If divisor is not set, must have exactly 2 inputs")
            return ops.mod(inputs[0], inputs[1])
