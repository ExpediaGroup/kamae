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


class ExponentLayer(BaseLayer):
    """
    Performs the x^exponent operation on a given input tensor.
    """

    jit_compatible = True

    _compatible_dtypes = [
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]
    _params = {
        "exponent": ParamSpec(
            default=None,
            doc="The exponent to raise the input to",
        ),
    }

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs: Any) -> Tensor:
        """
        Performs the x^exponent operation on a given input tensor.


        :param inputs: Single tensor or iterable of tensors to perform the x^pow
         operation on.
        :returns: The tensor raised to the power of the exponent.
        """
        if self.exponent is not None:
            if len(inputs) > 1:
                raise ValueError("If exponent is set, cannot have multiple inputs")
            return ops.power(
                inputs[0],
                ops.cast(self.exponent, dtype=inputs[0].dtype),
            )
        else:
            if not len(inputs) == 2:
                raise ValueError("If exponent is not set, must have exactly 2 inputs")
            return ops.power(inputs[0], inputs[1])
