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

from functools import reduce
from typing import Any, Iterable, Union

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input
from kamae.keras.core.utils.ops_utils import divide_no_nan
from kamae.params import ParamSpec


class DivideLayer(BaseLayer):
    """
    Performs the divide(x, y) operation on a given input tensor. If divisor is not set,
    inputs must be a list. If divisor is set, inputs must be a tensor.
    """

    jit_compatible = True

    _compatible_dtypes = [
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ]
    _params = {
        "divisor": ParamSpec(
            default=None,
            doc="The divisor to divide the input by",
        ),
    }

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs: Any) -> Tensor:
        """
        Performs the divide(x, y) operation on either an iterable of input tensors or
        a single input tensor and a constant.


        :param inputs: Single tensor or iterable of tensors to perform the
        divide(x, y) operation on.
        :returns: The tensor resulting from the divide(x, y) operation.
        """
        if self.divisor is not None:
            if len(inputs) > 1:
                raise ValueError("If divisor is set, cannot have multiple inputs")
            divisor_tensor = ops.cast(self.divisor, dtype=inputs[0].dtype)
            return divide_no_nan(inputs[0], divisor_tensor)
        else:
            if not len(inputs) > 1:
                raise ValueError("If divisor is not set, must have multiple inputs")
            return reduce(divide_no_nan, inputs)
