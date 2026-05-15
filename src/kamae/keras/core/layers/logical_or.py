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
from typing import Any, Iterable

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_multiple_tensor_input


class LogicalOrLayer(BaseLayer):
    """
    Performs the or(x, y) operation on a given input tensor.
    """

    jit_compatible = True

    _compatible_dtypes = ["bool"]

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        """
        Performs the or(x, y) operation on an iterable of input tensors


        :param inputs: Iterable of tensors to perform the or(x, y) operation on.
        :returns: The tensor resulting from the or(x, y) operation.
        """
        if len(inputs) == 1:
            raise ValueError("Expected multiple inputs, but got a single input")
        return reduce(ops.logical_or, inputs)
