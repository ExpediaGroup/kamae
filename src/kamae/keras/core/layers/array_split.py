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

from typing import Any, List

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import ParamSpec


class ArraySplitLayer(BaseLayer):
    """
    Performs a splitting of the input tensor into a list of tensors.
    Expands dimensions to ensure the output tensors are the same shape as the input.
    """

    jit_compatible = True

    _compatible_dtypes = None
    _params = {
        "axis": ParamSpec(default=-1, doc="Axis to split on"),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> List[Tensor]:
        """
        Splits the input tensor along the specified axis.

        :param inputs: Tensor to split.
        :returns: List of split tensors.
        """
        return [
            ops.expand_dims(y, axis=self.axis)
            for y in ops.unstack(inputs, axis=self.axis)
        ]
