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

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input


class IdentityLayer(BaseLayer):
    """
    Performs an identity transform on the input tensor.
    """

    jit_compatible = True

    _compatible_dtypes = None

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs an identity transform on the input tensor.


        :param inputs: Tensor to apply the identity transform to.
        :returns: The input tensor unchanged.
        """
        # For identity, simply return the input unchanged
        # Note: keras.ops.identity() exists but has bugs in TensorFlow backend
        return inputs
