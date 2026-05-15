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

from typing import Any, Iterable

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_multiple_tensor_input
from kamae.params import ParamSpec


class StringConcatenateLayer(BaseLayer):
    """
    Performs a concatenation of the input tensors.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "separator": ParamSpec(
            default="_",
            doc="The separator to use when joining the input tensors.",
        ),
    }

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        """
        Concatenates the input tensors.

        Decorated with `@enforce_multiple_tensor_input` to ensure that the input is an
        iterable of multiple tensors. Raises an error if a single tensor is passed in.

        :param inputs: Input tensors that will be concatenated on the last axis.
        Must be string tensors.
        :returns: A tensor with the concatenated values - same shape as each of
        the input tensors.
        """
        return tf.strings.join(inputs, separator=self.separator)
