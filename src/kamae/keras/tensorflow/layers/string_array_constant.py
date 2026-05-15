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

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import ParamSpec


class StringArrayConstantLayer(BaseLayer):
    """
    Tensorflow keras layer that outputs a constant string array.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = None
    _params = {
        "constant_string_array": ParamSpec(
            default=None,
            doc="The constant string array to output",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Returns the constant string array with the same shape as the input tensor.


        :param inputs: Tensor to replicate shape of for constant string array.
        :returns: A tensor with the constant string array
        """
        input_shape = tf.shape(inputs)
        string_tensor = tf.constant(self.constant_string_array)
        broadcast_shape = tf.concat(
            [input_shape[:-1], [tf.size(string_tensor)]], axis=0
        )
        broadcasted_strings = tf.broadcast_to(string_tensor, broadcast_shape)
        return broadcasted_strings
