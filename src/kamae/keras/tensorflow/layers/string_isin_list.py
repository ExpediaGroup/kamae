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
from kamae.params import _REQUIRED, ParamSpec


class StringIsInListLayer(BaseLayer):
    """
    Performs a string isin operation on the input tensor over entries in
    the string constant list.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "string_constant_list": ParamSpec(
            default=_REQUIRED,
            doc="The list of strings to match against.",
        ),
        "negation": ParamSpec(
            default=False,
            doc="Whether to negate the output.",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Checks if the input tensor is matching any string in the constant_string_array.


        :param inputs: Input string tensor.
        :returns: A boolean tensor indicating whether any of the string is matched.
        """
        strings = tf.constant(self.string_constant_list)
        tile_multiples = tf.concat(
            [tf.ones(tf.rank(inputs), dtype=tf.int32), tf.shape(strings)],
            axis=0,
        )
        x_tile = tf.tile(tf.expand_dims(inputs, -1), tile_multiples)
        matched_tensor = tf.reduce_any(tf.equal(x_tile, strings), -1)
        output_tensor = (
            tf.math.logical_not(matched_tensor) if self.negation else matched_tensor
        )
        return output_tensor
