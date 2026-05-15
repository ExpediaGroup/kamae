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


class StringToStringListLayer(BaseLayer):
    """
    A layer that converts a string to a list of strings by splitting on a
    separator. It takes a default value and a list_length parameter to ensure that
    the output tensor has the correct shape.

    If the separator is empty, the string is split on bytes/characters.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "separator": ParamSpec(
            default=",",
            doc="The separator to use when splitting the strings.",
        ),
        "default_value": ParamSpec(
            default="",
            doc="The value to use when the input is empty.",
        ),
        "list_length": ParamSpec(
            default=1,
            doc="The length of the string list in the output tensor.",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Splits the input string tensor by the separator and returns the list of
        strings. A list_length parameter is used to ensure that the output tensor has a
        fixed shape. If the separator is empty, the string is split on bytes/characters.

        :param inputs: Input tensor.
        :returns: Tensor with the list of strings.
        """
        input_shape = inputs.get_shape().as_list()
        input_shape.append(self.list_length)
        # If the separator is empty, we split on bytes/characters.
        # Otherwise, we use the standard string split.
        ragged_strings_split = (
            tf.strings.split(inputs, sep=self.separator)
            if self.separator != ""
            else tf.strings.bytes_split(inputs)
        )
        split_strings_tensor = ragged_strings_split.to_tensor(
            default_value=self.default_value, shape=input_shape
        )

        # Replace empty strings with the default value
        split_strings_tensor = tf.where(
            tf.equal(split_strings_tensor, ""), self.default_value, split_strings_tensor
        )

        # If the dimension of the feature was 1, we squeeze it out
        # E.g. (None, None, 1) -> (None, None, 1, N) -> (None, None, N)
        # But (None, None, M) -> (None, None, M, N)
        return (
            tf.squeeze(split_strings_tensor, axis=-2)
            if input_shape[-2] == 1
            else split_strings_tensor
        )
