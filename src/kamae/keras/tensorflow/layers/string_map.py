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


class StringMapLayer(BaseLayer):
    """
    StringMapLayer layer for TensorFlow.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "string_match_values": ParamSpec(
            default=_REQUIRED,
            doc="The list of strings to match against.",
        ),
        "string_replace_values": ParamSpec(
            default=_REQUIRED,
            doc="The list of strings to replace the matched strings with.",
        ),
        "default_replace_value": ParamSpec(
            default=None,
            doc="The default value to replace the unmatched strings with. If None, the original string is kept unchanged.",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Checks if the input tensor is matching any of the string_match_values
        and replaces it with the corresponding string_replace_values.

        If default_replace_value is set, it will replace the unmatched strings
        with the default_replace_value. If default_replace_value is None, the
        original string is kept unchanged.


        :param inputs: Input string tensor.
        :returns: A string tensor with the matched strings replaced.
        """

        # Iterate through each match/replace pair
        output_tensor = inputs
        for match_value, replace_value in zip(
            self.string_match_values, self.string_replace_values
        ):
            output_tensor = tf.where(
                tf.equal(output_tensor, match_value), replace_value, output_tensor
            )

        # Handle the default replacement for unmatched strings
        # Chain tf.logical_and for each match to check if there is no match
        if self.default_replace_value is not None:
            matches = self.string_match_values
            unmatched_condition = tf.not_equal(inputs, matches[0])
            if len(matches) > 1:
                for match in matches[1:]:
                    unmatched_condition = tf.logical_and(
                        unmatched_condition,
                        tf.not_equal(inputs, match),
                    )
            expected_dtype = output_tensor.dtype
            default_val = tf.constant(self.default_replace_value, dtype=expected_dtype)
            output_tensor = tf.where(unmatched_condition, default_val, output_tensor)

        return output_tensor
