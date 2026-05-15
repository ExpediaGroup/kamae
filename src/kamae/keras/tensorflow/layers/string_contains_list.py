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


class StringContainsListLayer(BaseLayer):
    """
    Performs a string contains operation on the input tensor over entries in
    the string constant list.

    This implementation does not support matching of newline characters or empty
    strings.
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
        Checks for the existence of any substring in the string_contains_list
        within a tensor.


        :param inputs: Input string tensor.
        :returns: A boolean tensor indicating whether any of the string constants are
        matched.
        """
        match_substring = "|".join(
            [
                "(.*" + self._escape_special_characters(x) + ".*)"
                for x in self.string_constant_list
            ]
        )
        matched_tensor = tf.strings.regex_full_match(
            inputs,
            match_substring,
        )

        output_tensor = (
            tf.math.logical_not(matched_tensor) if self.negation else matched_tensor
        )

        return output_tensor

    def _escape_special_characters(self, string: str) -> str:
        """
        Escapes special characters in a string so they are not parsed as regex.
        :param string: The string or string tensor to escape special characters in.
        :returns: The escaped string or string tensor.
        """
        escaped_string = string
        for char in [
            "\\",
            ".",
            "^",
            "$",
            "*",
            "+",
            "?",
            "{",
            "}",
            "[",
            "]",
            "(",
            ")",
            "|",
        ]:
            if isinstance(escaped_string, str):
                escaped_string = escaped_string.replace(char, "\\" + char)
            else:
                escaped_string = tf.strings.regex_replace(
                    escaped_string, "\\" + char, "\\" + char
                )
        return escaped_string
