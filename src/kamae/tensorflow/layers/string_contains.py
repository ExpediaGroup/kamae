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

from typing import Any, Dict, Iterable, List, Optional, Union

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import allow_single_or_multiple_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class StringContainsLayer(BaseLayer):
    """
    Performs a string contains operation on the input tensor,
    matching against a string constant or element-wise against a second input tensor.
    WARNING: While it works, the use of tensors in matching/replacement
        is not recommended due to the complexity of the regex matching which requires
        use of a map_fn. This will be comparatively VERY slow and may not be suitable
        for inference use-cases.
        If you know where in the string the match is, you will be much
        better off slicing the string and checking for equality.
    This implementation will only match an empty string with another empty string and
    does not support matching of newline characters.
    """

    def __init__(
        self,
        string_constant: Optional[str] = None,
        negation: bool = False,
        name: Optional[str] = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the StringContainsLayer layer.
        :param string_constant: The string to match against. Defaults to `None`.
        :param negation: Whether to negate the output. Defaults to `False`.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.negation = negation
        self.string_constant = string_constant

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs) -> Tensor:
        """
        Checks for the existence of a substring/pattern within a tensor.
        WARNING: While it works, the use of tensors in matching
        is not recommended due to the complexity of the regex matching which requires
        use of a map_fn. This will be comparatively VERY slow and may not be suitable
        for inference use-cases.
        If you know where in the string the match is, you will be much
        better off slicing the string and checking for equality.

        Decorated with `@allow_single_or_multiple_tensor_input` to ensure that the input
        is either a single tensor or an iterable of tensors. Returns this result as a
        list of tensors for easier use here.

        :param: inputs: A string tensor or iterable of up to two string tensors.
            In the case two tensors are passed, require that the first tensor is the
            tensor to match a pattern/substring against.
        :returns: A boolean tensor whether the string/string elements are matched.
        """

        match_all_pattern = ".*"

        # Checking input
        if self.string_constant is not None:
            if len(inputs) == 1:
                # To preserve shape, need to pass tensor to regex_full_match
                input_tensor = inputs[0]

                match_substring = self.string_constant
                match_substring = self._escape_special_characters(match_substring)
                matched_tensor = tf.strings.regex_full_match(
                    input_tensor,
                    tf.constant(
                        match_all_pattern + match_substring + match_all_pattern
                        if match_substring != ""
                        else "^$"
                    ),
                )
            else:
                raise ValueError(
                    "With string_constant defined, expected a single tensor as input."
                )
        else:
            if len(inputs) != 2:
                raise ValueError(
                    "Expected iterable of tensors of length 2, \
                     or string_constant to be defined."
                )

            # Two tensors provided
            @tf.function
            def tensor_match(x):
                match_substring = x[1]
                match_substring = self._escape_special_characters(match_substring)
                return tf.strings.regex_full_match(
                    x[0],
                    match_all_pattern + match_substring + match_all_pattern
                    if x[1] != ""
                    else "^$",
                )

            # Stack inputs to match element-wise with map_fn
            # Requires ordering of inputs to be correct
            stacked_inputs = tf.stack(inputs, axis=-1)
            input_shape = tf.shape(inputs[0])

            mappable_tensor = tf.reshape(stacked_inputs, [-1, 2])

            # Apply element-wise matching
            # TODO: tf.vectorized_map may be slightly faster with larger batches
            #  but this requires some refactoring
            matched_tensor = tf.map_fn(
                fn=tensor_match, elems=mappable_tensor, dtype=tf.bool
            )

            matched_tensor = tf.reshape(matched_tensor, input_shape)

        output_tensor = (
            tf.math.logical_not(matched_tensor) if self.negation else matched_tensor
        )

        return output_tensor

    def _escape_special_characters(
        self, string: Union[str, Tensor]
    ) -> Union[str, Tensor]:
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

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringContains layer.
        Used for saving and loading from a model.

        Specifically adds the string_constant and negation parameters to the config
        dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {"string_constant": self.string_constant, "negation": self.negation}
        )
        return config
