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
class StringReplaceLayer(BaseLayer):
    """
    StringReplaceLayer layer for TensorFlow.
    """

    def __init__(
        self,
        string_match_constant: Optional[str] = None,
        string_replace_constant: Optional[str] = None,
        regex: bool = False,
        name: Optional[str] = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the StringReplaceLayer layer.

        WARNING: While it works, the use of tensors in matching/replacement
        is not recommended due to the complexity of the regex matching which requires
        use of a map_fn. This will be comparatively VERY slow and may not be suitable
        for inference use-cases.
        If you know where in the string the match is, you will be much
        better off slicing the string and checking for equality.

        :param string_match_constant: The string to match against and replace.
            Defaults to `None`.
        :param string_replace_constant: The string to replace the matched string with.
            Defaults to `None`.
        :param regex: Whether to treat the string match as a regular expression.
            Defaults to `False`. In the case regex is enabled, the string_match_constant
            or second input tensor elements are treated as a regex pattern. Please be
            aware that while testing has tried to catch corner cases, this is not
            guaranteed to be bug-free due to slight differences in the regex
            implementations between Spark and TensorFlow.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.string_match_constant = string_match_constant
        self.string_replace_constant = string_replace_constant
        self.regex = regex

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
        Checks for the existence of a substring/pattern within a tensor and replaces
        if there is a match.

        KNOWN ISSUE: when replacing with a string that contains a backslash,
        the backslash must be double escaped (\\\\) in order to be added properly.
        This is consistent in both spark and tensorflow components.

        WARNING: While it works, the use of tensors in matching/replacement
        is not recommended due to the complexity of the regex matching which requires
        use of a map_fn. This will be comparatively VERY slow and may not be suitable
        for inference use-cases.
        If you know where in the string the match is, you will be much
        better off slicing the string and checking for equality.

        Decorated with `@allow_single_or_multiple_tensor_input` to ensure that the input
        is either a single tensor or an iterable of tensors. Returns this result as a
        list of tensors for easier use here.

        :param: inputs: A string tensor or iterable of up to three string
            tensors.
            In the case multiple tensors are passed, require that the order of inputs is
             [string input, {string match tensor}, {string replace tensor}].
        :returns: A string tensor of regex replaced strings.
        """

        match_all_pattern = r"([\w]\\+\_+\!+\?+)*"

        # Case both match and replacement are constant
        if (
            self.string_replace_constant is not None
            and self.string_match_constant is not None
        ):
            if len(inputs) == 1:
                # Need the tensor for shapes to be consistent
                input_tensor = inputs[0]

                match_substring = self.string_match_constant

                if not self.regex:
                    match_substring = self._escape_special_characters(match_substring)

                # Calls regex replace function on the input tensor, matching
                # with match constant and replacing with replace constant
                replaced_tensor = tf.strings.regex_replace(
                    input_tensor,
                    tf.constant(
                        match_all_pattern + match_substring + match_all_pattern
                        if match_substring != ""
                        else "^$"
                    ),
                    tf.constant(self.string_replace_constant),
                )

            else:
                raise ValueError(
                    """When string_match_constant and string_replace_constant are
                    defined, expected a single tensor as input."""
                )
        else:
            # Preserve input shape
            input_shape = tf.shape(inputs[0])
            # Generate a tensor that can be used by map_fn
            # First we define 3 tensors, the input string, the match string and the
            # replace string
            string_tensor = inputs[0]
            match_substring = (
                tf.constant(self.string_match_constant, shape=string_tensor.shape)
                if self.string_match_constant is not None
                else inputs[1]
            )
            replace_substring = (
                tf.constant(self.string_replace_constant, shape=string_tensor.shape)
                if self.string_replace_constant is not None
                else inputs[1 + (len(inputs) == 3)]
            )

            # Stack the input, match and replace elements into a single tensor
            # then flatten for use in map_fn
            mappable_tensor = tf.stack(
                [string_tensor, match_substring, replace_substring], axis=-1
            )
            mappable_tensor = tf.reshape(mappable_tensor, [-1, 3])

            def _tensor_replace(x):
                match_substring = x[1]
                if not self.regex:
                    match_substring = self._escape_special_characters(x[1])
                return tf.strings.regex_replace(
                    input=x[0],
                    pattern=match_all_pattern + match_substring + match_all_pattern
                    if match_substring != ""
                    else "^$",
                    rewrite=x[2],
                )

            # TODO: tf.vectorized_map may be slightly faster with larger batches
            #  but this requires some refactoring
            replaced_tensor = tf.map_fn(
                _tensor_replace,
                elems=mappable_tensor,
                dtype=tf.string,
            )

            # Reshape to the preserved input shape
            replaced_tensor = tf.reshape(replaced_tensor, input_shape)

        return replaced_tensor

    def _escape_special_characters(
        self, string_to_escape: Union[str, Tensor]
    ) -> Union[str, Tensor]:
        """
        Escapes special characters in a string so they are not parsed as regex.
        :param string: The string or string tensor to escape special characters in.
        :returns: The escaped string or string tensor.
        """

        for char in [
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
            if isinstance(string_to_escape, str):
                string_to_escape = string_to_escape.replace(char, "\\\\" + char)
            else:
                string_to_escape = tf.strings.regex_replace(
                    string_to_escape, "\\" + char, "\\\\" + char
                )
        return string_to_escape

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringReplace layer.
        Used for saving and loading the layer from disk.

        Specifically, `regex`, `string_match_constant` and `string_replace_constant`
        are added to the config.

        :returns: Dictionary configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "regex": self.regex,
                "string_match_constant": self.string_match_constant,
                "string_replace_constant": self.string_replace_constant,
            }
        )
        return config
