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

from typing import Any, Dict, List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class StringMapLayer(BaseLayer):
    """
    StringMapLayer layer for TensorFlow.
    """

    def __init__(
        self,
        string_match_values: List[str],
        string_replace_values: List[str],
        default_replace_value: str = None,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the StringMapLayer layer.

        :param string_match_values: The list of strings to match against.
        :param string_replace_values: The list of strings to replace the matched
        strings with.
        :param default_replace_value: The default value to replace the unmatched
        strings with. If None, the original string is kept unchanged.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.string_match_values = string_match_values
        self.string_replace_values = string_replace_values
        self.default_replace_value = default_replace_value

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Checks if the input tensor is matching any of the string_match_values
        and replaces it with the corresponding string_replace_values.

        If default_replace_value is set, it will replace the unmatched strings
        with the default_replace_value. If default_replace_value is None, the
        original string is kept unchanged.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param: inputs: Input string tensor.
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

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringMapLayer layer.
        Used for saving and loading the layer from disk.

        Specifically, `string_match_values` and `string_replace_values`
        are added to the config.

        :returns: Dictionary configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "string_match_values": self.string_match_values,
                "string_replace_values": self.string_replace_values,
                "default_replace_value": self.default_replace_value,
            }
        )
        return config
