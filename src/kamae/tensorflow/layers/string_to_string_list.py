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
class StringToStringListLayer(BaseLayer):
    """
    A layer that converts a string to a list of strings by splitting on a
    separator. It takes a default value and a list_length parameter to ensure that
    the output tensor has the correct shape.

    If the separator is empty, the string is split on bytes/characters.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        separator: str = ",",
        default_value: str = "",
        list_length: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialises the StringToStringListLayer layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param separator: The separator to use when joining the strings.
        Defaults to `","`.
        :param default_value: The value to use when the input is empty.
        Defaults to `""`.
        :param list_length: The length of the string list in the output tensor.
        Defaults to `1`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.separator = separator
        self.list_length = list_length
        self.default_value = default_value

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
        Splits the input string tensor by the separator and returns the list of
        strings. A list_length parameter is used to ensure that the output tensor has a
        fixed shape. If the separator is empty, the string is split on bytes/characters.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if an iterable of tensors is passed
        in.

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

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringToStringList layer.
        Used for saving and loading from a model.

        Specifically adds the `axis`, `separator` and `keepdims` to the config
        dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "separator": self.separator,
                "default_value": self.default_value,
                "list_length": self.list_length,
            }
        )
        return config
