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
class StringIsInListLayer(BaseLayer):
    """
    Performs a string isin operation on the input tensor over entries in
    the string constant list.
    """

    def __init__(
        self,
        string_constant_list: List[str],
        name: Optional[str] = None,
        input_dtype: str = None,
        output_dtype: str = None,
        negation: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialises the StringIsInListLayer layer.
        :param string_constant_list: The string to match against.
        :param negation: Whether to negate the output. Defaults to `False`.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.negation = negation
        self.string_constant_list = string_constant_list

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
        Checks if the input tensor is matching any string in the string_constant_list.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param: inputs: Input string tensor.
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

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringIsInListLayer layer.
        Used for saving and loading from a model.

        Specifically adds the string_constant_list and negation parameters to the
        config dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "string_constant_list": self.string_constant_list,
                "negation": self.negation,
            }
        )
        return config
