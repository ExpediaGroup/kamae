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
class StringArrayConstantLayer(BaseLayer):
    """
    Tensorflow keras layer that outputs a constant string array.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        constant_string_array: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialises the String Array Constant layer.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param constant_string_array: The constant string array to output.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.constant_string_array = constant_string_array

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return None

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Returns the constant string array with the same shape as the input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

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

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringArrayConstant layer.
        Used for saving and loading from a model.

        Specifically adds the `constant_string_array` to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"constant_string_array": self.constant_string_array})
        return config
