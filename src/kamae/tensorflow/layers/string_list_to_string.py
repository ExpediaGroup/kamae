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
class StringListToStringLayer(BaseLayer):
    """
    A layer that converts a list of strings to a single string along the specified
    axis.
    If `keepdims` is `True`, the shape is retained.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis: int = -1,
        separator: str = "",
        keepdims: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialises the StringListToStringLayer layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: The axis along which to join the strings. Defaults to `-1`.
        :param separator: The separator to use when joining the strings.
        Defaults to `""`.
        :param keepdims: Whether to keep the shape of the input tensor. Defaults to
        `False`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.axis = axis
        self.separator = separator
        self.keepdims = keepdims

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
        Joins the strings along the specified axis with the specified separator.
        If `keepdims` is `True`, the shape is retained. Otherwise the shape is
        reduced along the specified axis.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if an iterable of tensors is passed
        in.

        :param inputs: Input tensor.
        :returns: Tensor with strings joined along the specified axis.
        """
        return tf.strings.reduce_join(
            inputs, axis=self.axis, separator=self.separator, keepdims=self.keepdims
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringListToString layer.
        Used for saving and loading from a model.

        Specifically adds the `axis`, `separator` and `keepdims` to the config
        dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "separator": self.separator,
                "keepdims": self.keepdims,
            }
        )
        return config
