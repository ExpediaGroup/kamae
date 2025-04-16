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

from typing import List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class ArraySubtractMinimumLayer(BaseLayer):
    """
    TensorFlow layer that computes the difference across an axis from the minimum
    non-paded element in the input tensor.

    It takes a tensor of numerical value and calculates the differences between
    each value and the minimum value in the tensor. The calculation preserves
    the pad value elements.

    The principal use case for this layer is to calculate the time difference
    from the first event to all events in a sequence, where the tensor is a array of
    timestamps.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis: int = -1,
        pad_value=None,
        **kwargs,
    ) -> None:
        """
        Initialises the ArraySubtractMinimum layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: The axis along which the differences are calculated.
        Defaults to -1.
        :param pad_value: The value to be considered as padding. Defaults to `None`.
        :returns: None
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.axis = axis
        self.pad_value = pad_value

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [
            tf.bfloat16,
            tf.float16,
            tf.float32,
            tf.float64,
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.int32,
            tf.int64,
            tf.uint32,
            tf.uint64,
        ]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs the calculation of the differences on the input tensor.

        Example:
         input_tensor = tf.Tensor([
            [19, 18, 13, 11, 10, -1, -1, -1],
            [12, 2, 1, -1, -1, -1, -1, -1],
            ]
         )
         layer = ArraySubtractMinimumLayer()
         differences = layer(input_tensor)
         print(differences)
         Output: tf.Tensor([[
            [9, 8, 3, 1, 0, -1, -1, -1],
            [11, 1, 0, -1, -1, -1, -1, -1],
            ]
        )

        :param inputs: The input tensor.
        :returns: Tensor of differences from the minimum (non-padded) timestamp.
        """
        if self.pad_value is None:
            # If pad value is not defined, then the smallest value in the tensor is
            # considered as the first value and subtracted from all the values.
            first_value = tf.reduce_min(inputs, axis=self.axis)
            subtracted_val = tf.subtract(inputs, tf.expand_dims(first_value, self.axis))
            return subtracted_val

        # Otherwise, we find the smallest non padded value and subtract it from all
        # the values. Padded values are preserved.
        inputs, pad_tensor = self._force_cast_to_compatible_numeric_type(
            inputs, self.pad_value
        )
        first_non_pad_value = tf.reduce_min(
            tf.where(tf.equal(inputs, pad_tensor), inputs.dtype.max, inputs),
            axis=self.axis,
        )
        subtracted_val = tf.subtract(
            inputs, tf.expand_dims(first_non_pad_value, self.axis)
        )
        return tf.where(tf.equal(inputs, pad_tensor), inputs, subtracted_val)

    def get_config(self):
        """
        Returns the configuration of the layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer
        """
        config = super().get_config()
        config.update(
            {
                "pad_value": self.pad_value,
                "axis": self.axis,
            }
        )
        return config
