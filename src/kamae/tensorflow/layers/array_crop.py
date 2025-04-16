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

from typing import Any, Dict, List, Optional, Union

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(kamae.__name__)
class ArrayCropLayer(BaseLayer):
    """
    Performs a cropping of the input tensor to a certain length.
    If the tensor is shorter than the specified length, it is
    padded with specified pad value.

    TODO: Currently only supports cropping the final dimension of the tensor.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: Union[str, int, float] = None,
        output_dtype: Union[str, int, float] = None,
        array_length: int = 128,
        pad_value: Union[str, int, float] = None,
        **kwargs,
    ) -> None:
        """
        Initialises the ArrayCropLayer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param array_length: The length to crop or pad the arrays to. Defaults to 128.
        :param pad_value: The value to pad the arrays with. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if array_length < 1:
            raise ValueError("Array length must be greater than 0.")
        self.array_length = array_length

        if pad_value is None:
            raise ValueError("Pad value must be provided and not None.")
        self.pad_value = pad_value

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
        Crops the tensor to specified length and pads with specified value.

        :param inputs: Tensor to split.
        :returns: Cropped and padded tensor
        """
        inputs_shape = tf.shape(inputs)

        # Crop final dimension of tensor
        crop_length = tf.minimum(self.array_length, inputs_shape[-1])
        cropped = inputs[..., :crop_length]

        # Pad final dim of tensor if necessary
        padding_length = tf.maximum(self.array_length - inputs_shape[-1], 0)
        paddings = [[0, 0]] * (inputs_shape.shape[0] - 1) + [[0, padding_length]]
        padded = tf.pad(cropped, paddings, constant_values=self.pad_value)
        new_shape = tf.concat(
            [
                tf.shape(padded)[:-1],
                tf.expand_dims(tf.constant(self.array_length), axis=-1),
            ],
            axis=0,
        )
        return tf.reshape(padded, new_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the ArrayCrop layer.
        Used for saving and loading from a model.

        Specifically, adds the `array_length` amd `pad_value to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"array_length": self.array_length, "pad_value": self.pad_value})
        return config
