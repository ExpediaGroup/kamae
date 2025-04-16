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
class RoundToDecimalLayer(BaseLayer):
    """
    Performs a rounding to the nearest decimal operation on the input tensor.

    If the specified number of decimals is too large for the input precision type,
    this operation can result in overflow. This is because the operation is performed by
    multiplying the input tensor by 10 to the power of the number of decimals, rounding
    the result to the nearest integer, and then dividing by 10 to the power of the
    number of decimals.
    """

    def __init__(
        self,
        decimals: int = 1,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the RoundToDecimalLayer layer.

        :param decimals: The number of decimal places to round to.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if decimals < 0:
            raise ValueError("""decimals must be greater than or equal to 0.""")
        self.decimals = decimals

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.float16, tf.float32, tf.float64, tf.int32, tf.int64]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs the rounding operation on the input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that the input is a
        single tensor. Raises an error if multiple tensors are passed in as an iterable.

        :param inputs: Input tensor to perform the rounding on.
        :returns: The input tensor with the rounding applied.
        """
        # WARNING: Depending on the type of the input and the number of decimals,
        # this multiplier could overflow.
        max_val = inputs.dtype.max
        if 10**self.decimals > max_val:
            raise ValueError(
                """The number of decimals is too large for the input dtype.
                Overflow expected."""
            )
        multiplier = tf.constant(10**self.decimals, dtype=inputs.dtype)
        return tf.round(inputs * multiplier) / multiplier

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the RoundToDecimal layer.
        Used for saving and loading from a model.

        Specifically adds the `decimals` value to the configuration.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"decimals": self.decimals})
        return config
