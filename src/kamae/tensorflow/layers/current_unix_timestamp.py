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
class CurrentUnixTimestampLayer(BaseLayer):
    """
    Returns the current unix timestamp in either seconds or milliseconds.

    NOTE: Parity between this and its Spark counterpart is very difficult at the
    millisecond level. TensorFlow provides much more precision of the timestamp,
    and has floating 64-bit precision of the unix timestamp in seconds.
    Whereas Spark 3.4.0 only supports millisecond precision (3 decimal places of unix
    timestamp in seconds). Therefore, parity is not guaranteed at this precision.

    It is recommended not to rely on parity at the millisecond level.
    """

    def __init__(
        self,
        name=None,
        input_dtype: str = None,
        output_dtype: str = None,
        unit: str = "s",
        **kwargs,
    ) -> None:
        """
        Initialises an instance of the CurrentUnixTimestampLayer layer.

        :param name: Name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if unit not in ["milliseconds", "seconds", "ms", "s"]:
            raise ValueError(
                """Unit must be one of ["milliseconds", "seconds", "ms", "s"]"""
            )
        if unit == "milliseconds":
            unit = "ms"
        elif unit == "seconds":
            unit = "s"
        self.unit = unit

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer. Returns `None` as the layer
        only returns the current date as a string. It does not transform any input.

        :returns: The compatible dtypes of the layer.
        """
        return None

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> tf.Tensor:
        """
        Returns the current unix timestamp in either seconds or milliseconds.
        Uses the input tensor to determine the shape of the output tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that
        the input is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input tensor to determine the shape of the output tensor.
        :returns: The current timestamp tensor in yyyy-MM-dd format.
        """
        current_timestamp_in_seconds = tf.fill(tf.shape(inputs), tf.timestamp())
        return (
            current_timestamp_in_seconds
            if self.unit == "s"
            else current_timestamp_in_seconds * 1000.0
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the CurrentUnixTimestamp layer.
        Used for saving and loading from a model.

        Specifically adds the `unit` parameter to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()

        config.update(
            {
                "unit": self.unit,
            }
        )
        return config
