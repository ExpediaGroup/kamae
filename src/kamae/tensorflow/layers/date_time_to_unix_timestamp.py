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
from kamae.tensorflow.utils import (
    datetime_to_unix_timestamp,
    enforce_single_tensor_input,
)

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class DateTimeToUnixTimestampLayer(BaseLayer):
    """
    Returns the unix timestamp from a datetime in either yyyy-MM-dd HH:mm:ss.SSS
    or yyyy-MM-dd format.
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
        Initialises an instance of the DateTimeToUnixTimstamp layer.

        :param name: Name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param unit: Unit of the timestamp. Can be `milliseconds` (or `ms`)
        or `seconds` (or `s`). Defaults to `s`.
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
        if unit == "seconds":
            unit = "s"
        self.unit = unit

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> tf.Tensor:
        """
        Returns the unix timestamp from a datetime in either yyyy-MM-dd HH:mm:ss.SSS
        or yyyy-MM-dd format.

        Decorated with `@enforce_single_tensor_input` to ensure that
        the input is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input tensor to determine the shape of the output tensor.
        :returns: Unix timestamp in either milliseconds or seconds.
        """
        # Timestamp needs to be in float64 for unix_timestamp_to_datetime
        unix_timestamp_in_seconds = datetime_to_unix_timestamp(inputs)
        return (
            unix_timestamp_in_seconds
            if self.unit == "s"
            else unix_timestamp_in_seconds * 1000.0
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the DateTimeToUnixTimstamp layer.
        Used for saving and loading from a model.

        Specifically sets the `unit` parameters in the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "unit": self.unit,
            }
        )
        return config
