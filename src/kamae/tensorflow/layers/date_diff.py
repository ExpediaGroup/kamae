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
from kamae.tensorflow.utils import datetime_total_days, enforce_multiple_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class DateDiffLayer(BaseLayer):
    """A preprocessing layer that returns the difference between two dates in days.

    The inputs must be in yyyy-MM-dd (HH:mm:ss.SSS) format and
    must be passed to the layer in the order [start date , end date].
    The transformer will return a negative value if the order is reversed.
    """

    def __init__(
        self,
        name=None,
        input_dtype: str = None,
        output_dtype: str = None,
        default_value: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the DateDiffLayer layer.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.default_value = default_value

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_multiple_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs the date difference operation on two input tensors.

        Decorated with `@enforce_multiple_tensor_input` to ensure that the input
        is an iterable. Raises an error if a single tensor is passed.

        We also then check if the length of the iterable is 2.
        If not, we raise an error.

        :param inputs: Iterable of two tensors to perform the date difference operation
        on.
        :returns: Single tensor with the difference between the two dates in days.
        """
        if len(inputs) != 2:
            raise ValueError("Input shape must be an iterable of two tensors")

        start_date, end_date = inputs
        if self.default_value is not None:
            # Trick to replace empty strings with a valid dummy date, that we ignore
            # later. Otherwise, the date_difference function will raise an error
            replaced_start_date = tf.where(
                tf.equal(start_date, ""), "2000-01-01 00:00:00.000", start_date
            )
            replaced_end_date = tf.where(
                tf.equal(end_date, ""), "2000-01-01 00:00:00.000", end_date
            )
            outputs = tf.where(
                tf.logical_or(tf.equal(start_date, ""), tf.equal(end_date, "")),
                tf.constant(self.default_value, dtype=tf.int64),
                self.date_difference(replaced_end_date, replaced_start_date),
            )
        else:
            outputs = self.date_difference(end_date, start_date)
        return outputs

    def date_difference(self, end_date: Tensor, start_date: Tensor) -> Tensor:
        """
        Calculates the difference between two dates.

        :param end_date: Tensor of end dates.
        :param start_date: Tensor of start dates.
        :returns: Tensor of date difference in days.
        """
        return datetime_total_days(end_date) - datetime_total_days(start_date)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the DateDiff layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"default_value": self.default_value})
        return config
