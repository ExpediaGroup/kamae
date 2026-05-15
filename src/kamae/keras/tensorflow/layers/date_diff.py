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

from typing import Any

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_multiple_tensor_input
from kamae.keras.tensorflow.utils.date_utils import datetime_total_days
from kamae.params import ParamSpec


class DateDiffLayer(BaseLayer):
    """A preprocessing layer that returns the difference between two dates in days.

    The inputs must be in yyyy-MM-dd (HH:mm:ss.SSS) format and
    must be passed to the layer in the order [start date , end date].
    The transformer will return a negative value if the order is reversed.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "default_value": ParamSpec(
            default=None,
            doc="Default value to use when the date is the empty string",
        ),
    }

    @enforce_multiple_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
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
