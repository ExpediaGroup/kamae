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
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.keras.tensorflow.utils.date_utils import datetime_to_unix_timestamp
from kamae.params.shared_specs import UNIX_TIMESTAMP_PARAMS


class DateTimeToUnixTimestampLayer(BaseLayer):
    """
    Returns the unix timestamp from a datetime in either yyyy-MM-dd HH:mm:ss.SSS
    or yyyy-MM-dd format.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        **UNIX_TIMESTAMP_PARAMS,
    }

    @staticmethod
    def _post_init(self):
        if self.unit not in ["milliseconds", "seconds", "ms", "s"]:
            raise ValueError(
                """Unit must be one of ["milliseconds", "seconds", "ms", "s"]"""
            )
        if self.unit == "milliseconds":
            self.unit = "ms"
        elif self.unit == "seconds":
            self.unit = "s"

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Returns the unix timestamp from a datetime in either yyyy-MM-dd HH:mm:ss.SSS
        or yyyy-MM-dd format.


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
