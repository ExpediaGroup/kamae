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
from kamae.params.shared_specs import UNIX_TIMESTAMP_PARAMS


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

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = None
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
        Returns the current unix timestamp in either seconds or milliseconds.
        Uses the input tensor to determine the shape of the output tensor.


        :param inputs: Input tensor to determine the shape of the output tensor.
        :returns: The current timestamp tensor in yyyy-MM-dd format.
        """
        current_timestamp_in_seconds = tf.fill(tf.shape(inputs), tf.timestamp())
        return (
            current_timestamp_in_seconds
            if self.unit == "s"
            else current_timestamp_in_seconds * 1000.0
        )
