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
from kamae.keras.tensorflow.utils.date_utils import unix_timestamp_to_datetime


class CurrentDateLayer(BaseLayer):
    """
    Returns the current UTC date in yyyy-MM-dd format.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = None

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Returns the current timestamp in yyyy-MM-dd format.
        Uses the input tensor to determine the shape of the output tensor.


        :param inputs: Input tensor to determine the shape of the output tensor.
        :returns: The current timestamp tensor in yyyy-MM-dd format.
        """
        current_timestamp = tf.fill(tf.shape(inputs), tf.timestamp())
        outputs = unix_timestamp_to_datetime(current_timestamp, False)
        return outputs
