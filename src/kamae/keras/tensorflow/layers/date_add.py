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
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input
from kamae.keras.tensorflow.utils.date_utils import datetime_add_days
from kamae.params import ParamSpec


class DateAddLayer(BaseLayer):
    """
    Adds or subtracts a number of days from a date(time) string.

    WARNING: This layer destroys the time component of the date column.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string", "int8", "int16", "int32", "int64"]
    _params = {
        "num_days": ParamSpec(
            default=None,
            doc="Number of days to add or subtract",
        ),
    }

    @staticmethod
    def _post_init(self):
        if self.num_days is not None and not isinstance(self.num_days, int):
            raise ValueError(
                f"Expected `num_days` to be an integer, but got {self.num_days}."
            )
        if self.num_days is None and self._input_dtype is not None:
            raise ValueError(
                """When `num_days` is not set, the layer expects two inputs of different
                dtypes. Therefore input auto-casting via `input_dtype` is not supported.
                """
            )

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Adds or subtracts a number of days from a date(time) string.
        """
        if inputs[0].dtype != tf.string:
            raise ValueError(
                f"Expected input dtype to be tf.string, but got {inputs[0].dtype}."
            )
        if self.num_days is not None:
            if len(inputs) > 1:
                raise ValueError(
                    "When `num_days` is set, the input should be a single tensor."
                )
            return datetime_add_days(
                inputs[0],
                tf.constant(self.num_days, dtype=tf.float64),
                include_time=False,
            )
        else:
            if len(inputs) != 2:
                raise ValueError(
                    "When `num_days` is not set, the input should be two tensors."
                )
            if not inputs[1].dtype.is_integer:
                raise ValueError(
                    f"""Expected second input dtype to be integer, but got
                    {inputs[1].dtype}."""
                )
            return datetime_add_days(
                inputs[0],
                # Casting is necessary since all datetime ops are in float64
                # Furthermore, due to the input dtypes being different (e.g. first input
                # must be tf.string, second input must be integer), we cast to
                # potentially undo the auto-casting done by specifying input_dtype.
                self._cast(inputs[1], cast_dtype="float64"),
                include_time=False,
            )
