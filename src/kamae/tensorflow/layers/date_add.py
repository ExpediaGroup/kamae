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
    allow_single_or_multiple_tensor_input,
    datetime_add_days,
)

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class DateAddLayer(BaseLayer):
    """
    Adds or subtracts a number of days from a date(time) string.

    WARNING: This layer destroys the time component of the date column.
    """

    def __init__(
        self,
        name=None,
        input_dtype: str = None,
        output_dtype: str = None,
        num_days: int = None,
        **kwargs,
    ) -> None:
        """
        Initialises an instance of the DateAddLayer.

        :param num_days: Number of days to add or subtract.
        :param name: Name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if num_days is not None and not isinstance(num_days, int):
            raise ValueError(
                f"Expected `num_days` to be an integer, but got {num_days}."
            )
        if num_days is None and input_dtype is not None:
            raise ValueError(
                """When `num_days` is not set, the layer expects two inputs of different
                dtypes. Therefore input auto-casting via `input_dtype` is not supported.
                """
            )
        self.num_days = num_days

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string, tf.int8, tf.int16, tf.int32, tf.int64]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
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

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the DateAdd layer.
        Used for saving and loading from a model.

        Specifically adds the `num_days` to the config dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"num_days": self.num_days})
        return config
