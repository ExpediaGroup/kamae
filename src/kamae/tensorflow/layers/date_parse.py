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
    datetime_day,
    datetime_day_of_year,
    datetime_hour,
    datetime_millisecond,
    datetime_minute,
    datetime_month,
    datetime_second,
    datetime_weekday,
    datetime_year,
    enforce_single_tensor_input,
)

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class DateParseLayer(BaseLayer):
    """
    Parses a date(time) string from yyyy-MM-dd (HH:mm:ss.SSS) format
    into a specified date part tensor.

    Date parts can be one of the following:
    - `DayOfWeek` - day of week (Monday = 1, Sunday = 7)
    - `DayOfMonth` - day of month
    - `DayOfYear` - day of year e.g. (2021-01-01 = 1, 2021-12-31 = 365)
    - `MonthOfYear` - month of year
    - `Year` - year
    - `Hour` - hour e.g. (2021-01-01 00:00:00 = 0, 2021-01-01 23:59:59 = 23)
    - `Minute` - minute e.g. (2021-01-01 00:00:00 = 0, 2021-01-01 00:59:00 = 59)
    - `Second` - second e.g. (2021-01-01 00:00:00 = 0, 2021-01-01 00:00:59 = 59)
    - `Millisecond` - millisecond (2021-01-01 00:00:00.357 = 357)

    In the case a timestamp is not provided, all hour, minutes, seconds and milliseconds
    fields will be returned as 0.

    All date parts except seconds and milliseconds are returned as int32, but due to the
    precision of seconds and milliseconds, these are returned as int64 to prevent
    overflow.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown and you will get a nonsense output.
    """

    def __init__(
        self,
        date_part: str,
        name=None,
        input_dtype: str = None,
        output_dtype: str = None,
        default_value: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialises an instance of the DateParseLayer layer.

        :param date_part: Date part to extract from date.
        :param name: Name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param default_value: Default value to use when the date is the empty string.
        Empty strings can be used when the date is not available.
        :returns: None - class instantiated.
        """
        self.allowed_date_parts = {
            "DayOfWeek",
            "DayOfMonth",
            "DayOfYear",
            "MonthOfYear",
            "Year",
            "Hour",
            "Minute",
            "Second",
            "Millisecond",
        }
        if date_part not in self.allowed_date_parts:
            raise ValueError(f"date_part must be one of {self.allowed_date_parts}")
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.date_part = date_part
        self.default_value = default_value

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Extracts date part from date(time) string.

        Decorated with `@enforce_single_tensor_input` to ensure that only a single
        tensor is passed in. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Tensor of date(time) strings in the yyyy-MM-dd (HH:mm:ss.SSS)
        format.
        :returns: Date part tensor.

        WARNING: Dates are not checked for validity, so if you pass in a date such
        as "2020-02-30" no errors will be thrown and you will get a nonsense output.
        """
        if self.default_value is not None:
            # Trick to replace empty strings with a valid dummy date, that we ignore
            # later. Otherwise, the parse_date function will raise an error
            replaced_date = tf.where(
                tf.equal(inputs, ""), "2000-01-01 00:00:00.000", inputs
            )
            outputs = tf.where(
                tf.equal(inputs, ""),
                tf.constant(self.default_value, dtype=tf.int64),
                self._parse_date(replaced_date, self.date_part),
            )
        else:
            outputs = self._parse_date(inputs, self.date_part)
        return outputs

    @staticmethod
    def _parse_date(date_tensor: Tensor, date_part: str) -> Tensor:
        """
        Parse date(time) string into a dictionary of date part tensors.

        :param date_tensor: Tensor of date(time) strings in the
        YYYY-mm-dd (HH:MM:ss.SSS) format.
        :returns: Dictionary of date part tensors.
        """

        date_part_functions = {
            "DayOfWeek": datetime_weekday,
            "DayOfMonth": datetime_day,
            "DayOfYear": datetime_day_of_year,
            "MonthOfYear": datetime_month,
            "Year": datetime_year,
            "Hour": datetime_hour,
            "Minute": datetime_minute,
            "Second": datetime_second,
            "Millisecond": datetime_millisecond,
        }

        try:
            return date_part_functions[date_part](date_tensor)
        except KeyError:
            raise ValueError(
                f"""date_part must be one of {list(date_part_functions.keys())}"""
            )

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the DateParse layer.
        Used for saving and loading from a model.

        Specifically adds the `date_part` to the config dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {"date_part": self.date_part, "default_value": self.default_value}
        )
        return config
