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

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
from itertools import chain

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import StringType

from kamae.keras.tensorflow.layers import DateParseLayer
from kamae.params import ParamSpec
from kamae.params.shared_specs import DEFAULT_INT_VALUE_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform


def _validate_date_part(value: str) -> str:
    """Validator for datePart parameter."""
    allowed_date_parts = {
        "Year",
        "DayOfYear",
        "MonthOfYear",
        "DayOfMonth",
        "DayOfWeek",
        "Hour",
        "Minute",
        "Second",
        "Millisecond",
    }
    if value not in allowed_date_parts:
        raise ValueError(
            f"Invalid date part: {value}. Must be one of {allowed_date_parts}"
        )
    return value


class DateParseTransformer(BaseTransformer, SingleInputSingleOutputParams):
    """
    Date parse transform layer.
    This layer parses a date(time) column into a specified date part.
    We require the date format to be yyyy-MM-dd (HH:mm:ss.SSS).

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
    """

    _compatible_dtypes = [StringType()]
    _keras_layer_class = DateParseLayer
    _params = {
        **DEFAULT_INT_VALUE_PARAMS,
        "datePart": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default=None,
            doc="Date part to extract from date",
            validator=_validate_date_part,
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input date column into the specified date part.

        Utilises date format, which itself uses the following:
        https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html

        :param dataset: Input dataframe.
        :returns: Transformed dataframe.
        """

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def date_parse(x: Column) -> Column:
            if self.getDefaultValue() is not None:
                return F.when(x == F.lit(""), F.lit(self.getDefaultValue())).otherwise(
                    self._parse_date(x)
                )
            return self._parse_date(x)

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: date_parse(x),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def _parse_date(self, column: Column) -> Column:
        """
        Private function to parse the date column.
        :param column: Input column.
        :returns: Parsed datepart column.
        """

        date_part_to_format_pattern = {
            "Year": "y",
            "DayOfYear": "D",
            "MonthOfYear": "M",
            "DayOfMonth": "d",
            "DayOfWeek": "E",
            "Hour": "H",
            "Minute": "m",
            "Second": "s",
            "Millisecond": "SSS",
        }

        formatted_date = F.date_format(
            column, date_part_to_format_pattern[self.getDatePart()]
        )

        if self.getDatePart() == "DayOfWeek":
            day_of_week_mapping = {
                "Mon": 1,
                "Tue": 2,
                "Wed": 3,
                "Thu": 4,
                "Fri": 5,
                "Sat": 6,
                "Sun": 7,
            }

            day_of_week_spark_mapping = F.create_map(
                [F.lit(x) for x in chain(*day_of_week_mapping.items())]
            )

            return day_of_week_spark_mapping[formatted_date]

        return formatted_date.cast("int")
