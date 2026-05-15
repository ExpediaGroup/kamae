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
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ByteType,
    DataType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
)

from kamae.keras.tensorflow.layers import DateAddLayer
from kamae.params import ParamSpec
from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import (
    get_element_type,
    multi_input_single_output_scalar_transform,
)


def _validate_num_days(self, value: int) -> int:
    if self.isDefined("inputCols"):
        raise ValueError("Cannot set numDays if using multiple inputCols.")
    return value


class DateAddTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
):
    """
    Transformer to add or subtract a static or dynamic (column) number of days
    from a date column.

    WARNING: This transform destroys the time component of the date column.
    """

    _compatible_dtypes = [
        StringType(),
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType(),
    ]
    _keras_layer_class = DateAddLayer
    _params = {
        "numDays": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=None,
            doc="Number of days to add/subtract. Negative values subtract.",
            validator=_validate_num_days,
        ),
    }

    def setInputCols(self, value: List[str]) -> "DateAddTransformer":
        """
        Sets the value of the inputCols parameter.

        :param value: Input column names.
        :returns: Class instance.
        """
        if len(value) != 2:
            raise ValueError("If using multiple inputs, exactly two are required.")
        if self.getNumDays() is not None:
            raise ValueError("Cannot use multiple inputs if numDays is set.")
        if self.getInputDtype() is not None:
            raise ValueError(
                """Input auto-casting is set via inputDtype, however multiple inputs are
                being used. Auto-casting inputs is not supported for multiple inputs in
                the DateAddTransformer because the two inputs must be
                different types."""
            )
        return self._set(inputCols=value)

    def setInputDtype(self, value: str) -> "DateAddTransformer":
        """
        Overrides setting the parameter inputDtype to the given string value.

        If multiple input columns are being used, the inputDtype parameter cannot be
        set.

        :param value: String to set the inputDtype parameter to.
        :raises ValueError: If inputCols is set.
        :returns: Instance of class mixed in.
        """
        if self.isDefined("inputCols"):
            raise ValueError(
                """Input auto-casting is not supported for multiple inputs in the
                DateAddTransformer because the two inputs must be different types."""
            )
        return self._set(inputDtype=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Adds or subtracts a number of days from a date column.

        :param dataset: Input dataframe.
        :returns: Transformed dataframe.
        """
        input_cols = self.get_multiple_input_cols(
            constant_param_name="numDays", input_cols_limit=2
        )
        # input_cols can contain either actual columns or lit(constants). In order to
        # determine the datatype of the input columns, we select them from the dataset
        # first.
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]
        if not isinstance(get_element_type(input_col_datatypes[0]), StringType):
            raise ValueError(
                f"""Expected input column {input_col_names[0]} to have element type
                StringType, but got {input_col_datatypes[0]}."""
            )
        if not isinstance(
            get_element_type(input_col_datatypes[1]),
            (ByteType, ShortType, IntegerType, LongType),
        ):
            raise ValueError(
                f"""Expected input column {input_col_names[1]} to have element type
                 ByteType, ShortType or IntegerType, but got {input_col_datatypes[1]}.
                 """
            )
        date_col_name = input_col_names[0]
        num_days_col_name = input_col_names[1]
        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            # Cast to int since LongType not supported in date_add, but we want to allow
            # it as an input type.
            func=lambda x: F.date_add(
                x[date_col_name], x[num_days_col_name].cast("int")
            ).cast("string"),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
