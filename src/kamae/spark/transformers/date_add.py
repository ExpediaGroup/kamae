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
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ByteType,
    DataType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
)

from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import (
    get_element_type,
    multi_input_single_output_scalar_transform,
)
from kamae.tensorflow.layers import DateAddLayer


class DateAdditionParams(Params):
    """
    Mixin class for a date addition transformer.
    """

    numDays = Param(
        Params._dummy(),
        "numDays",
        "Number of days to add/subtract. Negative values subtract.",
        typeConverter=TypeConverters.toInt,
    )

    def getNumDays(self) -> int:
        """
        Gets the value of the numDays parameter.

        :returns: Number of days to add/subtract.
        """
        return self.getOrDefault(self.numDays)

    def setNumDays(self, value: int) -> "DateAdditionParams":
        """
        Sets the value of the numDays parameter.

        :param value: Number of days to add/subtract.
        :returns: Class instance.
        """
        if self.isDefined("inputCols"):
            raise ValueError("Cannot set numDays if using multiple inputCols.")
        return self._set(numDays=value)


class DateAddTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
    DateAdditionParams,
):
    """
    Transformer to add or subtract a static or dynamic (column) number of days
    from a date column.

    WARNING: This transform destroys the time component of the date column.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        numDays: Optional[int] = None,
    ) -> None:
        """
        Initialises the date add transform layer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Layer name. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param numDays: Number of days to add/subtract. Negative values subtract.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(numDays=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

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

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [StringType(), ByteType(), ShortType(), IntegerType(), LongType()]

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

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer.

        :returns: DateAddLayer Tensorflow layer.
        """
        return DateAddLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            num_days=self.getNumDays(),
        )
