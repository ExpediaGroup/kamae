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
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import DefaultIntValueParams, MultiInputSingleOutputParams
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.tensorflow.layers import DateDiffLayer

from .base import BaseTransformer


class DateDiffTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
    DefaultIntValueParams,
):
    """
    DateDiffLayer Spark Transformer for use in Spark pipelines.
    This transformer calculates the difference between two dates.
    """

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        defaultValue: Optional[int] = None,
    ) -> None:
        """
        Initializes an DateDiffTransformer transformer.

        :param inputCols: Input column names.
        The inputs must be in yyyy-MM-dd (HH:mm:ss.SSS) format and
        must be passed to the layer in the order [start date , end date].
        The transformer will return a negative value if the order is reversed.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param defaultValue: Default value to use when one of the dates is the empty
        string. Empty strings can be used when the date is not available.
        :returns: None - class instantiated.
        """

        super().__init__()
        self._setDefault(defaultValue=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [StringType()]

    def setInputCols(self, value: List[str]) -> "DateDiffTransformer":
        """
        Overrides setting the input columns for the transformer.
        Throws an error if we do not have exactly two input columns.

        :param value: List of input columns.
        :returns: Class instance.
        """
        if len(value) != 2:
            raise ValueError(
                """When setting inputCols for DateDiffTransformer,
                there must be exactly two input columns."""
            )
        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which contains the result of the date difference operation of the inputCols

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_col_names = self.getInputCols()
        input_cols = [F.col(c) for c in input_col_names]
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset, column_name=c)
            for c in input_col_names
        ]

        def date_diff(x: Column) -> Column:
            if self.getDefaultValue() is not None:
                return F.when(
                    (x[input_col_names[0]] == F.lit(""))
                    | (x[input_col_names[1]] == F.lit("")),
                    F.lit(self.getDefaultValue()),
                ).otherwise(F.datediff(x[input_col_names[1]], x[input_col_names[0]]))
            return F.datediff(x[input_col_names[1]], x[input_col_names[0]])

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_datatypes=input_col_datatypes,
            input_col_names=input_col_names,
            func=lambda x: date_diff(x),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the absolute value transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs an absolute value operation.
        """
        return DateDiffLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            default_value=self.getDefaultValue(),
        )
