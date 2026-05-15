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
from pyspark.sql.types import DataType, DoubleType, FloatType

from kamae.keras.core.layers import ExponentLayer
from kamae.params import _UNSET, ParamSpec
from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform

from .base import BaseTransformer


class ExponentTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
):
    """
    Exponent Spark Transformer for use in Spark pipelines.
    This transformer applies x^exponent in the case of single input and or x^y in the
    case of two inputs.
    """

    jit_compatible = True

    _compatible_dtypes = [
        FloatType(),
        DoubleType(),
    ]
    _keras_layer_class = ExponentLayer
    _params = {
        "exponent": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=_UNSET,
            doc="Value to use in exponent transform: x^exponent",
        ),
    }

    def setInputCols(self, value: List[str]) -> "ExponentTransformer":
        """
        Overrides setting the input columns for the transformer.
        Throws an error if we do not have exactly two input columns.

        :param value: List of input columns.
        :returns: Class instance.
        """
        if len(value) != 2:
            raise ValueError(
                """When setting inputCols for ExponentTransformer,
                there must be exactly two input columns."""
            )
        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which applies the exponent transform to the input column(s).

        If one column is provided via inputCol, we raise that column to the power of
        the exponent parameter. If two columns are provided via inputCols,
        we raise the first column to the power of the second column.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self.get_multiple_input_cols(
            constant_param_name="exponent", input_cols_limit=2
        )
        # input_cols can contain either actual columns or lit(constants). In order to
        # determine the datatype of the input columns, we select them from the dataset
        # first.
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            func=lambda x: F.pow(x[input_col_names[0]], x[input_col_names[1]]),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
