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
from typing import List

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringContainsLayer
from kamae.params import ParamSpec
from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform

from .base import BaseTransformer


class StringContainsTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
):
    """
    String contains Spark Transformer for use in Spark pipelines.
    This transformer performs a string contains operation on the input column.
    If the string constant is specified, we use it for the string contains
    on the single input. Otherwise, if multiple input columns are specified,
    we check if the first input column contains the second.
    Used for cases where you want to keep the input the same.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = StringContainsLayer
    _params = {
        "stringConstant": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default=None,
            doc="String constant to use in string contains operation. "
            "Only used in single input scenario.",
        ),
        "negation": ParamSpec(
            spark_typeconverter=TypeConverters.toBoolean,
            default=False,
            doc="Whether to negate the string contains operation.",
        ),
    }

    def setInputCols(self, value: List[str]) -> "StringContainsTransformer":
        """
        Overrides setting the input columns for the transformer.
        Throws an error if we do not have exactly two input columns.

        :param value: List of input columns.
        :returns: Class instance.
        """
        if len(value) != 2:
            raise ValueError(
                """When setting inputCols for StringContainsTransformer,
                there must be exactly two input columns."""
            )
        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which contains the result of the string contains operation.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self.get_multiple_input_cols(
            constant_param_name="stringConstant", input_cols_limit=2
        )

        # input_cols can contain either actual columns or lit(constants). In order to
        # determine the datatype of the input columns, we select them from the dataset
        # first.
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]

        def string_contains(
            x: Column, input_col_names: List[str], negation: bool
        ) -> Column:
            col_expr = F.when(
                x[input_col_names[1]] == F.lit(""), x[input_col_names[0]] == F.lit("")
            ).otherwise(x[input_col_names[0]].contains(x[input_col_names[1]]))
            return col_expr if not negation else ~col_expr

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            func=lambda x: string_contains(x, input_col_names, self.getNegation()),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
