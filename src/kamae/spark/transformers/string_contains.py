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

from kamae.spark.params import (
    MultiInputSingleOutputParams,
    NegationParams,
    SingleInputSingleOutputParams,
    StringConstantParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringContainsLayer

from .base import BaseTransformer


class StringContainsTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
    NegationParams,
    StringConstantParams,
):
    """
    String contains Spark Transformer for use in Spark pipelines.
    This transformer performs a string contains operation on the input column.
    If the string constant is specified, we use it for the string contains
    on the single input. Otherwise, if multiple input columns are specified,
    we check if the first input column contains the second.
    Used for cases where you want to keep the input the same.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        stringConstant: Optional[str] = None,
        negation: bool = False,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initializes an StringContainsTransformer transformer.

        :param inputCol: Input column name.
        :param inputCols: List of input column names.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param stringConstant: String constant to use in string contains
        operation.
        Only used in single input scenario.
        :param negation: Whether to negate the string contains operation.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(negation=False, stringConstant=None)
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

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the StringContainsLayer transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs a string contains operation.
        """
        return StringContainsLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            negation=self.getNegation(),
            string_constant=self.getStringConstant(),
        )
