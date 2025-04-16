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
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringMapLayer

from .base import BaseTransformer


class StringMapTransformerParams(Params):
    """
    Mixin class containing StringMatchValues and StringReplaceValues
    needed for string map layers.
    """

    stringMatchValues = Param(
        Params._dummy(),
        "stringMatchValues",
        "String match constant to use in string replace.",
        typeConverter=TypeConverters.toListString,
    )

    stringReplaceValues = Param(
        Params._dummy(),
        "stringReplaceValues",
        "String replace constant to use in string replace.",
        typeConverter=TypeConverters.toListString,
    )

    defaultReplaceValue = Param(
        Params._dummy(),
        "defaultReplaceValue",
        """
        Default value to replace the unmatched strings with.
        If None, the original string is kept unchanged.
        """,
        typeConverter=TypeConverters.toString,
    )

    def setStringMatchValues(self, value: List[str]) -> "StringMapTransformerParams":
        """
        Sets the stringMatchValues parameter.

        :param value: List of string match constants.
        :returns: Instance of class mixed in.
        """
        if value is None or len(value) == 0:
            raise ValueError("stringMatchValues cannot be empty.")
        return self._set(stringMatchValues=value)

    def getStringMatchValues(self) -> List[str]:
        """
        Gets the stringMatchValues parameter.

        :returns: List of string match constants.
        """
        return self.getOrDefault(self.stringMatchValues)

    def setStringReplaceValues(self, value: List[str]) -> "StringMapTransformerParams":
        """
        Sets the stringReplaceValues parameter.

        :param value: List of string replace constants.
        :returns: Instance of class mixed in.
        """
        if value is None or len(value) == 0:
            raise ValueError("stringReplaceValues cannot be empty.")
        return self._set(stringReplaceValues=value)

    def getStringReplaceValues(self) -> List[str]:
        """
        Gets the stringReplaceValues parameter.

        :returns: List of string replace constants.
        """
        return self.getOrDefault(self.stringReplaceValues)

    def setDefaultReplaceValue(self, value: str) -> "StringMapTransformerParams":
        """
        Sets the defaultReplaceValue parameter.

        :param value: Default value to replace the unmatched strings with.
        If None, the original string is kept unchanged.
        :returns: Instance of class mixed in.
        """
        return self._set(defaultReplaceValue=value)

    def getDefaultReplaceValue(self) -> str:
        """
        Gets the defaultReplaceValue parameter.

        :returns: Default value to replace the unmatched strings with.
        """
        return self.getOrDefault(self.defaultReplaceValue)


class StringMapTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    StringMapTransformerParams,
):
    """
    String Map Spark Transformer for use in Spark Pipelines.
    This transformer replaces a list of strings with the respective mapping value.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        stringMatchValues: Optional[List[str]] = None,
        stringReplaceValues: Optional[List[str]] = None,
        defaultReplaceValue: Optional[str] = None,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initializes an StringMapTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param stringMatchValues: List of string match constants.
        :param stringReplaceValues: List of string replace constants.
        :param defaultReplaceValue: Default value to replace the unmatched strings with.
        If None, the original string is kept unchanged.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(
            stringMatchValues=None,
            stringReplaceValues=None,
            defaultReplaceValue=None,
        )
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

    def _string_map(self, column: Column) -> Column:
        """
        Helper function to create a string map expression.

        :param column: Column to apply the string map operation to.
        :returns: Column with string map operation applied.
        """
        col_expr: Column = None
        for match_value, replace_value in zip(
            self.getStringMatchValues(), self.getStringReplaceValues()
        ):
            if col_expr is None:
                col_expr = F.when(column == F.lit(match_value), replace_value)
            else:
                col_expr = col_expr.when(column == F.lit(match_value), replace_value)
        if self.getDefaultReplaceValue() is not None:
            col_expr = col_expr.otherwise(self.getDefaultReplaceValue())
        else:
            col_expr = col_expr.otherwise(column)
        return col_expr

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which contains the result of the string map operation.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        if self.getStringMatchValues() is None or self.getStringReplaceValues() is None:
            raise ValueError(
                "stringMatchValues and stringReplaceValues cannot be None."
            )
        if len(self.getStringMatchValues()) != len(self.getStringReplaceValues()):
            raise ValueError(
                "Length of stringMatchValues and stringReplaceValues must be equal."
            )
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: self._string_map(x),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the StringMapLayer transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
        performs a string replace operation.
        """
        return StringMapLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            string_match_values=self.getStringMatchValues(),
            string_replace_values=self.getStringReplaceValues(),
            default_replace_value=self.getDefaultReplaceValue(),
        )
