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

from kamae.keras.tensorflow.layers import StringMapLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


def _validate_string_match_values(value: List[str]) -> List[str]:
    """Validates stringMatchValues parameter."""
    if value is None or len(value) == 0:
        raise ValueError("stringMatchValues cannot be empty.")
    return value


def _validate_string_replace_values(value: List[str]) -> List[str]:
    """Validates stringReplaceValues parameter."""
    if value is None or len(value) == 0:
        raise ValueError("stringReplaceValues cannot be empty.")
    return value


class StringMapTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    String Map Spark Transformer for use in Spark Pipelines.
    This transformer replaces a list of strings with the respective mapping value.
    """

    _compatible_dtypes = [StringType()]
    _keras_layer_class = StringMapLayer
    _params = {
        "stringMatchValues": ParamSpec(
            spark_typeconverter=TypeConverters.toListString,
            default=None,
            doc="String match constant to use in string replace.",
            validator=_validate_string_match_values,
        ),
        "stringReplaceValues": ParamSpec(
            spark_typeconverter=TypeConverters.toListString,
            default=None,
            doc="String replace constant to use in string replace.",
            validator=_validate_string_replace_values,
        ),
        "defaultReplaceValue": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default=None,
            doc="Default value to replace the unmatched strings with. "
            "If None, the original string is kept unchanged.",
        ),
    }

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
