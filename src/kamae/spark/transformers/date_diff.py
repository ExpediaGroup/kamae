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
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import DateDiffLayer
from kamae.params.shared_specs import DEFAULT_INT_VALUE_PARAMS
from kamae.spark.params import MultiInputSingleOutputParams
from kamae.spark.utils import multi_input_single_output_scalar_transform

from .base import BaseTransformer


class DateDiffTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
):
    """
    DateDiffLayer Spark Transformer for use in Spark pipelines.
    This transformer calculates the difference between two dates.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = DateDiffLayer
    _params = {**DEFAULT_INT_VALUE_PARAMS}

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
