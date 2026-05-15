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
from functools import reduce
from operator import or_
from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import BooleanType, DataType

from kamae.keras.core.layers import LogicalOrLayer
from kamae.spark.params import MultiInputSingleOutputParams
from kamae.spark.utils import multi_input_single_output_scalar_transform

from .base import BaseTransformer


class LogicalOrTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
):
    """
    Logical Or Spark Transformer for use in Spark pipelines.
    This transformer performs an element-wise logical or operation on multiple columns.
    """

    jit_compatible = True

    _compatible_dtypes = [BooleanType()]
    _keras_layer_class = LogicalOrLayer

    def setInputCols(self, value: List[str]) -> "LogicalOrTransformer":
        """
        Sets the inputCols parameter. Raises an error if the value is a list of
        length 1.

        :param value: List of input column names.
        :returns: Instance of class with inputCols parameter set.
        """
        if len(value) == 1:
            raise ValueError("inputCols must be a list of length > 1")
        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the logical or of the input columns.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_col_names = self.getInputCols()
        input_cols = [F.col(c) for c in input_col_names]
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset, column_name=c)
            for c in input_col_names
        ]
        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_datatypes=input_col_datatypes,
            input_col_names=input_col_names,
            func=lambda x: reduce(or_, [x[c] for c in input_col_names]),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
