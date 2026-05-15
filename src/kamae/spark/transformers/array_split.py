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
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType

from kamae.keras.core.layers import ArraySplitLayer
from kamae.spark.params import SingleInputMultiOutputParams
from kamae.spark.utils import single_input_single_output_array_transform

from .base import BaseTransformer


class ArraySplitTransformer(
    BaseTransformer,
    SingleInputMultiOutputParams,
):
    """
    ArraySplit Spark Transformer for use in Spark pipelines.
    This transformer splits an array column into multiple columns.
    """

    jit_compatible = True

    _compatible_dtypes = None
    _keras_layer_class = ArraySplitLayer

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column for each output column equal
        to the value of the input column at the given index.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_cols = []
        for i, column in enumerate(self.getOutputCols()):
            output_col = single_input_single_output_array_transform(
                input_col=F.col(self.getInputCol()),
                input_col_datatype=input_datatype,
                func=lambda x: F.element_at(x, i + 1),
            )
            output_cols.append(output_col.alias(self.getOutputCols()[i]))
        original_columns = [F.col(c) for c in dataset.columns]
        select_cols = original_columns + output_cols
        return dataset.select(select_cols)
