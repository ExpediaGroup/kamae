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

from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import (
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.spark.params import ListwiseParams, SingleInputSingleOutputParams
from kamae.spark.utils import check_listwise_columns
from kamae.tensorflow.layers import ListRankLayer

from .base import BaseTransformer


class ListRankTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    ListwiseParams,
):
    """
    Calculate the listwise rank across the query id column.

    Example: calculate the rank of items within a query, given the score.

    :param inputCol: Value column, on which to calculate the rank.
    :param outputCol: Name of output col.
    :param inputDtype: Data Type of input.
    :param outputDtype: Data Type of output.
    :param layerName: The name of the transformer, which typically
    should be the name of the produced feature.
    :param queryIdCol: Name of column to aggregate upon. It is required.
    :param sortOrder: Option of 'asc' or 'desc' which defines order
    for listwise operation. Default is 'desc'.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        queryIdCol: Optional[str] = None,
        sortOrder: str = "desc",
    ) -> None:
        super().__init__()
        self._setDefault(sortOrder="desc")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [
            FloatType(),
            DoubleType(),
            ByteType(),
            ShortType(),
            IntegerType(),
            LongType(),
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Calculate the rank.
        :param dataset: The dataframe with signals and features.
        :returns: The dataframe dataset with the new feature.
        """
        if not self.isDefined("queryIdCol"):
            raise ValueError("queryIdCol must be set on listwise transformers.")

        # Get params
        input_col = self.getInputCol()
        query_id_col = self.getQueryIdCol()
        output_col = self.getOutputCol()
        sort_order = self.getSortOrder()

        # Validate listwise cols
        check_listwise_columns(
            dataset=dataset,
            query_col_name=query_id_col,
            value_col_name=input_col,
            sort_col_name=None,
        )

        # Set sort order
        if sort_order == "asc":
            sort_col = F.col(input_col).asc()
        elif sort_order == "desc":
            sort_col = F.col(input_col).desc()
        else:
            raise ValueError(f"Invalid sortOrder: {sort_order}")
        # Define window spec
        window_spec = Window.partitionBy(query_id_col).orderBy(sort_col)

        # Calculate the rank
        dataset = dataset.withColumn(output_col, F.row_number().over(window_spec))

        return dataset

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the listwise-rank transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs a ranking operation.
        """
        return ListRankLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            sort_order=self.getSortOrder(),
        )
