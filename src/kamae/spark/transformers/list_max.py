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
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.spark.params import (
    ListwiseStatisticsParams,
    MultiInputSingleOutputParams,
    NanFillValueParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import check_listwise_columns, get_listwise_condition_and_window
from kamae.tensorflow.layers import ListMaxLayer

from .base import BaseTransformer


class ListMaxTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
    ListwiseStatisticsParams,
    NanFillValueParams,
):
    """
    Calculate the listwise maximum across the query id column.
    - If inputCol is set, the transformer calculates the maximum of the input column
    based on all the items in the same query id column.
    - If inputCols is set, the transformer calculates the maximum of the first column
    based on second column's topN items in the same query id column.

    By using the topN items to calculate the statistics, we can better approximate
    the real statistics in production. It should be used a large enough topN to get a
    good approximation of the statistics, and an important feature to sort on, such as
    item's production.

    Example: calculate the maximum price in the same query, based on the top N
    items sorted by descending production.

    :param inputCol: Value column, on which to calculate the maximum.
    :param inputCols: Input column name.
    - The first is the value column, on which to calculate the maximum.
    - The second is the sort column, based on which to sort the items.
    :param outputCol: Name of output col.
    :param inputDtype: Data Type of input.
    :param outputDtype: Data Type of output.
    :param layerName: The name of the transformer, which typically
    should be the name of the produced feature.
    :param queryIdCol: Name of column to aggregate upon. It is required.
    :param topN: Filter for limiting the items to calculate the statistics.
    :param sortOrder: Option of 'asc' or 'desc' which defines order
    for listwise operation. Default is 'asc'.
    :param minFilterValue: Minimum value to remove padded values
    defaults to >= 0.
    :nanFillValue: Value to fill NaNs results with. Defaults to 0.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        queryIdCol: Optional[str] = None,
        topN: Optional[int] = None,
        sortOrder: str = "asc",
        minFilterValue: Optional[float] = None,
        nanFillValue: float = 0.0,
    ) -> None:
        super().__init__()
        self._setDefault(
            topN=None,
            sortOrder="asc",
            minFilterValue=None,
            nanFillValue=0,
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
        Calculate the listwise maximum, optionally sorting and
        filtering based on the second input column.
        :param dataset: The dataframe with signals and features.
        :returns: The dataframe dataset with the new feature.
        """
        if not self.isDefined("queryIdCol"):
            raise ValueError("queryIdCol must be set on listwise transformers.")

        # Define the columns to use for the calculation
        with_sort = self.isDefined("inputCols")
        if with_sort:
            val_col_name = self.getInputCols()[0]
            sort_col_name = self.getInputCols()[1]
        else:
            val_col_name = self.getInputCol()
            sort_col_name = None

        check_listwise_columns(
            dataset=dataset,
            query_col_name=self.getQueryIdCol(),
            value_col_name=val_col_name,
            sort_col_name=sort_col_name,
        )

        condition_col, window_spec = get_listwise_condition_and_window(
            query_col=F.col(self.getQueryIdCol()),
            value_col=F.col(val_col_name),
            sort_col=F.col(sort_col_name) if with_sort else None,
            sort_order=self.getSortOrder(),
            sort_top_n=self.getTopN(),
            min_filter_value=self.getMinFilterValue(),
        )

        # Calculate the statistics under the conditions
        dataset = dataset.withColumn(
            self.getOutputCol(),
            F.max(F.when(condition_col, F.col(val_col_name))).over(window_spec),
        )

        # Replace Nulls/Nans
        dataset = dataset.fillna({self.getOutputCol(): self.getNanFillValue()})

        return dataset

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the listwise-maximum transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs an averaging operation.
        """
        return ListMaxLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            top_n=self.getTopN(),
            sort_order=self.getSortOrder(),
            min_filter_value=self.getMinFilterValue(),
            nan_fill_value=self.getNanFillValue(),
            axis=1,
        )
