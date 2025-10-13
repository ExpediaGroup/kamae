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
    StringType,
)

from kamae.spark.params import (
    ListwiseStatisticsParams,
    MultiInputSingleOutputParams,
    NanFillValueParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import check_and_apply_listwise_op
from kamae.tensorflow.layers import ListMeanLayer

from .base import BaseTransformer


class ListMeanTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
    ListwiseStatisticsParams,
    NanFillValueParams,
):
    """
    Calculate the listwise mean across the query id column.
    - If inputCol is set, the transformer calculates the mean of the input column
    based on all the items with the same query id column value.
    - If inputCols is set, behaviour depends on the value of withSegment:
        - If withSegment = True: the transformer calculates the mean of the first column
        with the same query id column value, segmented by values of the second column.

        Example: calculate the mean price of hotels within star ratings, in the same query.

        - If withSegment = False: the transformer calculates the mean of the first column
        with the same query id column value, based on second column's topN items.
        When using the second input as sorting column, topN must be provided.
        By using the topN items to calculate the statistics, we can better approximate
        the real statistics in production. A large enough topN should be used, to obtain a
        good approximation of the statistics, and an important feature to sort on, such as
        item's production.

        Example: calculate the mean price in the same query, based on the top N
        items sorted by descending production.

    :param inputCol: Value column, on which to calculate the mean.
    :param inputCols: Input column names.
    - The first is the value column, on which to calculate the mean.
    - The second is the sort or segment column. The role of the second input is governed
    by the value of withSegment as described above.
    :param outputCol: Name of output col.
    :param inputDtype: Data Type of input.
    :param outputDtype: Data Type of output.
    :param layerName: The name of the transformer, which typically
    should be the name of the produced feature.
    :param queryIdCol: Name of column to aggregate upon. It is required.
    :param topN: Filter for limiting the items to calculate the statistics. Not used when withSegment = True.
    :param sortOrder: Option of 'asc' or 'desc' which defines order
    for listwise operation. Default is 'asc'. Not used when withSegment = True.
    :param withSegment: Whether to use the second input column to partition the statistic
    calculation. Defaults to False.
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
        withSegment: bool = False,
        minFilterValue: Optional[float] = None,
        nanFillValue: float = 0.0,
    ) -> None:
        super().__init__()
        self._setDefault(
            topN=None,
            sortOrder="asc",
            minFilterValue=None,
            nanFillValue=0,
            withSegment=False,
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
            StringType(),
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Calculate the listwise mean, optionally sorting and
        filtering based on the second input column.
        :param dataset: The dataframe with signals and features.
        :returns: The dataframe dataset with the new feature.
        """
        if not self.isDefined("queryIdCol"):
            raise ValueError("queryIdCol must be set on listwise transformers.")

        # Define the columns to use for the calculation
        if self.isDefined("inputCols"):
            with_segment = self.getWithSegment()
            if with_segment:
                val_col_name = self.getInputCols()[0]
                segment_col_name = self.getInputCols()[1]
                sort_col_name = None
            else:
                val_col_name = self.getInputCols()[0]
                sort_col_name = self.getInputCols()[1]
                segment_col_name = None
        else:
            val_col_name = self.getInputCol()
            sort_col_name = None
            segment_col_name = None

        dataset = dataset.withColumn(
            self.getOutputCol(),
            check_and_apply_listwise_op(
                dataset,
                F.mean,
                self.getQueryIdCol(),
                val_col_name,
                sort_col_name,
                segment_col_name,
                self.getSortOrder(),
                self.getTopN(),
                self.getMinFilterValue(),
            ),
        )

        # Replace Nulls/Nans
        dataset = dataset.fillna({self.getOutputCol(): self.getNanFillValue()})

        return dataset

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the listwise-mean transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs an averaging operation.
        """
        return ListMeanLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            top_n=self.getTopN(),
            sort_order=self.getSortOrder(),
            with_segment=self.getWithSegment(),
            min_filter_value=self.getMinFilterValue(),
            nan_fill_value=self.getNanFillValue(),
            axis=1,
        )
