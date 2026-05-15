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
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType, StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import ListMaxLayer
from kamae.params.shared_specs import (
    LISTWISE_FILTER_PARAMS,
    LISTWISE_PARAMS,
    LISTWISE_SEGMENT_PARAMS,
)
from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import check_and_apply_listwise_op

from .base import BaseTransformer


class ListMaxTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
):
    """
    Calculate the listwise maximum across the query id column.
    - If inputCol is set, the transformer calculates the maximum of the input column
    based on all the items with the same query id column value.
    - If inputCols is set, behaviour depends on the value of withSegment:
        - If withSegment = True: the transformer calculates the maximum of the first column
        with the same query id column value, segmented by values of the second column.

        Example: calculate the maximum price of hotels within star ratings, in the same query.

        - If withSegment = False: the transformer calculates the maximum of the first column
        with the same query id column value, based on second column's topN items.
        When using the second input as sorting column, topN must be provided.
        By using the topN items to calculate the statistics, we can better approximate
        the real statistics in production. A large enough topN should be used, to obtain a
        good approximation of the statistics, and an important feature to sort on, such as
        item's production.

        Example: calculate the maximum price in the same query, based on the top N
        items sorted by descending production.

    :param inputCol: Value column, on which to calculate the maximum.
    :param inputCols: Input column names.
    - The first is the value column, on which to calculate the maximum.
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

    jit_compatible = True

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [FloatType(), DoubleType(), StringType()]
    _keras_layer_class = ListMaxLayer
    _params = {
        **LISTWISE_PARAMS,
        **LISTWISE_SEGMENT_PARAMS,
        **LISTWISE_FILTER_PARAMS,
    }

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
                F.max,
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
