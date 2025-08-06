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

from kamae.spark.params import ListwiseStatisticsParams, MultiInputSingleOutputParams
from kamae.spark.utils import check_listwise_columns, get_listwise_condition_and_window
from kamae.tensorflow.layers import SegmentMeanLayer

from .base import BaseTransformer


class SegmentMeanTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
    ListwiseStatisticsParams,
):
    """
    Calculate the listwise mean across the query id and segment column combinations.

    Example: calculate the mean price of rates within rooms in the same query.

    :param inputCols: Input column name.
    - The first is the value column, on which to calculate the mean.
    - The second is the segment column, based on which to segment the items.
    :param outputCol: Name of output col.
    :param inputDtype: Data Type of input.
    :param outputDtype: Data Type of output.
    :param layerName: The name of the transformer, which typically
    should be the name of the produced feature.
    :param queryIdCol: Name of column to denoting a single query. It is required.
    """

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        queryIdCol: Optional[str] = None,
    ) -> None:
        super().__init__()
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
            StringType(),
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Calculate the mean, segmented based on the second input column.
        :param dataset: The dataframe with signals and features.
        :returns: The dataframe dataset with the new feature.
        """

        val_col_name = self.getInputCols()[0]
        segment_col_name = self.getInputCols()[1]

        if not self.isDefined("queryIdCol"):
            raise ValueError("queryIdCol must be set on listwise transformers.")

        check_listwise_columns(
            dataset=dataset,
            query_col_name=self.getQueryIdCol(),
            value_col_name=val_col_name,
            segment_col_name=segment_col_name,
        )

        condition_col, window_spec = get_listwise_condition_and_window(
            query_col=F.col(self.getQueryIdCol()),
            value_col=F.col(val_col_name),
            segment_col=F.col(segment_col_name),
        )

        # Calculate the statistics under the conditions
        dataset = dataset.withColumn(
            self.getOutputCol(),
            F.mean(F.col(val_col_name)).over(window_spec),
        )

        return dataset

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the segment mean transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs an averaging operation.
        """

        return SegmentMeanLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            axis=1,
        )
