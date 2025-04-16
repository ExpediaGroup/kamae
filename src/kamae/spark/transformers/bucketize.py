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
from bisect import bisect_right
from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType, IntegerType, LongType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils.transform_utils import (
    single_input_single_output_scalar_udf_transform,
)
from kamae.tensorflow.layers import BucketizeLayer

from .base import BaseTransformer


class BucketizeParams(Params):
    """
    Mixin class containing splits parameter needed for bucketing.
    """

    splits = Param(
        Params._dummy(),
        "splits",
        "List of split points for bucketing.",
        typeConverter=TypeConverters.toListFloat,
    )

    @staticmethod
    def check_splits_sorted(splits: List[float]):
        """
        Checks that the splits parameter is sorted.

        :param splits: List of float values to use for bucketing.
        """
        if splits is not None and splits != sorted(splits):
            raise ValueError("`splits` argument must be a sorted list!")

    def setSplits(self, value: List[float]) -> "BucketizeParams":
        """
        Sets the splits parameter.

        :param value: List of float values to use for bucketing.
        :returns: Instance of class mixed in.
        """
        self.check_splits_sorted(value)
        return self._set(splits=value)

    def getSplits(self) -> List[float]:
        """
        Gets the splits parameter.

        :returns: List of float values to use for bucketing.
        """
        return self.getOrDefault(self.splits)


class BucketizeTransformer(
    BaseTransformer,
    BucketizeParams,
    SingleInputSingleOutputParams,
):
    """
    BucketizeLayer Spark Transformer for use in Spark pipelines.
    This transformer buckets a numerical column into bins.
    Buckets will be created based on the splits parameter.
    The bins are integer values starting at 1 and ending at the number of splits + 1.
    The 0 index is reserved for masking/padding.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        splits: Optional[List[float]] = None,
    ) -> None:
        """
        Initializes an BucketizeTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param splits: List of float values to use for bucketing.
        :returns: None - class instantiated.
        """
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
        return [IntegerType(), LongType(), FloatType(), DoubleType()]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the `inputCol` bucketed into bins accoring to the `splits` parameter.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        splits = self.getSplits()
        # We need to create a UDF to perform binary search on the splits.

        def bucketize(value):
            # If null, keep null. There is no best bucket to place these into.
            if value is None:
                return None
            # We add 1 because we want to reserve the 0 index for mask/padding.
            return bisect_right(splits, value) + 1

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_udf_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: bucketize(x),
            udf_return_element_datatype=IntegerType(),
        )

        return dataset.withColumn(
            self.getOutputCol(),
            output_col,
        )

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the BucketizeLayer transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs a bucketing operation.
        """
        return BucketizeLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            splits=self.getSplits(),
        )
