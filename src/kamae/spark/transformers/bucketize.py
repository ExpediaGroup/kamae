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
from typing import List, Optional, Union

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType, IntegerType, LongType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import BucketizeLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils.transform_utils import (
    single_input_single_output_scalar_udf_transform,
)

from .base import BaseTransformer


def _validate_splits(value: List[float]) -> List[float]:
    if value is not None and value != sorted(value):
        raise ValueError("`splits` argument must be a sorted list!")
    return value


class BucketizeTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    BucketizeLayer Spark Transformer for use in Spark pipelines.
    This transformer buckets a numerical column into bins.
    Buckets will be created based on the splits parameter.
    The bins are integer values starting at 1 and ending at the number of splits + 1.
    The 0 index is reserved for masking/padding.
    """

    jit_compatible = True

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [IntegerType(), LongType(), FloatType(), DoubleType()]
    _keras_layer_class = BucketizeLayer
    _params = {
        "splits": ParamSpec(
            spark_typeconverter=TypeConverters.toListFloat,
            default=None,
            doc="List of split points for bucketing.",
            validator=_validate_splits,
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the `inputCol` bucketed into bins accoring to the `splits` parameter.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        splits = self.getSplits()
        # We need to create a UDF to perform binary search on the splits.

        def bucketize(value: Optional[Union[float, int]]) -> Optional[int]:
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
