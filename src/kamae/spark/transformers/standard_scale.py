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

import keras
import numpy as np
import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DoubleType, FloatType

from kamae.keras.core.layers import StandardScaleLayer
from kamae.params.shared_specs import MASK_VALUE_PARAMS, STANDARD_SCALE_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_array_transform

from .base import BaseTransformer


class StandardScaleTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Standard scaler transformer for use in Spark pipelines.
    This is used to standardize/transform the input column
    using the mean and standard deviation.

    WARNING: If the input is an array, we assume that the array has a constant
    shape across all rows.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    # Codegen cannot bridge this: Spark uses stddev, Keras uses variance
    _keras_layer_class = None
    _params = {
        **MASK_VALUE_PARAMS,
        **STANDARD_SCALE_PARAMS,
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset using the mean and standard deviation
        to standardize the input column. If a mask value is set, it is used
        to ignore elements in the dataset with that value, and they will remain
        unchanged in the standardization process.

        :param dataset: Pyspark dataframe to transform.
        :returns: Pyspark dataframe with the input column standardized,
         named as the output column.
        """
        original_input_datatype = self.get_column_datatype(dataset, self.getInputCol())
        if not isinstance(original_input_datatype, ArrayType):
            input_col = F.array(F.col(self.getInputCol()))
            input_datatype = ArrayType(original_input_datatype)
        else:
            input_col = F.col(self.getInputCol())
            input_datatype = original_input_datatype

        shift = F.array([F.lit(m) for m in self.getMean()])
        scale = F.array([F.lit(1.0 / s if s != 0 else 0.0) for s in self.getStddev()])

        output_col = single_input_single_output_array_transform(
            input_col=input_col,
            input_col_datatype=input_datatype,
            func=lambda x: F.transform(
                x,
                lambda y, i: F.when(y == self.getMaskValue(), y).otherwise(
                    (y - F.lit(shift)[i]) * F.lit(scale)[i]
                ),
            ),
        )

        if not isinstance(original_input_datatype, ArrayType):
            output_col = output_col.getItem(0)

        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_keras_layer(self) -> keras.layers.Layer:
        """
        Gets the Keras layer for the standard scaler transformer.

        :returns: Keras layer with name equal to the layerName parameter
         that performs the standardization.
        """
        # Overrides codegen: Spark param is stddev but Keras layer requires variance
        np_mean = np.array(self.getMean())
        np_variance = np.array(self.getStddev()) ** 2
        mask_value = self.getMaskValue()
        return StandardScaleLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            mean=np_mean,
            variance=np_variance,
            mask_value=mask_value,
        )
