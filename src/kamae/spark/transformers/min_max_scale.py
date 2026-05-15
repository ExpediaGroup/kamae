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

import numpy as np
import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DoubleType, FloatType

from kamae.keras.core.layers import MinMaxScaleLayer
from kamae.params import ParamSpec
from kamae.params.shared_specs import MASK_VALUE_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_array_transform

from .base import BaseTransformer


def _validate_min_max(value: List[float]) -> List[float]:
    """Validates min/max parameter for null values."""
    if None in set(value):
        ids = [i for i, x in enumerate(value) if x is None]
        raise ValueError("Got null values at positions: ", ids)
    return value


class MinMaxScaleTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    MinMax scale transformer for use in Spark pipelines.
    This is used to standardize/transform the input column
    to the range [0, 1] using the minimum and maximum values.

    Formula: (x - min)/(max - min)

    WARNING: If the input is an array, we assume that the array has a constant
    shape across all rows.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = MinMaxScaleLayer
    _params = {
        **MASK_VALUE_PARAMS,
        "min": ParamSpec(
            spark_typeconverter=TypeConverters.toListFloat,
            default=None,
            doc="Minimum of the feature values.",
            validator=_validate_min_max,
        ),
        "max": ParamSpec(
            spark_typeconverter=TypeConverters.toListFloat,
            default=None,
            doc="Maximum of the feature values.",
            validator=_validate_min_max,
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset using the minimum and maximum values
        to standardize the input column.

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

        shift = F.array([F.lit(m) for m in self.getMin()])
        scale = F.array(
            [
                F.lit(1.0 / (m1 - m0) if m1 != m0 else 0.0)
                for m0, m1 in zip(self.getMin(), self.getMax())
            ]
        )

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

    def get_keras_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the Keras layer for the min max transformation.

        :returns: Keras layer with name equal to the layerName parameter
         that performs the standardization.
        """
        np_min = np.array(self.getMin())
        np_max = np.array(self.getMax())
        mask_value = self.getMaskValue()
        return MinMaxScaleLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            min=np_min,
            max=np_max,
            mask_value=mask_value,
        )
