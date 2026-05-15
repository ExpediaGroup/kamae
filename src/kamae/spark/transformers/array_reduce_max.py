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
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType

from kamae.keras.core.layers import ArrayReduceMaxLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_array_transform

from .base import BaseTransformer


class ArrayReduceMaxTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Reduces an array column to its maximum element.

    Input:  Array[Float/Double] of size N.
    Output: Float/Double scalar (the maximum element).

    Returns defaultValue when the array is empty or null.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = ArrayReduceMaxLayer

    _params = {
        "defaultValue": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=0.0,
            doc="Value to return when the array is empty or null",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        input_col = F.col(self.getInputCol())
        default = self.getDefaultValue()

        output_col = single_input_single_output_array_transform(
            input_col=input_col,
            input_col_datatype=self.get_column_datatype(
                dataset=dataset, column_name=self.getInputCol()
            ),
            func=lambda x: F.coalesce(F.array_max(x), F.lit(default)),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
