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

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType

from kamae.keras.core.layers import LogLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


class LogTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Log Spark Transformer for use in Spark pipelines.
    This transformer applies a log(alpha + x) transform to the input column.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = LogLayer
    _params = {
        "alpha": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=0.0,
            doc="Value to use in log transform: log(alpha + x)",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        alpha = self.getAlpha()
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: F.log(x + F.lit(alpha)),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
