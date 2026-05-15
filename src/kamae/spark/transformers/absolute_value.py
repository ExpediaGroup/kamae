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

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.keras.core.layers import AbsoluteValueLayer
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


class AbsoluteValueTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Absolute value Spark Transformer for use in Spark pipelines.
    This transformer applies abs(x) operation to the input.
    """

    jit_compatible = True

    _compatible_dtypes = [
        FloatType(),
        DoubleType(),
        IntegerType(),
        LongType(),
        ShortType(),
        ByteType(),
    ]
    _keras_layer_class = AbsoluteValueLayer

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which applies abs(`inputCol`).

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_data_type = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_data_type,
            func=lambda x: F.abs(x),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
