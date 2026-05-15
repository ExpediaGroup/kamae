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
from functools import reduce

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DoubleType, FloatType

from kamae.keras.core.layers import DivideLayer
from kamae.params import _UNSET, ParamSpec
from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform

from .base import BaseTransformer


class DivideTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
):
    """
    DivideLayer Spark Transformer for use in Spark pipelines.
    This transformer divides a column by a constant or another column.
    """

    jit_compatible = True

    _compatible_dtypes = [
        FloatType(),
        DoubleType(),
    ]
    _keras_layer_class = None
    _params = {
        "mathFloatConstant": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=_UNSET,
            doc="Float constant to divide by.",
        ),
    }

    def get_keras_layer(self):
        return DivideLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            divisor=self.getMathFloatConstant(),
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the division of the input columns or the input column by mathFloatConstant.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self.get_multiple_input_cols(
            constant_param_name="mathFloatConstant",
        )
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]

        def divide_no_nan(column1: Column, column2: Column) -> Column:
            """
            Divide two columns, and if the result is NaN, return 0.0 instead
            """
            return F.coalesce(column1 / column2, F.lit(0.0))

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            func=lambda x: reduce(divide_no_nan, [x[c] for c in input_col_names]),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
