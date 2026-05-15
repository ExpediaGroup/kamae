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
from operator import mul

import keras
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.keras.core.layers import MultiplyLayer
from kamae.params import _UNSET, ParamSpec
from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform

from .base import BaseTransformer


class MultiplyTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
):
    """
    MultiplyLayer Spark Transformer for use in Spark pipelines.
    This transformer multiplies a column by a constant or another column.
    """

    jit_compatible = True

    _compatible_dtypes = [
        FloatType(),
        DoubleType(),
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType(),
    ]
    _keras_layer_class = None
    _params = {
        "mathFloatConstant": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=_UNSET,
            doc="Float constant to multiply by.",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the product of the input columns or the input column and mathFloatConstant.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self.get_multiple_input_cols(
            constant_param_name="mathFloatConstant"
        )
        # input_cols can contain either actual columns or lit(constants). In order to
        # determine the datatype of the input columns, we select them from the dataset
        # first.
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            func=lambda x: reduce(mul, [x[c] for c in input_col_names]),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_keras_layer(self) -> keras.layers.Layer:
        """
        Gets the Keras layer for the multiply transformer.

        :returns: Keras layer with name equal to the layerName parameter that
         performs a multiply operation.
        """
        return MultiplyLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            multiplier=self.getMathFloatConstant(),
        )
