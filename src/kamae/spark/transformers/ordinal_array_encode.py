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
from pyspark.sql.types import ArrayType, DataType, IntegerType, StringType

from kamae.keras.tensorflow.layers import OrdinalArrayEncodeLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import (
    ordinal_array_encode_udf,
    single_input_single_output_array_udf_transform,
)

from .base import BaseTransformer


class OrdinalArrayEncodeTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Transformer that encodes an array of strings into an array of integers.

    The transformer will map each unique string in the array to an integer,
    according to the order in which they appear in the array. It will also
    ignore the pad value if specified.
    """

    _compatible_dtypes = [StringType()]
    _keras_layer_class = OrdinalArrayEncodeLayer
    _params = {
        "padValue": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default=None,
            doc="The value to be considered as padding.",
        ),
        "axis": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=-1,
            doc="The axis along which to encode the array. Defaults to -1.",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Performs the ordinal encoding on the input dataset.
        Example:
         dataset = spark.Dataframe(
            [
                ['a', 'a', 'a', 'b', 'c', '-1', '-1', '-1'],
                ['x', 'x', 'x', 'x', 'y', 'z', '-1', '-1'],
            ],
            'input_col'
         )
         Output: spark.Dataframe(
            [
                ['a', 'a', 'a', 'b', 'c', '-1', '-1', '-1'],
                ['x', 'x', 'x', 'x', 'y', 'z', '-1', '-1'],
            ],
            [
                [0, 0, 0, 1, 2, -1, -1, -1],
                [0, 0, 0, 0, 1, 2, -1, -1],
            ],
            'input_col', 'output_col'
        )
        :param dataset: The input dataframe.
        :returns: Transformed pyspark dataframe.
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        if not isinstance(input_datatype, ArrayType):
            raise ValueError(
                f"""Input column {self.getInputCol()} must be of ArrayType.
                        Got {input_datatype} instead."""
            )

        output_col = single_input_single_output_array_udf_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: ordinal_array_encode_udf(x, self.getPadValue()),
            udf_return_element_datatype=IntegerType(),
        )

        return dataset.withColumn(
            self.getOutputCol(),
            output_col,
        )
