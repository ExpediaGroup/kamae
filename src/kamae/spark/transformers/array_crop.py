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

from typing import List, Optional, Union

import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import BooleanType, DataType, FloatType, IntegerType, StringType

from kamae.keras.core.layers import ArrayCropLayer
from kamae.params import ParamSpec
from kamae.spark.params import _UNSET, SingleInputSingleOutputParams
from kamae.spark.utils import (
    get_array_nesting_level_and_element_dtype,
    single_input_single_output_array_transform,
)

from .base import BaseTransformer


class ArrayCropTransformer(BaseTransformer, SingleInputSingleOutputParams):
    """
    Transformer that reshapes arrays into consistent shapes by
    either cropping or padding.

    If the tensor is shorter than the specified length, it is
    padded with specified pad value.
    """

    jit_compatible = True

    _compatible_dtypes = None
    _keras_layer_class = ArrayCropLayer
    _params = {
        "arrayLength": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=128,
            doc="The length to crop or pad the arrays to. Defaults to 128.",
            validator=lambda self, value: (
                value
                if value >= 1
                else (_ for _ in ()).throw(
                    ValueError("Array length must be greater than 0.")
                )
            ),
        ),
        "padValue": ParamSpec(
            spark_typeconverter=TypeConverters.identity,
            default=None,
            doc="The value pad the arrays with. Defaults to `None`.",
        ),
    }

    _pad_type_to_valid_element_types = {
        "int": ["int", "bigint", "smallint"],
        "float": ["float", "double", "decimal(10,0)"],
        "string": ["string"],
        "boolean": ["boolean"],
    }

    @staticmethod
    def _get_pad_value_type(
        pad_value: Union[int, str, float, bool]
    ) -> Optional[DataType]:
        if isinstance(pad_value, int):
            return IntegerType()
        if isinstance(pad_value, str):
            return StringType()
        if isinstance(pad_value, float):
            return FloatType()
        if isinstance(pad_value, bool):
            return BooleanType()
        raise TypeError(f"Unsupported pad value type: {type(pad_value)}")

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Performs the cropping and/or padding on the input dataset.
        Example, crop to length 3, with value '-1':

         dataset = spark.Dataframe(
            [
                ['a', 'a', 'a', 'b', 'c'],
                ['x', 'z', 'y'],
                ['a', 'b',],
                ['a', 'x', 'a', 'b',],
                []
            ],
            'input_col'
         )
         Output: spark.Dataframe(
            [
                ['a', 'a', 'a', 'b', 'c'],
                ['x', 'z', 'y'],
                ['a', 'b',],
                ['a', 'x', 'a', 'b',],
                []
            ],
            [
                ['a', 'a', 'a'],
                ['x', 'z', 'y'],
                ['a', 'b', '-1'],
                ['a', 'x', 'a'],
                ['-1', '-1', '-1']
            ],
            'input_col', 'output_col'
        )
        :param dataset: The input dataframe.
        :returns: Transformed pyspark dataframe.
        """
        pad_value_spark_type = self._get_pad_value_type(self.getPadValue())
        input_col_type = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        input_col_element_type = get_array_nesting_level_and_element_dtype(
            input_col_type
        )[1]

        if (
            input_col_element_type.simpleString()
            not in self._pad_type_to_valid_element_types[
                pad_value_spark_type.simpleString()
            ]
        ):
            raise ValueError(
                f"""
            The pad value type '{type(pad_value_spark_type)}' does
            not match the element type of the input
            column '{type(input_col_element_type)}'.
            """
            )

        output_col = single_input_single_output_array_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_col_type,
            func=lambda x: F.concat(
                F.slice(x, 1, self.getArrayLength()),
                F.array_repeat(
                    F.lit(self.getPadValue()),
                    self.getArrayLength() - F.size(x),
                ),
            ),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
