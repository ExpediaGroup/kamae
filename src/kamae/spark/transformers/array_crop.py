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
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import BooleanType, DataType, FloatType, IntegerType, StringType

from kamae.spark.params import PadValueParams, SingleInputSingleOutputParams
from kamae.spark.utils import (
    get_array_nesting_level_and_element_dtype,
    single_input_single_output_array_transform,
)
from kamae.tensorflow.layers import ArrayCropLayer

from .base import BaseTransformer


class ArrayCropParams(PadValueParams):
    """
    Mixin class containing pad value parameters needed
    for array crop transformers.
    """

    arrayLength = Param(
        PadValueParams._dummy(),
        "arrayLength",
        "The length to crop or pad the arrays to. Defaults to 128.",
        typeConverter=TypeConverters.toInt,
    )

    def setArrayLength(self, value: int) -> "ArrayCropParams":
        """
        Sets the parameter array length to the given value.
        :param value: array length.
        :returns: Instance of class mixed in.
        """
        if value < 1:
            raise ValueError("Array length must be greater than 0.")
        return self._set(arrayLength=value)

    def getArrayLength(self) -> int:
        """
        Gets the array length parameter.
        :returns: array length.
        """
        return self.getOrDefault(self.arrayLength)


class ArrayCropTransformer(
    BaseTransformer, SingleInputSingleOutputParams, ArrayCropParams
):
    """
    Transformer that reshapes arrays into consistent shapes by
    either cropping or padding.

    If the tensor is shorter than the specified length, it is
    padded with specified pad value.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[Union[str, int, float]] = None,
        outputDtype: Optional[Union[str, int, float]] = None,
        layerName: Optional[str] = None,
        arrayLength: Optional[int] = 128,
        padValue: Optional[Union[str, int, float]] = None,
    ) -> None:
        """
        Initialises the ArrayCropTransformer
        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        :param arrayLength: The length to crop or pad the arrays to. Defaults to 128.
        :param padValue: The value pad the arrays with. Defaults to `None`.
        :returns: None
        """
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self._pad_type_to_valid_element_types = {
            "int": ["int", "bigint", "smallint"],
            "float": ["float", "double", "decimal(10,0)"],
            "string": ["string"],
            "boolean": ["boolean"],
        }

    @staticmethod
    def _get_pad_value_type(pad_value) -> Optional[DataType]:
        if isinstance(pad_value, int):
            return IntegerType()
        if isinstance(pad_value, str):
            return StringType()
        if isinstance(pad_value, float):
            return FloatType()
        if isinstance(pad_value, bool):
            return BooleanType()
        raise TypeError(f"Unsupported pad value type: {type(pad_value)}")

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return None

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

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that performs the array cropping and padding.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that performs the array cropping and padding operation.
        """
        return ArrayCropLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            array_length=self.getArrayLength(),
            pad_value=self.getPadValue(),
        )
