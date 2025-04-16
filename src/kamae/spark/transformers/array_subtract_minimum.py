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
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import (
    ArrayType,
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_array_transform
from kamae.tensorflow.layers import ArraySubtractMinimumLayer

from .base import BaseTransformer


class ArraySubtractMinimumParams(Params):
    """
    Mixin class containing pad value parameters needed
    for array subtract min transformers.
    """

    padValue = Param(
        Params._dummy(),
        "padValue",
        "The value to be considered as padding. Defaults to `None`.",
        typeConverter=TypeConverters.toFloat,
    )

    def setPadValue(self, value: float) -> "ArraySubtractMinimumParams":
        """
        Sets the parameter pad value to the given value.

        :param value: pad value.
        :returns: Instance of class mixed in.
        """
        return self._set(padValue=value)

    def getPadValue(self) -> float:
        """
        Gets the pad value parameter.

        :returns: float pad value.
        """
        return self.getOrDefault(self.padValue)


class ArraySubtractMinimumTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    ArraySubtractMinimumParams,
):
    """
    ArraySubtractMinimumTransformer that computes the difference within an array from
    the minimum non-paded element in the input tensor. The calculation preserves the pad
    value elements.

    The main use case in mind for this is working with an array of timestamps.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        padValue: Optional[float] = None,
    ) -> None:
        """
        Initialise the ArraySubtractMinimumTransformer

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        :param padValue: The value to be considered as padding. Defaults to `None`.
        :returns: None
        """
        super().__init__()
        self._setDefault(padValue=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [
            FloatType(),
            DoubleType(),
            ByteType(),
            ShortType(),
            IntegerType(),
            LongType(),
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Performs the calculation of the differences on the input dataset.

        Example:
         dataset = spark.Dataframe(
            [
                [19, 18, 13, 11, 10, -1, -1, -1],
                [12, 2, 1, -1, -1, -1, -1, -1],
            ],
            'input_col'
         )
         Output: spark.Dataframe(
            [
                [19, 18, 13, 11, 10, -1, -1, -1],
                [12, 2, 1, -1, -1, -1, -1, -1],
            ],
            [
                [9, 8, 3, 1, 0, -1, -1, -1],
                [11, 1, 0, -1, -1, -1, -1, -1],
            ],
            'input_col', 'output_col'
        )

        :param dataset: The input dataframe.
        :returns: Transformed pyspark dataframe.
        """
        input_column_type = self.get_column_datatype(dataset, self.getInputCol())
        if not isinstance(input_column_type, ArrayType):
            raise ValueError(
                f"""Input column {self.getInputCol()} must be of ArrayType.
                Got {input_column_type} instead."""
            )
        padded_value = self.getPadValue()

        def array_subtract_min(x: Column, pad_value: Optional[float]) -> Column:
            if pad_value is None:
                return F.transform(x, lambda y: y - F.array_min(x))
            else:
                return F.transform(
                    x,
                    lambda y: F.when(
                        y != F.lit(pad_value),
                        y - F.array_min(F.filter(x, lambda z: z != F.lit(pad_value))),
                    ).otherwise(y),
                )

        array_subtract = single_input_single_output_array_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_column_type,
            func=lambda x: array_subtract_min(x, padded_value),
        )
        return dataset.withColumn(self.getOutputCol(), array_subtract)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the sequential difference transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
        performs the sequential difference operation.
        """
        return ArraySubtractMinimumLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            pad_value=self.getPadValue(),
        )
