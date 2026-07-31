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

import keras
import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.sql import DataFrame
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

from kamae.keras.core.backend import ALL_BACKENDS
from kamae.keras.core.layers import ArrayContainsLayer
from kamae.spark.params import MultiInputSingleOutputParams

from .base import BaseTransformer

_NUMERIC_TYPES = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType)


class ArrayContainsTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
):
    """
    ArrayContainsLayer Spark Transformer for use in Spark pipelines.

    This transformer checks whether a scalar value is contained in an array.

    Input:  Two columns `[arrayCol, valueCol]`, where `arrayCol` is an
    `Array[Numeric]` and `valueCol` is a scalar `Numeric`.
    Output: Scalar `Double` equal to `1.0` if the value is in the array,
    else `0.0`.
    """

    supported_backends = ALL_BACKENDS
    jit_compatible = True

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initializes an ArrayContainsTransformer transformer.

        :param inputCols: Input column names, given as `[arrayCol, valueCol]`.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the Keras layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
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

    def setInputCols(self, value: List[str]) -> "ArrayContainsTransformer":
        """
        Sets the input columns, ensuring exactly two are provided:
        `[arrayCol, valueCol]`.

        :param value: List of two input column names.
        :returns: Instance of class with input columns set.
        """
        if len(value) != 2:
            raise ValueError(
                f"Expected 2 input columns, received {len(value)} instead."
            )
        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        equal to `1.0` if the value in `valueCol` is contained in the array in
        `arrayCol`, else `0.0`.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        array_col_name, value_col_name = self.getInputCols()

        array_type = self.get_column_datatype(dataset, array_col_name)
        value_type = self.get_column_datatype(dataset, value_col_name)

        if not isinstance(array_type, ArrayType):
            raise TypeError(
                f"arrayCol '{array_col_name}' must be an ArrayType, got {array_type}"
            )

        elem_type = array_type.elementType
        if not isinstance(elem_type, _NUMERIC_TYPES):
            raise TypeError(
                f"arrayCol '{array_col_name}' element type must be numeric, "
                f"got {elem_type}"
            )

        if not isinstance(value_type, _NUMERIC_TYPES):
            raise TypeError(
                f"valueCol '{value_col_name}' must be numeric, got {value_type}"
            )

        output_col = (
            F.when(
                F.array_contains(F.col(array_col_name), F.col(value_col_name)),
                F.lit(1.0),
            )
            .otherwise(F.lit(0.0))
            .cast(DoubleType())
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_keras_layer(self) -> keras.layers.Layer:
        """
        Gets the Keras layer for the array contains transformer.

        :returns: Keras layer with name equal to the layerName parameter that
        performs the array contains operation.
        """
        return ArrayContainsLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
        )
