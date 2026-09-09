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
from kamae.spark.params import (
    MathFloatConstantParams,
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import single_input_single_output_array_transform

from .base import BaseTransformer


class ArrayContainsTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
    MathFloatConstantParams,
):
    """
    ArrayContainsLayer Spark Transformer for use in Spark pipelines.

    Checks whether a scalar value is present in a (possibly nested) numeric
    array. The value is either a second input column (`inputCols`) or the
    `mathFloatConstant` (`inputCol`). Outputs a boolean; set `outputDtype` to
    cast the result.

    Example:

    >>> df.show()
    +---------+
    |        a|
    +---------+
    |[1, 2, 3]|
    |[4, 5, 6]|
    +---------+
    >>> t = ArrayContainsTransformer(inputCol="a", outputCol="b", mathFloatConstant=2)
    >>> t.transform(df).show()
    +---------+-----+
    |        a|    b|
    +---------+-----+
    |[1, 2, 3]| true|
    |[4, 5, 6]|false|
    +---------+-----+
    """

    supported_backends = ALL_BACKENDS
    jit_compatible = True

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        mathFloatConstant: Optional[float] = None,
    ) -> None:
        """
        Initializes an ArrayContainsTransformer transformer.

        :param inputCol: Input array column name. Only used if inputCols is not
        specified. If specified, we check whether `mathFloatConstant` is contained
        in this array column.
        :param inputCols: Input column names, given as `[arrayCol, valueCol]`.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the Keras layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param mathFloatConstant: Optional constant value to check for. Used with
        `inputCol`. If not provided, then `inputCols` is required.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(mathFloatConstant=None)
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
            raise ValueError(f"Expected 2 input cols, received {len(value)} instead.")

        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Adds `outputCol`, `True` where the value is present in the innermost
        array. The value is a second input column or `mathFloatConstant`.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        array_col, value_col = self.get_multiple_input_cols("mathFloatConstant", 2)
        df = dataset.select(array_col, value_col)

        output_col = single_input_single_output_array_transform(
            input_col=array_col,
            input_col_datatype=self.get_column_datatype(df, df.columns[0]),
            func=lambda x: F.array_contains(x, value_col),
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
            axis=-1,
            keepdims=True,
            value_constant=self.getMathFloatConstant(),
        )
