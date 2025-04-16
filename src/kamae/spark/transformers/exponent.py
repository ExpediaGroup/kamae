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
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType

from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.tensorflow.layers import ExponentLayer

from .base import BaseTransformer


class ExponentParams(Params):
    """
    Mixin class containing alpha parameter needed for exponent transform layers.
    """

    exponent = Param(
        Params._dummy(),
        "exponent",
        "Value to use in exponent transform: x^exponent",
        typeConverter=TypeConverters.toFloat,
    )

    def setExponent(self, value: float) -> "ExponentParams":
        """
        Sets the exponent parameter.

        :param value: Float value to use in exponent transform: x^exponent.
        :returns: Instance of class mixed in.
        """
        return self._set(exponent=value)

    def getExponent(self) -> float:
        """
        Gets the exponent parameter.

        :returns: Float value of exponent used in exponent transform.
        """
        return self.getOrDefault(self.exponent)


class ExponentTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
    ExponentParams,
):
    """
    Exponent Spark Transformer for use in Spark pipelines.
    This transformer applies x^exponent in the case of single input and or x^y in the
    case of two inputs.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        exponent: Optional[float] = None,
    ) -> None:
        """
        Initializes an ExponentTransformer transformer.

        :param inputCol: Input column name. Only used if inputCols is not specified.
        If specified, we raise this column by the exponent.
        :param inputCols: Input column names. If provided, we raise the first column by
        the second column. Must have exactly two columns.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param exponent: Optional exponent/power to raise the input to. If not provided,
        then two input columns are required.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(exponent=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setInputCols(self, value: List[str]) -> "ExponentTransformer":
        """
        Overrides setting the input columns for the transformer.
        Throws an error if we do not have exactly two input columns.

        :param value: List of input columns.
        :returns: Class instance.
        """
        if len(value) != 2:
            raise ValueError(
                """When setting inputCols for ExponentTransformer,
                there must be exactly two input columns."""
            )
        return self._set(inputCols=value)

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
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which applies the exponent transform to the input column(s).

        If one column is provided via inputCol, we raise that column to the power of
        the exponent parameter. If two columns are provided via inputCols,
        we raise the first column to the power of the second column.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self.get_multiple_input_cols(
            constant_param_name="exponent", input_cols_limit=2
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
            func=lambda x: F.pow(x[input_col_names[0]], x[input_col_names[1]]),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the exp value transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs an exp value operation.
        """
        return ExponentLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            exponent=self.getExponent(),
        )
