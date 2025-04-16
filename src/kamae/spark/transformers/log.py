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

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import LogLayer

from .base import BaseTransformer


class LogParams(Params):
    """
    Mixin class containing alpha parameter needed for log transform layers.
    """

    alpha = Param(
        Params._dummy(),
        "alpha",
        "Value to use in log transform: log(alpha + x)",
        typeConverter=TypeConverters.toFloat,
    )

    def setAlpha(self, value: float) -> "LogParams":
        """
        Sets the alpha parameter.

        :param value: Float value to use in log transform: log(alpha + x).
        :returns: Instance of class mixed in.
        """
        return self._set(alpha=value)

    def getAlpha(self) -> float:
        """
        Gets the alpha parameter.

        :returns: Float value of alpha used in log transform.
        """
        return self.getOrDefault(self.alpha)


class LogTransformer(
    BaseTransformer,
    LogParams,
    SingleInputSingleOutputParams,
):
    """
    Log Spark Transformer for use in Spark pipelines.
    This transformer applies a log(alpha + x) transform to the input column.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        alpha: float = 0.0,
    ) -> None:
        """
        Instantiates a LogTransformer transformer. Sets the default values of:

        - alpha: 0

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param alpha: Value to use in log transform: log(alpha + x). Default is 0.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(alpha=0.0)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [FloatType(), DoubleType()]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column named outputCol with the
        log transform of the inputCol.

        :param dataset: Pyspark DataFrame to transform.
        :returns: Transformed pyspark dataFrame.
        """
        alpha = self.getAlpha()
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: F.log(x + F.lit(alpha)),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that performs the log transform.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that performs the log(alpha + x) operation.
        """
        return LogLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            alpha=self.getAlpha(),
        )
