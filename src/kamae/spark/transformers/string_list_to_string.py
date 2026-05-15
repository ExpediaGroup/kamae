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
from pyspark.sql.types import ArrayType, DataType, StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringListToStringLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_array_transform

from .base import BaseTransformer


class StringListToStringTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    StringListToStringLayer Spark Transformer for use in Spark pipelines.
    This transformer takes a column of string lists and joins them into a single string.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    # Overrides codegen: Keras layer has axis/keepdims params with no Spark equivalent
    _keras_layer_class = None
    _params = {
        "separator": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="",
            doc="Separator to use when joining the string list.",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which applies the given stringCaseType to the input column.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        separator = self.getSeparator()

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        if not isinstance(input_datatype, ArrayType):
            raise TypeError(
                f"""Input column {self.getInputCol()} must be of type ArrayType,
                not {input_datatype}."""
            )
        output_col = single_input_single_output_array_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: F.concat_ws(separator, x),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_keras_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the Keras layer for the StringListToStringLayer transformer.

        :returns: Keras layer with name equal to the layerName parameter that
        joins the string list.
        """
        # Hardcodes axis=-1, keepdims=True to match Spark transform behaviour
        return StringListToStringLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            separator=self.getSeparator(),
            axis=-1,
            keepdims=True,
        )
