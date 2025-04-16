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
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import MultiInputSingleOutputParams
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringConcatenateLayer

from .base import BaseTransformer


class StringConcatenateParams(Params):
    """
    Mixin class containing separator parameter needed for string concatenation
    transforms.
    """

    separator = Param(
        Params._dummy(),
        "separator",
        "Value to use as a separator when joining the strings.",
        typeConverter=TypeConverters.toString,
    )

    def setSeparator(self, value: str) -> "StringConcatenateParams":
        """
        Sets the separator parameter.

        :param value: String value to use as a separator when joining the strings.
        :returns: Instance of class mixed in.
        """
        return self._set(separator=value)

    def getSeparator(self) -> str:
        """
        Gets the separator parameter.

        :returns: String value to use as a separator when joining the strings.
        """
        return self.getOrDefault(self.separator)


class StringConcatenateTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
    StringConcatenateParams,
):
    """
    String Concatenate Spark Transformer for use in Spark pipelines.
    This transformer takes in multiple columns and concatenates them together into a
    single column using a separator. Input columns must be of type string.
    """

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        separator: str = "_",
    ) -> None:
        """
        Initializes the string concatenate transformer.
        :param inputCols: columns to concatenate together. Must be of type string.
        :param outputCol: column to output the concatenated string to.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param separator: String to use as a separator when joining the strings.
        If not provided, underscore `_` is used.
        """
        super().__init__()
        kwargs = self._input_kwargs
        self._setDefault(separator="_")
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [StringType()]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        where the value is the result of concatenating the values of the input columns.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        separator = self.getSeparator()

        input_col_names = self.getInputCols()
        input_cols = [F.col(c) for c in input_col_names]

        input_datatypes = [
            self.get_column_datatype(dataset=dataset, column_name=input_col_name)
            for input_col_name in input_col_names
        ]

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_datatypes=input_datatypes,
            input_col_names=input_col_names,
            func=lambda x: F.concat_ws(
                separator, *[x[input_col_name] for input_col_name in input_col_names]
            ),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the concatenate transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs a concatenation.
        """
        return StringConcatenateLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            separator=self.getSeparator(),
        )
