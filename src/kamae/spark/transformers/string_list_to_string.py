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

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_array_transform
from kamae.tensorflow.layers import StringListToStringLayer

from .base import BaseTransformer


class StringListToStringParams(Params):
    """
    Mixin class containing separator parameter needed for
    string list to string transforms.
    """

    separator = Param(
        Params._dummy(),
        "separator",
        "Separator to use when joining the string list.",
        typeConverter=TypeConverters.toString,
    )

    def getSeparator(self) -> str:
        """
        Gets the separator to use when joining the string list.

        :returns: Separator to use when joining the string list.
        """
        return self.getOrDefault(self.separator)

    def setSeparator(self, value: str) -> "StringListToStringParams":
        """
        Sets the separator to use when joining the string list.

        :param value: Separator to use when joining the string list.
        :returns: Instance of class mixed in.
        """
        return self._set(separator=value)


class StringListToStringTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    StringListToStringParams,
):
    """
    StringListToStringLayer Spark Transformer for use in Spark pipelines.
    This transformer takes a column of string lists and joins them into a single string.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        separator: str = "",
    ) -> None:
        """
        Initializes an StringListToStringTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param separator: Separator to use when joining the string list.
        Default is the empty string.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(separator="")
        kwargs = self._input_kwargs
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

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the StringListToStringLayer transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
        joins the string list.
        """
        return StringListToStringLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            separator=self.getSeparator(),
            axis=-1,
            keepdims=True,
        )
