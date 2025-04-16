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
import re
from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import SubStringDelimAtIndexLayer

from .base import BaseTransformer


class SubStringDelimAtIndexParams(Params):
    """
    Mixin class containing delimiter & index parameter needed for sub string transforms.
    """

    delimiter = Param(
        Params._dummy(),
        "delimiter",
        "Value to use to split the string into substrings.",
        typeConverter=TypeConverters.toString,
    )

    index = Param(
        Params._dummy(),
        "index",
        "Once the string is split using delimiter, which index to return.",
        typeConverter=TypeConverters.toInt,
    )

    defaultValue = Param(
        Params._dummy(),
        "defaultValue",
        "If the index is out of bounds after string split, what value to return.",
        typeConverter=TypeConverters.toString,
    )

    def setDelimiter(self, value: str) -> "SubStringDelimAtIndexParams":
        """
        Sets the delimiter parameter.

        :param value: String value to split substring on.
        :returns: Instance of class mixed in.
        """
        return self._set(delimiter=value)

    def getDelimiter(self) -> str:
        """
        Gets the delimiter parameter.

        :returns: String value to split substring on.
        """
        return self.getOrDefault(self.delimiter)

    def setIndex(self, value: int) -> "SubStringDelimAtIndexParams":
        """
        Sets the delimiter parameter.

        :param value: Index of substring to return.
        :returns: Instance of class mixed in.
        """
        return self._set(index=value)

    def getIndex(self) -> int:
        """
        Gets the index parameter.

        :returns: Integer value of index of substring to return.
        """
        return self.getOrDefault(self.index)

    def setDefaultValue(self, value: str) -> "SubStringDelimAtIndexParams":
        """
        Sets the defaultValue parameter.

        :param value: String value use as default if index is out of bounds.
        :returns: Instance of class mixed in.
        """
        return self._set(defaultValue=value)

    def getDefaultValue(self) -> str:
        """
        Gets the defaultValue parameter.

        :returns: String value use as default if index is out of bounds.
        """
        return self.getOrDefault(self.defaultValue)


class SubStringDelimAtIndexTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    SubStringDelimAtIndexParams,
):
    """
    Sub string at delimiter Spark Transformer for use in Spark pipelines.
    This transformer splits a string at a delimiter and returns the substring
    at the specified index. If the delimiter is the empty string, the string
    is split by characters.
    If the index is negative, start counting from the end of the string.
    If the index is out of bounds, the default value is returned.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        delimiter: Optional[str] = None,
        index: Optional[int] = None,
        defaultValue: Optional[str] = None,
    ) -> None:
        """
        Initializes an SubStringDelimAtIndexTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param delimiter: Value to use to split the string into substrings.
        Default is "_".
        :param index: Once the string is split using delimiter, which index to return.
        If the index is negative, start counting from the end of the string.
        Default is 0.
        :param defaultValue: If the index is out of bounds after string split,
         what value to return. Default is empty string.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(delimiter="_")
        self._setDefault(defaultValue="")
        self._setDefault(index=0)
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
        which splits the input column at the delimiter and returns the substring
        at the specified index. If the index is out of bounds, the default value
        is returned.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        delimiter = re.escape(self.getDelimiter())
        index = self.getIndex()
        default_value = self.getDefaultValue()

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        # Since element_at is a 1-based index , we need to add 1 to the index if it
        # is non-negative.
        one_based_index = index + 1 if index >= 0 else index
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: F.coalesce(
                F.element_at(F.split(x, delimiter), one_based_index),
                F.lit(default_value),
            ),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for SubStringDelimAtIndexTransformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs sub string at delimiter.
        """
        return SubStringDelimAtIndexLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            delimiter=self.getDelimiter(),
            index=self.getIndex(),
            default_value=self.getDefaultValue(),
        )
