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
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringToStringListLayer

from .base import BaseTransformer


class StringToStringListParams(Params):
    """
    Mixin class containing separator parameter needed for
    string to string list transforms.
    """

    separator = Param(
        Params._dummy(),
        "separator",
        "Separator to use when joining the string list.",
        typeConverter=TypeConverters.toString,
    )

    listLength = Param(
        Params._dummy(),
        "listLength",
        "Length of the output list.",
        typeConverter=TypeConverters.toInt,
    )

    defaultValue = Param(
        Params._dummy(),
        "defaultValue",
        "Default value to use when the input is empty.",
        typeConverter=TypeConverters.toString,
    )

    def getSeparator(self) -> str:
        """
        Gets the separator to use when joining the string list.

        :returns: Separator to use when joining the string list.
        """
        return self.getOrDefault(self.separator)

    def setSeparator(self, value: str) -> "StringToStringListParams":
        """
        Sets the separator to use when joining the string list.

        :param value: Separator to use when joining the string list.
        :returns: Instance of class mixed in.
        """
        return self._set(separator=value)

    def getListLength(self) -> int:
        """
        Gets the length of the output list.

        :returns: Length of the output list.
        """
        return self.getOrDefault(self.listLength)

    def setListLength(self, value: int) -> "StringToStringListParams":
        """
        Sets the length of the list.

        :param value: Length of the output list.
        :returns: Instance of class mixed in.
        """
        if value < 1:
            raise ValueError("listLength must be greater than 0.")
        return self._set(listLength=value)

    def getDefaultValue(self) -> str:
        """
        Gets the default value to use when the input is empty.

        :returns: Default value to use when the input is empty.
        """
        return self.getOrDefault(self.defaultValue)

    def setDefaultValue(self, value: str) -> "StringToStringListParams":
        """
        Sets the default value to use when the input is empty.

        :param value: Default value to use when the input is empty.
        :returns: Instance of class mixed in.
        """
        return self._set(defaultValue=value)


class StringToStringListTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    StringToStringListParams,
):
    """
    StringToStringListLayer Spark Transformer for use in Spark pipelines.
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
        separator: str = ",",
        listLength: int = 1,
        defaultValue: str = "",
    ) -> None:
        """
        Initializes an StringToStringListTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param separator: Separator to use when joining the string list.
        Defaults to ",".
        :param listLength: Length of the output list. Default is 1.
        :param defaultValue: Default value to use when the input is empty.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(separator=",")
        self._setDefault(listLength=1)
        self._setDefault(defaultValue="")
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
        which is an array of strings created by splitting the input column by the
        separator.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        separator = re.escape(self.getSeparator())

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def string_to_string_list(x: Column, separator: str) -> Column:
            split_col = F.split(x, pattern=separator)
            # Replace empty strings with default value
            split_array_col = F.transform(
                split_col,
                lambda x: F.when(x == F.lit(""), self.getDefaultValue()).otherwise(x),
            )
            # Pad/truncate array to size
            padded_split_array_col = F.concat(
                F.slice(split_array_col, 1, self.getListLength()),
                F.array_repeat(
                    F.lit(self.getDefaultValue()),
                    self.getListLength() - F.size(split_array_col),
                ),
            )
            return padded_split_array_col

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: string_to_string_list(x=x, separator=separator),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the StringToStringListLayer transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
        splits the string into a list of strings.
        """
        return StringToStringListLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            separator=self.getSeparator(),
            default_value=self.getDefaultValue(),
            list_length=self.getListLength(),
        )
