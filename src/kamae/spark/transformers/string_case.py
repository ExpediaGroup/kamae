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
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringCaseLayer

from .base import BaseTransformer


class StringCaseParams(Params):
    """
    Mixin class containing stringCaseType parameter needed for string casing transforms.
    """

    stringCaseType = Param(
        Params._dummy(),
        "stringCaseType",
        "How to change the case of the string.",
        typeConverter=TypeConverters.toString,
    )

    def setStringCaseType(self, value: str) -> "StringCaseParams":
        """
        Sets the stringCaseType parameter to the given value.
        Must be one of:
        - 'upper'
        - 'lower'

        :param value: String to set the stringCaseType parameter to.
        :returns: Instance of class mixed in.
        """
        possible_order_options = [
            "upper",
            "lower",
        ]
        if value not in possible_order_options:
            raise ValueError(
                f"stringCaseType must be one of {', '.join(possible_order_options)}"
            )
        return self._set(stringCaseType=value)

    def getStringCaseType(self) -> str:
        """
        Gets the stringCaseType parameter.

        :returns: String value of how to change the case of the string.
        """
        return self.getOrDefault(self.stringCaseType)


class StringCaseTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    StringCaseParams,
):
    """
    StringCaseLayer Spark Transformer for use in Spark pipelines.
    This transformer applies an upper, lower or capitalise operation
    on the input column.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        stringCaseType: Optional[str] = None,
    ) -> None:
        """
        Initializes an StringCaseTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param stringCaseType: How to change the case of the string. Must be one of:
        - 'upper'
        - 'lower'
        Default is 'lower'.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(stringCaseType="lower")
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
        string_case_type = self.getStringCaseType()

        def string_case(x: Column, case_type: str) -> Column:
            if case_type == "upper":
                return F.upper(x)
            elif case_type == "lower":
                return F.lower(x)
            else:
                raise ValueError(
                    f"""stringCaseType must be one of 'upper' or 'lower'.
                    Got {case_type}"""
                )

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: string_case(x, string_case_type),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the StringCaseLayer transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs the string casing operation.
        """
        return StringCaseLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            string_case_type=self.getStringCaseType(),
        )
