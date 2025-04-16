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

from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
    StringRegexParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringReplaceLayer

from .base import BaseTransformer


class StringReplaceParams(StringRegexParams):
    """
    Mixin class containing StringMatchConstant, StringReplaceConstant needed for string
    replace layers.
    """

    stringMatchConstant = Param(
        Params._dummy(),
        "stringMatchConstant",
        "String match constant to use in string replace",
        typeConverter=TypeConverters.toString,
    )

    stringReplaceConstant = Param(
        Params._dummy(),
        "stringReplaceConstant",
        "String replace constant to use in string replace",
        typeConverter=TypeConverters.toString,
    )

    def setStringMatchConstant(self, value: str) -> "StringReplaceParams":
        """
        Sets the stringConstant parameter.

        :param value: String constant value to use in different string transformers.
        :returns: Instance of class mixed in.
        """
        return self._set(stringMatchConstant=value)

    def getStringMatchConstant(self) -> str:
        """
        Gets the stringConstant parameter.

        :returns: String replacement constant value to use in string replace layer.
        """
        return self.getOrDefault(self.stringMatchConstant)

    def setStringReplaceConstant(self, value: str) -> "StringReplaceParams":
        """
        Sets the stringReplaceConstant parameter.
        """
        return self._set(stringReplaceConstant=value)

    def getStringReplaceConstant(self) -> str:
        """
        Gets the stringReplaceConstant parameter.

        :returns: String replacement constant value to use in string replace layer.
        """
        return self.getOrDefault(self.stringReplaceConstant)


class StringReplaceTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
    StringReplaceParams,
):
    """
    String replace Spark Transformer for use in Spark Pipelines.
    This transformer performs a string replace operation on the input column.

    The transformer takes up to 3 input columns.
    The first input column is always required and is the column we operate on.
    A match constant/column is provided to match against and the replace constant/column
    is provided to replace the matched substrings with.

    KNOWN ISSUE: when replacing with a string that contains a backslash,
    the backslash must be double escaped (\\\\) in order to be added properly.
    This is consistent in both spark and tensorflow components.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        stringMatchConstant: Optional[str] = None,
        stringReplaceConstant: Optional[str] = None,
        regex: bool = False,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initializes an StringReplaceTransformer transformer.

        :param inputCol: Input column name.
        :param inputCols: List of up to 3 input column names.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param stringMatchConstant: String constant to match for.
        operation.
        :param stringReplaceConstant: String constant to replace with.
        :param regex: Whether to allow regex-matching in the string matching.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(
            regex=False, stringMatchConstant=None, stringReplaceConstant=None
        )
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

    def setInputCols(self, value: List[str]) -> "StringReplaceTransformer":
        """
        Overrides setting the input columns for the transformer.
        Throws an error if we have more than 3 input columns.

        :param value: List of input columns.
        :returns: Class instance.
        """
        if len(value) > 3:
            raise ValueError(
                """When setting inputCols for StringReplaceTransformer,
                there must be 1-3 columns."""
            )

        return self._set(inputCols=value)

    def _construct_input_cols(self) -> List[Column]:
        """
        Constructs the input columns for the transformer.

        :returns: List of pyspark columns.
        """
        if self.isDefined("inputCol"):
            input_cols = [F.col(self.getInputCol())]
        elif self.isDefined("inputCols"):
            input_cols = [F.col(c) for c in self.getInputCols()]
        else:
            raise ValueError("Must specify either inputCol or inputCols.")

        if self.getStringReplaceConstant() is not None:
            input_cols.insert(
                len(input_cols),
                F.lit(self.getStringReplaceConstant()).alias(
                    self.uid + "_stringReplaceConstant"
                ),
            )
        if self.getStringMatchConstant() is not None:
            input_cols.insert(
                1,
                F.lit(self.getStringMatchConstant()).alias(
                    self.uid + "_stringMatchConstant"
                ),
            )

        if len(input_cols) > 3:
            raise ValueError(
                """When setting inputCols for StringReplaceTransformer,
            there must be 1-3 columns. In the case 3 columns are specified, string
            constants should not be specified."""
            )
        elif len(input_cols) < 3:
            raise ValueError(
                """Less than 3 input columns were provided,
            but no string constants were passed as arguments."""
            )

        return input_cols

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which contains the result of the string replace operation.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self._construct_input_cols()
        # input_cols can contain either actual columns or lit(constants). In order to
        # determine the datatype of the input columns, we select them from the dataset
        # first.
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]
        regex = self.getRegex()

        def string_replace(
            x: Column, input_col_names: List[str], regex: bool
        ) -> Column:
            col_expr = (
                F.regexp_replace(
                    x[input_col_names[0]],
                    F.regexp_replace(
                        x[input_col_names[1]], "[^A-Za-z0-9]", "\\\\" + "$0"
                    ),
                    x[input_col_names[2]],
                )
                if not regex
                else F.regexp_replace(
                    x[input_col_names[0]],
                    F.when(x[input_col_names[1]] == F.lit(""), F.lit("^$")).otherwise(
                        x[input_col_names[1]]
                    ),
                    x[input_col_names[2]],
                )
            )
            return col_expr

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            func=lambda x: string_replace(x, input_col_names, regex),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the StringReplaceLayer transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs a string replace operation.
        """
        return StringReplaceLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            regex=self.getRegex(),
            string_match_constant=self.getStringMatchConstant(),
            string_replace_constant=self.getStringReplaceConstant(),
        )
