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

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringToStringListLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


def _validate_list_length(value: int) -> int:
    if value < 1:
        raise ValueError("listLength must be greater than 0.")
    return value


class StringToStringListTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    StringToStringListLayer Spark Transformer for use in Spark pipelines.
    This transformer takes a column of string lists and joins them into a single string.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = StringToStringListLayer
    _params = {
        "separator": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default=",",
            doc="Separator to use when joining the string list.",
        ),
        "listLength": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=1,
            doc="Length of the output list.",
            validator=_validate_list_length,
        ),
        "defaultValue": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="",
            doc="Default value to use when the input is empty.",
        ),
    }

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
