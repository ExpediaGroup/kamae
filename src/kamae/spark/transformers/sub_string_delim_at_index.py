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

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import SubStringDelimAtIndexLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


class SubStringDelimAtIndexTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Sub string at delimiter Spark Transformer for use in Spark pipelines.
    This transformer splits a string at a delimiter and returns the substring
    at the specified index. If the delimiter is the empty string, the string
    is split by characters.
    If the index is negative, start counting from the end of the string.
    If the index is out of bounds, the default value is returned.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = SubStringDelimAtIndexLayer
    _params = {
        "delimiter": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="_",
            doc="Value to use to split the string into substrings.",
        ),
        "index": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=0,
            doc="Once the string is split using delimiter, which index to return. "
            "If the index is negative, start counting from the end of the string.",
        ),
        "defaultValue": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="",
            doc="If the index is out of bounds after string split, what value to return.",
        ),
    }

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
