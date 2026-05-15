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

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringConcatenateLayer
from kamae.params import ParamSpec
from kamae.spark.params import MultiInputSingleOutputParams
from kamae.spark.utils import multi_input_single_output_scalar_transform

from .base import BaseTransformer


class StringConcatenateTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
):
    """
    String Concatenate Spark Transformer for use in Spark pipelines.
    This transformer takes in multiple columns and concatenates them together into a
    single column using a separator. Input columns must be of type string.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = StringConcatenateLayer
    _params = {
        "separator": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="_",
            doc="Value to use as a separator when joining the strings.",
        ),
    }

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
