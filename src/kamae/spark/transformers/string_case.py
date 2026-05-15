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

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringCaseLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


def _validate_string_case_type(value: str) -> str:
    possible_order_options = [
        "upper",
        "lower",
    ]
    if value not in possible_order_options:
        raise ValueError(
            f"stringCaseType must be one of {', '.join(possible_order_options)}"
        )
    return value


class StringCaseTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    StringCaseLayer Spark Transformer for use in Spark pipelines.
    This transformer applies an upper, lower or capitalise operation
    on the input column.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = StringCaseLayer
    _params = {
        "stringCaseType": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="lower",
            doc="How to change the case of the string.",
            validator=_validate_string_case_type,
        ),
    }

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
