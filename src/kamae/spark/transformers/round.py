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

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, FloatType

from kamae.keras.core.layers import RoundLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


def _validate_round_type(value: str) -> str:
    """Validator for roundType parameter."""
    if value not in ["floor", "ceil", "round"]:
        raise ValueError("roundType must be one of 'floor', 'ceil' or 'round'")
    return value


class RoundTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Round Spark Transformer for use in Spark pipelines.
    This transformer rounds the input column to the nearest integer using the
    specified rounding type.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = RoundLayer
    _params = {
        "roundType": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="round",
            doc="Round type to use in round transform, one of 'floor', 'ceil' or 'round'.",
            validator=_validate_round_type,
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        func_dict = {
            "floor": F.floor,
            "ceil": F.ceil,
            "round": F.round,
        }

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: func_dict[self.getRoundType()](x),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
