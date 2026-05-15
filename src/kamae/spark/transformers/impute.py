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
from typing import List, Optional, Union

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType

from kamae.keras.core.layers import ImputeLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


def _validate_impute_value(value: Union[float, int, str]) -> Union[float, int, str]:
    if value is None:
        raise ValueError("Impute value cannot be None")
    return value


class ImputeTransformer(BaseTransformer, SingleInputSingleOutputParams):
    """
    Imputation transformer for use in Spark pipelines.
    This is used to impute the mean or median value when
    value is null or equalling a mask
    """

    jit_compatible = True

    _compatible_dtypes = None
    _keras_layer_class = ImputeLayer
    _params = {
        "imputeValue": ParamSpec(
            spark_typeconverter=TypeConverters.identity,
            default=None,
            doc="Value to be imputed.",
            validator=_validate_impute_value,
        ),
        "maskValue": ParamSpec(
            spark_typeconverter=TypeConverters.identity,
            default=None,
            doc="Value which is to be replaced with the imputation value. "
            "Value is ignored when computing the imputation value.",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input data by imputing the specified
        statistic of the input column provided.
        Nulls and any values equalling the mask are imputed over.

        :param dataset: Pyspark dataframe to transform.
        :returns: Pyspark dataframe with the input columns
        imputed with the specified statistic, and names set to
        output column names provided
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: F.when(
                (x == F.lit(self.getMaskValue())) | (x.isNull()),
                F.lit(self.getImputeValue()),
            ).otherwise(x),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
