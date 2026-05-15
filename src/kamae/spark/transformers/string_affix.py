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

from kamae.keras.tensorflow.layers import StringAffixLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


class StringAffixTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    String Affix Spark Transformer for use in Spark pipelines.
    This transformer takes in a column and pre- and su- fixes it.
    Input columns must be of type string.
    """

    _compatible_dtypes = [StringType()]
    _keras_layer_class = StringAffixLayer
    _params = {
        "prefix": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default=None,
            doc="Value to use as a prefix when joining the strings.",
        ),
        "suffix": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default=None,
            doc="Value to use as a suffix when joining the strings.",
        ),
    }

    def _validate_params(self) -> None:
        """
        Validates the parameters passed to the transformer.
        """
        prefix = self.getPrefix()
        suffix = self.getSuffix()
        if (prefix is None or prefix == "") and (suffix is None or suffix == ""):
            raise ValueError(
                "Either prefix or suffix must be set. Otherwise nothing to affix."
            )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        where the value is origin column combined with prefix and or suffix.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        self._validate_params()

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def add_prefix_suffix(
            column: Column, prefix: Optional[str] = None, suffix: Optional[str] = None
        ) -> Column:
            if prefix is not None and prefix != "":
                column = F.concat(F.lit(prefix), column)
            if suffix is not None and suffix != "":
                column = F.concat(column, F.lit(suffix))
            return column

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: add_prefix_suffix(x, self.getPrefix(), self.getSuffix()),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)
