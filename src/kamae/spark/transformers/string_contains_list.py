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
from functools import reduce
from typing import List

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark.ml.param import TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringContainsListLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform


class StringContainsListTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    String contains list Spark Transformer for use in Spark pipelines.
    This transformer performs a string contains operation on the input column over all
    constants in the passed constantStringArray.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    # Overrides codegen: Spark param constantStringArray maps to Keras string_constant_list (name mismatch)
    _keras_layer_class = None
    _params = {
        "constantStringArray": ParamSpec(
            spark_typeconverter=TypeConverters.toListString,
            default=None,
            doc="String constant array to use in string contains list operation.",
        ),
        "negation": ParamSpec(
            spark_typeconverter=TypeConverters.toBoolean,
            default=False,
            doc="Whether to negate the string contains list operation.",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which contains the result of the string contains operation.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        if not self.isDefined("constantStringArray"):
            raise ValueError("constantStringArray must be defined.")

        def string_contains_list(
            x: Column, string_list: List[str], negation: bool
        ) -> Column:
            contains_cols = [
                x.contains(string_constant) for string_constant in string_list
            ]
            col_expr = reduce(lambda y, z: y | z, contains_cols)
            return col_expr if not negation else ~col_expr

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: string_contains_list(
                x=x,
                string_list=self.getConstantStringArray(),
                negation=self.getNegation(),
            ),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_keras_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the Keras layer for the StringContainsLayer transformer.

        :returns: Keras layer with name equal to the layerName parameter that
         performs a string contains operation.
        """

        if not self.isDefined("constantStringArray"):
            raise ValueError("constantStringArray must be defined.")

        return StringContainsListLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            negation=self.getNegation(),
            string_constant_list=self.getConstantStringArray(),
        )
