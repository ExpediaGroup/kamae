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

from kamae.keras.tensorflow.layers import StringArrayConstantLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform

from .base import BaseTransformer


class StringArrayConstantTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    String Array Constant Spark Transformer for use in Spark pipelines.
    This transformer populates a column with a constant string array.
    """

    _compatible_dtypes = None
    _keras_layer_class = StringArrayConstantLayer
    _params = {
        "constantStringArray": ParamSpec(
            spark_typeconverter=TypeConverters.toListString,
            default=None,
            doc="List of strings to use as a constant string array.",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        populates it with the constant string array.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: F.lit(self.getConstantStringArray()).cast("array<string>"),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
