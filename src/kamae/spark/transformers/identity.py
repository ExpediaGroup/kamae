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
from pyspark.sql import DataFrame

from kamae.keras.core.layers import IdentityLayer
from kamae.spark.params import SingleInputSingleOutputParams

from .base import BaseTransformer


class IdentityTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    IdentityLayer Spark Transformer for use in Spark pipelines.
    This transformer simply passes the input to the output unchanged.
    Used for cases where you want to keep the input the same.
    """

    jit_compatible = True

    _compatible_dtypes = None
    _keras_layer_class = IdentityLayer

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the same as the column with name `inputCol`.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        return dataset.withColumn(self.getOutputCol(), F.col(self.getInputCol()))
