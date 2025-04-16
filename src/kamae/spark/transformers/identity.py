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
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.tensorflow.layers import IdentityLayer

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

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initializes an IdentityTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return None

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the same as the column with name `inputCol`.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        return dataset.withColumn(self.getOutputCol(), F.col(self.getInputCol()))

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the identity transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs an IdentityLayer operation.
        """
        # Tensorflow <= 2.11 does not contain tf.keras.layers.IdentityLayer
        # so we use a lambda layer instead.
        # When we have a subclassed identity layer, we can use that.
        return IdentityLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
        )
