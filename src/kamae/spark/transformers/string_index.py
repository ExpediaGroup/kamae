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
import tensorflow as tf
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringIndexLayer
from kamae.params.shared_specs import STRING_INDEX_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import (
    indexer_udf,
    single_input_single_output_scalar_udf_transform,
)

from .base import BaseTransformer


class StringIndexTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    StringIndexTransformer Spark Transformer for use in Spark pipelines.
    This transformer is used to index/transform feature columns using the string labels
    collected by the StringIndexEstimator.

    NOTE: If your data contains null characters:
    https://en.wikipedia.org/wiki/Null_character
    This transformer could fail since the hashing algorithm uses cannot accept null
    characters. If you have null characters in your data, you should remove them.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = None
    _params = {**STRING_INDEX_PARAMS}

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset using the string index labels assigning a unique
        integer to each string label.

        :param dataset: Pyspark dataframe to transform.

        :returns: Pyspark dataframe with the input column indexed,
         named as the output column.
        """
        labels = self.getLabelsArray()
        num_oov_indices = self.getNumOOVIndices()
        mask_token = self.getMaskToken()

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_udf_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: indexer_udf(
                label=x,
                labels=labels,
                num_oov_indices=num_oov_indices,
                mask_token=mask_token,
            ),
            udf_return_element_datatype=IntegerType(),
        )

        return dataset.withColumn(
            self.getOutputCol(),
            output_col,
        )

    def get_keras_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the Keras layer for the string indexer transformer.

        :returns: Keras layer with name equal to the layerName parameter
        that performs the indexing.
        """
        # Spark param is labelsArray but Keras layer expects vocabulary
        return StringIndexLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            vocabulary=self.getLabelsArray(),
            mask_token=self.getMaskToken(),
            num_oov_indices=self.getNumOOVIndices(),
        )
