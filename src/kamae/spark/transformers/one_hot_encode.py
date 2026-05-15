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
from typing import List

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DataType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
)

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import OneHotEncodeLayer
from kamae.params.shared_specs import DROP_UNSEEN_PARAMS, STRING_INDEX_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import (
    one_hot_encoding_udf,
    single_input_single_output_scalar_udf_transform,
)

from .base import BaseTransformer


class OneHotEncodeTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    OneHotEncodeTransformer Spark Transformer for use in Spark pipelines.
    This transformer is used to one-hot feature columns using the string labels
    collected by the OneHotEncodeEstimator.

    NOTE: If your data contains null characters:
    https://en.wikipedia.org/wiki/Null_character
    This transformer could fail since the hashing algorithm uses cannot accept null
    characters. If you have null characters in your data, you should remove them.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [ShortType(), IntegerType(), LongType(), StringType()]
    _keras_layer_class = None
    _params = {**STRING_INDEX_PARAMS, **DROP_UNSEEN_PARAMS}

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset using the string index labels assigning array of
        one-hot encoded values to the output column.

        :param dataset: Pyspark dataframe to transform.

        :returns: Pyspark dataframe with the input column one-hot encoded,
         named as the output column.
        """
        labels = self.getLabelsArray()
        ohe_num_oov_indices = self.getNumOOVIndices()
        mask_token = self.getMaskToken()
        drop_unseen = self.getDropUnseen()

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_udf_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: one_hot_encoding_udf(
                label=x,
                labels=labels,
                num_oov_indices=ohe_num_oov_indices,
                mask_token=mask_token,
                drop_unseen=drop_unseen,
            ),
            udf_return_element_datatype=ArrayType(FloatType()),
        )

        return dataset.withColumn(
            self.getOutputCol(),
            output_col,
        )

    def get_keras_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the Keras layer for the one-hot encoder transformer.

        :returns: Keras layer with name equal to the layerName parameter
        that performs the one-hot encoding.
        """
        # Spark param is labelsArray but Keras layer expects vocabulary
        return OneHotEncodeLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            vocabulary=self.getLabelsArray(),
            num_oov_indices=self.getNumOOVIndices(),
            mask_token=self.getMaskToken(),
            drop_unseen=self.getDropUnseen(),
        )
