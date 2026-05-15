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
from pyspark.sql.types import IntegerType, StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringIndexLayer
from kamae.params.shared_specs import STRING_INDEX_PARAMS
from kamae.spark.params import MultiInputMultiOutputParams
from kamae.spark.utils import (
    indexer_udf,
    single_input_single_output_scalar_udf_transform,
)

from .base import BaseTransformer


class SharedStringIndexTransformer(
    BaseTransformer,
    MultiInputMultiOutputParams,
):
    """
    SharedStringIndexTransformer Spark Transformer for use in Spark pipelines.
    This transformer is used to index/transform feature columns using the string labels
    collected by the SharedStringIndexEstimator.

    NOTE: If your data contains null characters:
    https://en.wikipedia.org/wiki/Null_character
    This transformer could fail since the hashing algorithm uses cannot accept null
    characters. If you have null characters in your data, you should remove them.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = None  # Custom get_keras_layer due to multi-output
    _params = {**STRING_INDEX_PARAMS}

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset using the string index labels assigning a unique
        integer to each string label.

        :param dataset: Pyspark dataframe to transform.

        :returns: Pyspark dataframe with the input columns indexed,
         named as the output columns.
        """
        labels = self.getLabelsArray()
        num_oov_indices = self.getNumOOVIndices()
        mask_token = self.getMaskToken()

        # Assumption made that all the input columns have the same datatype/nesting.
        input_datatypes = [
            self.get_column_datatype(dataset=dataset, column_name=c)
            for c in self.getInputCols()
        ]

        output_cols = []
        for i, column in enumerate(self.getInputCols()):
            output_col = single_input_single_output_scalar_udf_transform(
                input_col=F.col(column),
                input_col_datatype=input_datatypes[i],
                func=lambda x: indexer_udf(
                    label=x,
                    labels=labels,
                    num_oov_indices=num_oov_indices,
                    mask_token=mask_token,
                ),
                udf_return_element_datatype=IntegerType(),
            )
            output_cols.append(output_col.alias(self.getOutputCols()[i]))

        select_cols = [F.col(c) for c in dataset.columns] + output_cols

        return dataset.select(*select_cols)

    def get_keras_layer(self) -> List[tf.keras.layers.Layer]:
        """
        Gets the list of Keras layers for the shared string indexer transformer.
        We need to use a list as each layer could operate on differing input shapes.

        :returns: List of Keras layer with name equal to the layerName
        parameter and the input column name, that performs the indexing.
        """
        return [
            StringIndexLayer(
                name=f"{self.getLayerName()}_{input_name}",
                input_dtype=self.getInputKerasDtype(),
                output_dtype=self.getOutputKerasDtype(),
                vocabulary=self.getLabelsArray(),
                mask_token=self.getMaskToken(),
                num_oov_indices=self.getNumOOVIndices(),
            )
            for input_name in self.getInputCols()
        ]
