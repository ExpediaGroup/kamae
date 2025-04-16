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
from pyspark.sql.types import DataType, IntegerType, StringType

from kamae.spark.params import MultiInputMultiOutputParams, StringIndexParams
from kamae.spark.utils import (
    indexer_udf,
    single_input_single_output_scalar_udf_transform,
)
from kamae.tensorflow.layers import StringIndexLayer

from .base import BaseTransformer


class SharedStringIndexTransformer(
    BaseTransformer,
    StringIndexParams,
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

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCols: Optional[List[str]] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        labelsArray: Optional[List[str]] = None,
        stringOrderType: Optional[str] = None,
        maskToken: Optional[str] = None,
        numOOVIndices: int = 1,
    ) -> None:
        """
        Initializes the SharedStringIndexTransformer transformer.

        :param inputCols: Input column names.
        :param outputCols: Output column names.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column(s) to after
        transforming. Must be the same length as inputCols.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param: stringOrderType: How to order the string indices.
        Options are 'frequencyAsc', 'frequencyDesc', 'alphabeticalAsc',
        'alphabeticalDesc'.
        :param maskToken: Token to use for masking.
        If set, the token will be indexed as 0.
        :param numOOVIndices: Number of out of vocabulary indices to use.
        Default is 1.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(
            stringOrderType="frequencyDesc", numOOVIndices=1, maskToken=None
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [StringType()]

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

    def get_tf_layer(self) -> List[tf.keras.layers.Layer]:
        """
        Gets the list of tensorflow layers for the shared string indexer transformer.
        We need to use a list as each layer could operate on differing input shapes.

        :returns: List of Tensorflow keras layer with name equal to the layerName
        parameter and the input column name, that performs the indexing.
        """
        return [
            StringIndexLayer(
                name=f"{self.getLayerName()}_{input_name}",
                input_dtype=self.getInputTFDtype(),
                output_dtype=self.getOutputTFDtype(),
                vocabulary=self.getLabelsArray(),
                mask_token=self.getMaskToken(),
                num_oov_indices=self.getNumOOVIndices(),
            )
            for input_name in self.getInputCols()
        ]
