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

from kamae.spark.params import HashIndexParams, SingleInputSingleOutputParams
from kamae.spark.utils import hash_udf, single_input_single_output_scalar_udf_transform
from kamae.tensorflow.layers import HashIndexLayer

from .base import BaseTransformer


class HashIndexTransformer(
    BaseTransformer,
    HashIndexParams,
    SingleInputSingleOutputParams,
):
    """
    Hash indexer Spark Transformer for use in Spark pipelines.
    This transformer hashes the input column and then bins it into
    the specified number of bins using modulo arithmetic.

    NOTE: If your data contains null characters:
    https://en.wikipedia.org/wiki/Null_character
    This transformer could fail since the hashing algorithm uses cannot accept null
    characters. If you have null characters in your data, you should remove them.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        numBins: Optional[int] = None,
        maskValue: Optional[str] = None,
    ) -> None:
        """
        Instantiates a HashIndexTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param numBins: Number of bins to use for hash indexing.
        :param maskValue: Mask value to use for hash indexing.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(maskValue=None)
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
        Transforms the input dataset. Creates a new column named outputCol with the
        hash indexed input column.

        :param dataset: Pyspark DataFrame to transform.
        :returns: Transformed pyspark dataFrame.
        """
        num_bins = self.getNumBins()
        mask_value = self.getMaskValue()

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_udf_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: hash_udf(label=x, num_bins=num_bins, mask_value=mask_value),
            udf_return_element_datatype=IntegerType(),
        )

        return dataset.withColumn(
            self.getOutputCol(),
            output_col,
        )

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that performs the hash indexing.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that performs the hash indexing operation.
        """
        return HashIndexLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            num_bins=self.getNumBins(),
            mask_value=self.getMaskValue(),
        )
