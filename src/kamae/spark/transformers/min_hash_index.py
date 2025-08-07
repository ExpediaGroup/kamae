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
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, IntegerType, StringType

from kamae.spark.params import MaskStringValueParams, SingleInputSingleOutputParams
from kamae.spark.utils import (
    min_hash_udf,
    single_input_single_output_array_udf_transform,
)
from kamae.tensorflow.layers import MinHashIndexLayer

from .base import BaseTransformer


class MinHashIndexParams(MaskStringValueParams):
    """
    Mixin class containing bin parameter needed for the MinHashIndexTransformer.
    """

    numPermutations = Param(
        Params._dummy(),
        "numPermutations",
        """Number of permutations to perform the min hashing.
        Will return an array with length equal to this.""",
        typeConverter=TypeConverters.toInt,
    )

    def setNumPermutations(self, value: int) -> "MinHashIndexParams":
        """
        Sets the numPermutations parameter.

        :param value: Integer value for the number of bins to use for hash indexing.
        :returns: Instance of class mixed in.
        """
        if value <= 0:
            raise ValueError("Number of permutations must be greater than 0.")
        return self._set(numPermutations=value)

    def getNumPermutations(self) -> int:
        """
        Gets the numPermutations parameter.

        :returns: Integer value for the number of bins to use for hash indexing.
        """
        return self.getOrDefault(self.numPermutations)


class MinHashIndexTransformer(
    BaseTransformer,
    MinHashIndexParams,
    SingleInputSingleOutputParams,
):
    """
    MinHash indexer Spark Transformer for use in Spark pipelines.
    This transformer hashes the input string set using the MinHash algorithm:
    https://en.wikipedia.org/wiki/MinHash

    MinHash approximates the Jaccard similarity between sets by hashing the elements of
    the sets and returning a fixed-length signature. This length is determined by the
    numPermutations parameter, which defaults to 128. The output is an array of integer
    bits.

    Setting the maskValue parameter allows you to ignore a specific value in the
    input column when computing the min hash. This is useful if you have padded arrays
    as then a padded array with the same unique elements as another non-padded array
    will be considered equal.

    NOTE: If your data contains null characters:
    https://en.wikipedia.org/wiki/Null_character
    This transformer could fail since the hashing algorithm used cannot accept null
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
        numPermutations: int = 128,
        maskValue: Optional[str] = None,
    ) -> None:
        """
        Instantiates a MinHashIndexTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param numPermutations: Number of permutations of your output min hash.
        Defaults to 128. This is the length of the output array.
        :param maskValue: Mask value to use when indexing the input column.
        If set, the mask value will be ignored when computing the min hash.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(numPermutations=128, maskValue=None)
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
        min hash indexed input column.

        :param dataset: Pyspark DataFrame to transform.
        :returns: Transformed pyspark dataFrame.
        """
        if not self.isDefined("numPermutations"):
            raise ValueError("numPermutations parameter must be set.")
        num_permutations = self.getNumPermutations()
        mask_value = self.getMaskValue()

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        if not isinstance(input_datatype, ArrayType):
            raise ValueError(
                f"""Input column {self.getInputCol()} must be of type ArrayType,
                but got {input_datatype}."""
            )
        output_col = single_input_single_output_array_udf_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: min_hash_udf(
                labels=x, num_permutations=num_permutations, mask_value=mask_value
            ),
            udf_return_element_datatype=IntegerType(),
        )
        return dataset.withColumn(
            self.getOutputCol(),
            output_col,
        )

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that performs the min hash indexing.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that performs the hash indexing operation.
        """
        return MinHashIndexLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            num_permutations=self.getNumPermutations(),
            mask_value=self.getMaskValue(),
        )
