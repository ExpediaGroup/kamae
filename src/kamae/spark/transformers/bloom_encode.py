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
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import ArrayType, DataType, IntegerType, StringType

from kamae.spark.params import HashIndexParams, SingleInputSingleOutputParams
from kamae.spark.utils import (
    hash_udf,
    single_input_single_output_array_udf_transform,
    single_input_single_output_scalar_transform,
)
from kamae.tensorflow.layers import BloomEncodeLayer

from .base import BaseTransformer


class BloomEncodeParams(HashIndexParams):
    """
    Mixin class containing parameters needed for bloom encoding.
    """

    numHashFns = Param(
        Params._dummy(),
        "numHashFns",
        "Number of hash functions to use for bloom encoding",
        typeConverter=TypeConverters.toInt,
    )

    featureCardinality = Param(
        Params._dummy(),
        "featureCardinality",
        "Dimension/cardinality of the feature",
        typeConverter=TypeConverters.toInt,
    )

    useHeuristicNumBins = Param(
        Params._dummy(),
        "useHeuristicNumBins",
        "Whether to use te heuristic from the paper to determine the number of bins",
        typeConverter=TypeConverters.toBoolean,
    )

    def setNumHashFns(self, value: int) -> "BloomEncodeParams":
        """
        Sets the `numHashFns` parameter.
        """
        if value < 2:
            raise ValueError("numHashFns must be at least 2.")
        return self._set(numHashFns=value)

    def getNumHashFns(self) -> int:
        """
        Gets the value of `numHashFns` parameter.
        """
        return self.getOrDefault(self.numHashFns)

    def setFeatureCardinality(self, value: int) -> "BloomEncodeParams":
        """
        Sets the `featureCardinality` parameter.
        """
        if value < 1:
            raise ValueError("featureCardinality must be greater than 0")
        return self._set(featureCardinality=value)

    def getFeatureCardinality(self) -> int:
        """
        Gets the value of `featureCardinality` parameter.
        """
        return self.getOrDefault(self.featureCardinality)

    def setUseHeuristicNumBins(self, value: bool) -> "BloomEncodeParams":
        """
        Sets the `useHeuristicNumBins` parameter.
        """
        return self._set(useHeuristicNumBins=value)

    def getUseHeuristicNumBins(self) -> bool:
        """
        Gets the value of `useHeuristicNumBins` parameter.
        """
        return self.getOrDefault(self.useHeuristicNumBins)

    def getNumBins(self) -> int:
        """
        Gets the number of bins to use for hash indexing.
        """
        if self.getUseHeuristicNumBins() and self.getFeatureCardinality() is not None:
            return max(round(self.getFeatureCardinality() * 0.2), 2)
        elif self.getUseHeuristicNumBins():
            raise ValueError(
                """If useHeuristicNumBins is set to True, then the featureCardinality
                parameter must be set."""
            )
        return self.getOrDefault(self.numBins)


class BloomEncodeTransformer(
    BaseTransformer,
    BloomEncodeParams,
    SingleInputSingleOutputParams,
):
    """
    Bloom encoder Spark Transformer for use in Spark pipelines.
    This transformer performs bloom encoding on the input column resulting in an
    array of integers of size equal to numHashFns.
    See paper for more details: https://arxiv.org/pdf/1706.03993.pdf
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        numHashFns: int = 3,
        numBins: Optional[int] = None,
        maskValue: Optional[str] = None,
        featureCardinality: Optional[int] = None,
        useHeuristicNumBins: bool = False,
    ) -> None:
        """
        Instantiates a BloomEncode transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param numHashFns: Number of hash functions to use. Defaults to 3.
        The paper suggests a range of 2-4 hash functions for optimal performance.
        :param numBins: Number of hash bins. Note that this includes the `maskValue`
        bin, so the effective number of bins is `(num_bins - 1)` if `maskValue`
        is set. If `useHeuristicNumBins` is set to True, then this parameter is
        ignored and the number of bins is automatically set. See the description of this
        parameter below for how the heuristic is built.
        :param maskValue: A value that represents masked inputs, which are mapped to
        index 0. Defaults to None, meaning no mask term will be added and the
        hashing will start at index 0.
        :param featureCardinality: The cardinality of the input tensor. Needed to
        use the num of bins heuristic. Defaults to None, meaning the number of bins will
        not use the heuristic and will need to be set manually.
        :param useHeuristicNumBins: If set to True, the number of bins is automatically
        set by fixing the ratio of the feature cardinality to the number of bins
        to be b/f = 0.2. This ratio was found to be optimal in the paper for a wide
        variety of usecases. Therefore, numBins = featureCardinality * 0.2. This reduces
        the cardinality of the input tensor by 5x.
        Requires the `featureCardinality` parameter to be set. Defaults to False.
        :returns: None - class instantiated.
        """
        super().__init__()
        kwargs = self._input_kwargs
        self._setDefault(
            numHashFns=3,
            numBins=None,
            maskValue=None,
            featureCardinality=None,
            useHeuristicNumBins=False,
        )
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [StringType()]

    def _create_salted_input(self, column_data_type: DataType) -> Column:
        """
        Builds the salted inputs according to how many hash functions are used.
        Specifically concatenates the input column with the string "0" using a
        separator of the hash function index.

        :returns: Salted input spark column
        """
        return single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=column_data_type,
            func=lambda x: F.array(
                [F.concat(F.lit(x), F.lit(i)) for i in range(self.getNumHashFns())]
            ),
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column named outputCol with the
        bloom encoded input column.

        :param dataset: Pyspark DataFrame to transform.
        :returns: Transformed pyspark dataFrame.
        """
        num_bins = self.getNumBins()
        if num_bins is None:
            # num_bins can be None only if useHeuristicNumBins is False
            raise ValueError("numBins must be set if useHeuristicNumBins is False.")
        mask_value = self.getMaskValue()

        input_data_type = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        salted_input = self._create_salted_input(input_data_type)

        # The salting process nests the input column into another array. Thus, the array
        # nesting level is increased by 1.
        def bloom_encode(x):
            return [
                hash_udf(
                    label=y,
                    num_bins=num_bins,
                    # If the user set a mask value, then this won't match the inputs
                    # as they have been salted. So we need to salt the mask value as
                    # well.
                    mask_value=f"{mask_value}{i}" if mask_value is not None else None,
                )
                for i, y in enumerate(x)
            ]

        output_col = single_input_single_output_array_udf_transform(
            input_col=salted_input,
            # Input datatype is from salted input, so it has an additional; nesting.
            input_col_datatype=ArrayType(input_data_type),
            func=bloom_encode,
            udf_return_element_datatype=IntegerType(),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that performs the bloom encoding.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that performs the bloom encoding operation.
        """
        return BloomEncodeLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            num_hash_fns=self.getNumHashFns(),
            num_bins=self.getNumBins(),
            mask_value=self.getMaskValue(),
            feature_cardinality=self.getFeatureCardinality(),
            use_heuristic_num_bins=self.getUseHeuristicNumBins(),
        )
