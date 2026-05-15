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
from pyspark.ml.param import TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import ArrayType, DataType, IntegerType, StringType

from kamae.keras.tensorflow.layers import BloomEncodeLayer
from kamae.params import ParamSpec
from kamae.params.shared_specs import MASK_STRING_VALUE_PARAMS
from kamae.spark.params import _UNSET, SingleInputSingleOutputParams
from kamae.spark.utils import (
    hash_udf,
    single_input_single_output_array_udf_transform,
    single_input_single_output_scalar_transform,
)

from .base import BaseTransformer


def _validate_num_hash_fns(value: int) -> int:
    """Validate numHashFns parameter."""
    if value < 2:
        raise ValueError("numHashFns must be at least 2.")
    return value


def _validate_feature_cardinality(value: int) -> int:
    """Validate featureCardinality parameter."""
    if value < 1:
        raise ValueError("featureCardinality must be greater than 0")
    return value


def _validate_num_bins(value: int) -> int:
    """Validate numBins parameter."""
    if value <= 0:
        raise ValueError("Number of bins must be greater than 0.")
    return value


class BloomEncodeTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Bloom encoder Spark Transformer for use in Spark pipelines.
    This transformer performs bloom encoding on the input column resulting in an
    array of integers of size equal to numHashFns.
    See paper for more details: https://arxiv.org/pdf/1706.03993.pdf
    """

    _compatible_dtypes = [StringType()]
    _keras_layer_class = BloomEncodeLayer
    _params = {
        "numHashFns": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=3,
            doc="Number of hash functions to use for bloom encoding",
            validator=_validate_num_hash_fns,
        ),
        "numBins": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=None,
            doc="Number of bins to use for hash indexing",
            validator=_validate_num_bins,
        ),
        **MASK_STRING_VALUE_PARAMS,
        "featureCardinality": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=None,
            doc="Dimension/cardinality of the feature",
            validator=_validate_feature_cardinality,
        ),
        "useHeuristicNumBins": ParamSpec(
            spark_typeconverter=TypeConverters.toBoolean,
            default=False,
            doc="Whether to use te heuristic from the paper to determine the number of bins",
        ),
    }

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
        def bloom_encode(x: List[str]) -> List[int]:
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
