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
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, IntegerType, StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import MinHashIndexLayer
from kamae.params import ParamSpec
from kamae.params.shared_specs import MASK_STRING_VALUE_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import (
    min_hash_udf,
    single_input_single_output_array_udf_transform,
)

from .base import BaseTransformer


def _validate_num_permutations(value: int) -> int:
    """Validate numPermutations parameter."""
    if value <= 0:
        raise ValueError("Number of permutations must be greater than 0.")
    return value


class MinHashIndexTransformer(
    BaseTransformer,
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

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = MinHashIndexLayer
    _params = {
        "numPermutations": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=128,
            doc="Number of permutations to perform the min hashing. Will return an array with length equal to this.",
            validator=_validate_num_permutations,
        ),
        **MASK_STRING_VALUE_PARAMS,
    }

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
