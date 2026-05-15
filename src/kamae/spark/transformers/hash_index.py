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
from pyspark.sql.types import DataType, IntegerType, StringType

from kamae.keras.tensorflow.layers import HashIndexLayer
from kamae.params import ParamSpec
from kamae.params.shared_specs import MASK_STRING_VALUE_PARAMS
from kamae.spark.params import _UNSET, SingleInputSingleOutputParams
from kamae.spark.utils import hash_udf, single_input_single_output_scalar_udf_transform

from .base import BaseTransformer


def _validate_num_bins(value: int) -> int:
    """Validate numBins parameter."""
    if value <= 0:
        raise ValueError("Number of bins must be greater than 0.")
    return value


class HashIndexTransformer(
    BaseTransformer,
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

    _compatible_dtypes = [StringType()]
    _keras_layer_class = HashIndexLayer
    _params = {
        "numBins": ParamSpec(
            spark_typeconverter=TypeConverters.toInt,
            default=_UNSET,
            doc="Number of bins to use for hash indexing",
            validator=_validate_num_bins,
        ),
        **MASK_STRING_VALUE_PARAMS,
    }

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
