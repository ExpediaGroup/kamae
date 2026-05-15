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

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import (
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.keras.tensorflow.layers import ListRankLayer
from kamae.params import ParamSpec
from kamae.params.shared_specs import LISTWISE_PARAMS, _validate_sort_order
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import check_listwise_columns

from .base import BaseTransformer


class ListRankTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Calculate the listwise rank across the query id column.
    """

    jit_compatible = True

    _compatible_dtypes = [
        FloatType(),
        DoubleType(),
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType(),
    ]
    _keras_layer_class = ListRankLayer
    _params = {
        **LISTWISE_PARAMS,
        "sortOrder": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="desc",
            doc="Either 'asc' for ascending values or 'desc' for descending values.",
            validator=_validate_sort_order,
        ),
        "withSegment": ParamSpec(
            spark_typeconverter=TypeConverters.toBoolean,
            default=None,
            doc="Whether the second input col should be used for segmentation of statistic calculation.",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Calculate the rank.
        :param dataset: The dataframe with signals and features.
        :returns: The dataframe dataset with the new feature.
        """
        if not self.isDefined("queryIdCol"):
            raise ValueError("queryIdCol must be set on listwise transformers.")

        # Get params
        input_col = self.getInputCol()
        query_id_col = self.getQueryIdCol()
        output_col = self.getOutputCol()
        sort_order = self.getSortOrder()

        # Validate listwise cols
        check_listwise_columns(
            dataset=dataset,
            query_col_name=query_id_col,
            value_col_name=input_col,
            sort_col_name=None,
        )

        # Set sort order
        if sort_order == "asc":
            sort_col = F.col(input_col).asc()
        elif sort_order == "desc":
            sort_col = F.col(input_col).desc()
        else:
            raise ValueError(f"Invalid sortOrder: {sort_order}")
        # Define window spec
        window_spec = Window.partitionBy(query_id_col).orderBy(sort_col)

        # Calculate the rank
        dataset = dataset.withColumn(output_col, F.row_number().over(window_spec))

        return dataset
