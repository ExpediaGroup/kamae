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
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from kamae.spark.utils.transform_utils import (
    single_input_single_output_scalar_udf_transform,
)


class TestTransformUtils:
    def test_scalar_udf_transform_maps_numeric_null_to_none(self, spark_session):
        """
        Spark NULLs in a nullable numeric column must reach the element func as Python
        None, not Arrow's NaN. Otherwise `is None` null/OOV guards in the element funcs
        (indexer/hash UDFs) silently misroute missing values. The func here would raise
        on NaN (int(NaN)), so this fails if the vectorized path leaks NaN through.
        """
        schema = StructType([StructField("x", DoubleType(), True)])
        df = spark_session.createDataFrame([(1.0,), (None,), (2.0,)], schema)

        out = df.withColumn(
            "y",
            single_input_single_output_scalar_udf_transform(
                input_col=F.col("x"),
                input_col_datatype=df.schema["x"].dataType,
                func=lambda v: -1 if v is None else int(v),
                udf_return_element_datatype=IntegerType(),
            ),
        )

        result = sorted(row["y"] for row in out.collect())
        assert result == [-1, 1, 2]
