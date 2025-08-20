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

import numpy as np
import pyspark.sql.functions as F
import pytest
import tensorflow as tf

from kamae.spark.transformers import ListRankTransformer

from ..test_helpers import tensor_to_python_type


class TestListRank:
    @pytest.fixture(scope="class")
    def listwise_rank_df(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 1.0, 4),
                (1, 1.5, 5),
                (1, 9.0, 9),
                (1, 4.0, 7),
                (1, 6.0, 8),
                (1, 2.0, 6),
                (1, 0.5, 3),
                (1, 0.0, 1),
                (1, 0.0, 2),
                (2, 1.0, 3),
                (2, 2.0, 2),
                (2, 3.0, 1),
            ],
            [
                "search_id",
                "value_col",
                "expected",
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, value_col, output_col, input_dtype, output_dtype",
        [
            (
                "listwise_rank_df",
                "value_col",
                "expected",
                "float",
                "float",
            ),
        ],
    )
    def test_spark_rank_transform(
        self,
        input_dataframe,
        value_col,
        output_col,
        input_dtype,
        output_dtype,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        transformer = ListRankTransformer(
            inputCol=value_col,
            outputCol=output_col,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            queryIdCol="search_id",
        )
        actual = transformer.transform(input_dataframe.drop("expected"))
        # then
        expected = input_dataframe.select(
            F.col("expected").cast(output_dtype).alias("expected")
        )
        diff = actual.select("expected").exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype",
        [
            (
                tf.constant(
                    [[[1.0], [1.5], [9.0], [4.0], [6.0], [2.0], [0.5], [0.0], [0.0]]],
                    dtype=tf.float32,
                ),
                None,
                None,
            ),
        ],
    )
    def test_list_rank_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
    ):
        # given
        transformer = ListRankTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            queryIdCol="search_id",
        )
        # when
        spark_df = spark_session.createDataFrame(
            [(1, v[0]) for v in input_tensor.numpy().tolist()[0]],
            ["search_id", "input"],
        )

        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = transformer.get_tf_layer()(input_tensor).numpy().tolist()

        # then
        np.testing.assert_almost_equal(
            spark_values,
            tensorflow_values,
            decimal=6,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
