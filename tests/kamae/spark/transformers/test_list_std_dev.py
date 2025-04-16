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

from kamae.spark.transformers import ListStdDevTransformer

from ..test_helpers import tensor_to_python_type


class TestListStdDev:
    @pytest.fixture(scope="class")
    def listwise_transform_df_no_filter(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3.0),
                (1, 2, 3.0),
                (1, 2, 3.0),
                (1, 8, 3.0),
                (2, 10, 7.071068),
                (2, 20, 7.071068),
                (3, None, 0.0),  # should be ignored
                (3, 5, 0.0),
            ],
            [
                "search_id",
                "value_col",
                "expected",
            ],
        )

    @pytest.fixture(scope="class")
    def listwise_transform_df_min_value(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, -999, 3.464101552963257),  # should be ignored
                (1, 2, 3.464101552963257),
                (1, 2, 3.464101552963257),
                (1, 8, 3.464101552963257),
                (2, -999, 0.0),  # should be ignored
                (2, 20, 0.0),
                (3, None, 0.0),  # should be ignored
                (3, 5, 0.0),
            ],
            [
                "search_id",
                "value_col",
                "expected",
            ],
        )

    @pytest.fixture(scope="class")
    def listwise_transform_df_sort_desc(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 1, 1, 3.464101552963257),  # should be ignored in top3 desc
                (1, 2, 2, 3.464101552963257),
                (1, 2, 3, 3.464101552963257),
                (1, 8, 4, 3.464101552963257),
            ],
            [
                "search_id",
                "value_col",
                "sort_col",
                "expected",
            ],
        )

    @pytest.fixture(scope="class")
    def listwise_transform_df_sort_asc(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 5, 1, 1.7320507764816284),
                (1, 2, 2, 1.7320507764816284),
                (1, 2, 3, 1.7320507764816284),
                (1, 8, 4, 1.7320507764816284),  # should be ignored in top3 asc
            ],
            [
                "search_id",
                "value_col",
                "sort_col",
                "expected",
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, value_col, min_filter_value, output_col, input_dtype, output_dtype",
        [
            (
                "listwise_transform_df_no_filter",
                "value_col",
                None,
                "expected",
                "float",
                "float",
            ),
            (
                "listwise_transform_df_min_value",
                "value_col",
                0.0,
                "expected",
                "float",
                "float",
            ),
        ],
    )
    def test_spark_stddev_transform(
        self,
        input_dataframe,
        value_col,
        min_filter_value,
        output_col,
        input_dtype,
        output_dtype,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        transformer = ListStdDevTransformer(
            inputCol=value_col,
            outputCol=output_col,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            queryIdCol="search_id",
            minFilterValue=min_filter_value,
        )
        actual = transformer.transform(input_dataframe.drop("expected"))
        # then
        expected = input_dataframe.select(
            F.col("expected").cast(output_dtype).alias("expected")
        )
        diff = actual.select("expected").exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_dataframe, value_col, sort_col, top_n, sort_order, output_col, input_dtype, output_dtype",
        [
            (
                "listwise_transform_df_sort_desc",
                "value_col",
                "sort_col",
                3,
                "desc",
                "expected",
                "float",
                "float",
            ),
            (
                "listwise_transform_df_sort_asc",
                "value_col",
                "sort_col",
                3,
                "asc",
                "expected",
                "float",
                "float",
            ),
        ],
    )
    def test_spark_stddev_transform_with_sort(
        self,
        input_dataframe,
        value_col,
        sort_col,
        top_n,
        sort_order,
        output_col,
        input_dtype,
        output_dtype,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        transformer = ListStdDevTransformer(
            inputCols=[value_col, sort_col],
            outputCol=output_col,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            queryIdCol="search_id",
            topN=top_n,
            sortOrder=sort_order,
        )
        actual = transformer.transform(input_dataframe.drop("expected"))
        # then
        expected = input_dataframe.select(
            F.col("expected").cast(output_dtype).alias("expected")
        )
        diff = actual.select("expected").exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "list_size, qid_tensor, input_tensors, min_filter_value, top_n, input_dtype, output_dtype",
        [
            # Base case
            (
                8,
                tf.constant(
                    [
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                    ],
                    dtype=tf.float32,
                ),
                [
                    # values
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [9.0],
                            [4.0],
                            [6.0],
                            [2.0],
                            [0.0],
                            [0.0],
                            [5.0],
                            [1.0],
                            [9.0],
                            [4.0],
                            [6.0],
                            [8.0],
                            [0.0],
                            [0.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                None,
                None,
                "double",
                "float",
            ),
            # With min_filter_value
            (
                8,
                tf.constant(
                    [
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                    ],
                    dtype=tf.float32,
                ),
                [
                    # values
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [9.0],
                            [4.0],
                            [6.0],
                            [2.0],
                            [0.0],
                            [0.0],
                            [5.0],
                            [1.0],
                            [9.0],
                            [4.0],
                            [6.0],
                            [8.0],
                            [0.0],
                            [0.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                1,
                None,
                "double",
                "float",
            ),
            # With top_n
            (
                8,
                tf.constant(
                    [
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                    ],
                    dtype=tf.float32,
                ),
                [
                    # values
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [9.0],
                            [4.0],
                            [6.0],
                            [2.0],
                            [0.0],
                            [0.0],
                            [5.0],
                            [1.0],
                            [9.0],
                            [4.0],
                            [6.0],
                            [8.0],
                            [0.0],
                            [0.0],
                        ],
                        dtype=tf.float32,
                    ),
                    # sort
                    tf.constant(
                        [
                            [1.0],
                            [2.0],
                            [3.0],
                            [4.0],
                            [5.0],
                            [6.0],
                            [7.0],
                            [8.0],
                            [8.0],
                            [7.0],
                            [6.0],
                            [5.0],
                            [4.0],
                            [3.0],
                            [2.0],
                            [1.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                None,
                5,
                "double",
                "float",
            ),
            # With top_n and filter
            (
                8,
                tf.constant(
                    [
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                        [2],
                    ],
                    dtype=tf.float32,
                ),
                [
                    # values
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [9.0],
                            [4.0],
                            [6.0],
                            [2.0],
                            [0.0],
                            [0.0],
                            [5.0],
                            [1.0],
                            [9.0],
                            [4.0],
                            [6.0],
                            [8.0],
                            [0.0],
                            [0.0],
                        ],
                        dtype=tf.float32,
                    ),
                    # sort
                    tf.constant(
                        [
                            [1.0],
                            [2.0],
                            [3.0],
                            [4.0],
                            [5.0],
                            [6.0],
                            [7.0],
                            [8.0],
                            [8.0],
                            [7.0],
                            [6.0],
                            [5.0],
                            [4.0],
                            [3.0],
                            [2.0],
                            [1.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                1,
                5,
                "double",
                "float",
            ),
            # With top_n > list size
            (
                3,
                tf.constant(
                    [
                        [1],
                        [1],
                        [1],
                        [2],
                        [2],
                        [2],
                    ],
                    dtype=tf.float32,
                ),
                [
                    # values
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [9.0],
                            [5.0],
                            [1.0],
                            [9.0],
                        ],
                        dtype=tf.float32,
                    ),
                    # sort
                    tf.constant(
                        [
                            [1.0],
                            [2.0],
                            [3.0],
                            [8.0],
                            [7.0],
                            [6.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                1,
                5,
                "double",
                "float",
            ),
        ],
    )
    def test_list_average_transform_spark_tf_parity(
        self,
        spark_session,
        list_size,
        qid_tensor,
        input_tensors,
        min_filter_value,
        top_n,
        input_dtype,
        output_dtype,
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = ListStdDevTransformer(
            inputCol=col_names[0] if len(col_names) == 1 else None,
            inputCols=col_names if len(col_names) > 1 else None,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            queryIdCol="search_id",
            minFilterValue=min_filter_value,
            topN=top_n,
            sortOrder="asc",
        )
        # when
        qid_inputs_tensors = [qid_tensor] + input_tensors
        qid_col_names = ["search_id"] + col_names
        spark_df = spark_session.createDataFrame(
            [
                tuple([tensor_to_python_type(ti) for ti in t])
                for t in zip(*qid_inputs_tensors)
            ],
            qid_col_names,
        )

        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        # reshape the input tensors to match the expected shape based on list size
        input_tensors = [tf.reshape(t, (-1, list_size, 1)) for t in input_tensors]
        tensorflow_values = np.reshape(
            [
                np.squeeze(v)
                for v in transformer.get_tf_layer()(input_tensors).numpy().tolist()
            ],
            -1,
        )

        # then
        if isinstance(spark_values[0], str):
            np.testing.assert_equal(
                spark_values,
                tensorflow_values,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
        else:
            np.testing.assert_almost_equal(
                spark_values,
                tensorflow_values,
                decimal=6,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
