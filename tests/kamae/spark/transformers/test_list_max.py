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

from kamae.spark.transformers import ListMaxTransformer

from ..test_helpers import tensor_to_python_type


class TestListMax:
    @pytest.fixture(scope="class")
    def listwise_transform_df_no_filter(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 8.0),
                (1, 2, 8.0),
                (1, 2, 8.0),
                (1, 8, 8.0),
                (2, 10, 20.0),
                (2, 20, 20.0),
                (3, None, 5.0),  # should be ignored
                (3, 5, 5.0),
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
                (1, -999, 8.0),  # should be ignored
                (1, 2, 8.0),
                (1, 2, 8.0),
                (1, 8, 8.0),
                (2, -999, 20.0),  # should be ignored
                (2, 20, 20.0),
                (3, None, 5.0),  # should be ignored
                (3, 5, 5.0),
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
                (1, 1, 1, 8.0),  # should be ignored in top3 desc
                (1, 2, 2, 8.0),
                (1, 2, 3, 8.0),
                (1, 8, 4, 8.0),
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
                (1, 5, 1, 5.0),
                (1, 2, 2, 5.0),
                (1, 2, 3, 5.0),
                (1, 8, 4, 5.0),  # should be ignored in top3 asc
            ],
            [
                "search_id",
                "value_col",
                "sort_col",
                "expected",
            ],
        )

    @pytest.fixture(scope="class")
    def listwise_transform_df_segment(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 5, 1, 8.0),
                (1, 2, 2, 2.0),
                (1, 2, 2, 2.0),
                (1, 8, 1, 8.0),
            ],
            [
                "search_id",
                "value_col",
                "segment_col",
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
    def test_spark_max_transform(
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
        transformer = ListMaxTransformer(
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
    def test_spark_max_transform_with_sort(
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
        transformer = ListMaxTransformer(
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
        "input_dataframe, value_col, segment_col, output_col, input_dtype, output_dtype",
        [
            (
                "listwise_transform_df_segment",
                "value_col",
                "segment_col",
                "expected",
                "float",
                "float",
            ),
        ],
    )
    def test_spark_mean_transform_with_segmentation(
        self,
        input_dataframe,
        value_col,
        segment_col,
        output_col,
        input_dtype,
        output_dtype,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        transformer = ListMaxTransformer(
            inputCols=[value_col, segment_col],
            outputCol=output_col,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            queryIdCol="search_id",
            withSegment=True,
        )
        actual = transformer.transform(input_dataframe.drop("expected"))
        # then
        expected = input_dataframe.select(
            F.col("expected").cast(output_dtype).alias("expected")
        )
        diff = actual.select("expected").exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "list_size, qid_tensor, input_tensors, min_filter_value, with_segment, top_n, input_dtype, output_dtype",
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
                False,
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
                False,
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
                False,
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
                False,
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
                False,
                5,
                "double",
                "float",
            ),
            # With Int
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
                            [1],
                            [1],
                            [9],
                            [5],
                            [1],
                            [9],
                        ],
                        dtype=tf.int32,
                    ),
                    # sort
                    tf.constant(
                        [
                            [1],
                            [2],
                            [3],
                            [8],
                            [7],
                            [6],
                        ],
                        dtype=tf.int32,
                    ),
                ],
                1,
                False,
                5,
                "int",
                "float",
            ),
            # With segmentation
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
                            [4.0],
                            [5.0],
                            [5.0],
                            [20.0],
                        ],
                        dtype=tf.float32,
                    ),
                    # segment
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [2.0],
                            [1.0],
                            [1.0],
                            [2.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                0,
                True,
                None,
                "double",
                "float",
            ),
            # With segmentation & filter
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
                            [4.0],
                            [5.0],
                            [5.0],
                            [20.0],
                        ],
                        dtype=tf.float32,
                    ),
                    # segment
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [2.0],
                            [1.0],
                            [1.0],
                            [2.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                2.0,
                True,
                None,
                "double",
                "float",
            ),
        ],
    )
    def test_list_max_transform_spark_tf_parity(
        self,
        spark_session,
        list_size,
        qid_tensor,
        input_tensors,
        min_filter_value,
        with_segment,
        top_n,
        input_dtype,
        output_dtype,
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = ListMaxTransformer(
            inputCol=col_names[0] if len(col_names) == 1 else None,
            inputCols=col_names if len(col_names) > 1 else None,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            queryIdCol="search_id",
            minFilterValue=min_filter_value,
            topN=top_n,
            withSegment=with_segment,
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
