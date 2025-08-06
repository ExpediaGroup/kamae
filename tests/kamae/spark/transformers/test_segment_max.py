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

from kamae.spark.transformers import SegmentMaxTransformer

from ..test_helpers import tensor_to_python_type


class TestSegmentMax:
    @pytest.fixture(scope="class")
    def segment_max(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 1, 2, 8.0),
                (1, 1, 2, 8.0),
                (1, 1, 2, 8.0),
                (1, 1, 8, 8.0),
                (1, 2, 10, 20.0),
                (1, 2, 20, 20.0),
                (1, 3, None, 5.0),
                (1, 3, 5, 5.0),
                (1, None, 5, 10.0),
                (1, None, 10, 10.0),
            ],
            [
                "search_id",
                "segment_col",
                "value_col",
                "expected",
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, value_col, segment_col, min_filter_value, output_col, input_dtype, output_dtype",
        [
            (
                "segment_max",
                "value_col",
                "segment_col",
                None,
                "expected",
                "float",
                "float",
            ),
        ],
    )
    def test_spark_segment_max_transform(
        self,
        input_dataframe,
        value_col,
        segment_col,
        min_filter_value,
        output_col,
        input_dtype,
        output_dtype,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        transformer = SegmentMaxTransformer(
            inputCols=[value_col, segment_col],
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
        "list_size, qid_tensor, input_tensors,input_dtype, output_dtype",
        [
            # Multi batch case
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
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [2.0],
                            [2.0],
                            [2.0],
                            [2.0],
                            [3.0],
                            [3.0],
                            [3.0],
                            [3.0],
                            [3.0],
                            [3.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                None,
                "float",
            ),
            # single batch
            (
                16,
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
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                    ],
                    dtype=tf.float32,
                ),
                [
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
                    tf.constant(
                        [
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [2.0],
                            [2.0],
                            [2.0],
                            [2.0],
                            [3.0],
                            [3.0],
                            [3.0],
                            [3.0],
                            [3.0],
                            [3.0],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                None,
                "float",
            ),
        ],
    )
    def test_segment_max_transform_spark_tf_parity(
        self,
        spark_session,
        list_size,
        qid_tensor,
        input_tensors,
        input_dtype,
        output_dtype,
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = SegmentMaxTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            queryIdCol="search_id",
        )

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
