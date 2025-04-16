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
import pytest
import tensorflow as tf

from kamae.spark.transformers import BucketizeTransformer


class TestBucketize:
    @pytest.fixture(scope="class")
    def bucketizer_col1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1),
                (4, 2, 6, "b", "c", [4, 2, 6], 3),
                (7, 8, 3, "a", "a", [7, 8, 3], 4),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "bucket_col1"],
        )

    @pytest.fixture(scope="class")
    def bucketizer_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 2),
                (4, 2, 6, "b", "c", [4, 2, 6], 2),
                (7, 8, 3, "a", "a", [7, 8, 3], 5),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "bucket_col2"],
        )

    @pytest.fixture(scope="class")
    def bucketizer_col1_2_3_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [2, 2, 2]),
                (4, 2, 6, "b", "c", [4, 2, 6], [2, 2, 4]),
                (7, 8, 3, "a", "a", [7, 8, 3], [5, 6, 2]),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "bucket_col1_col2_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def bucketizer_array_col1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [1.0, -2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, -3.0],
                        [4.0, 2.0, -6.0],
                    ],
                    [
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                    ],
                    [[1, 1, 3], [1, 2, 3], [1, 2, 1], [3, 2, 1]],
                ),
                (
                    [
                        [4.0, -2.0, 6.0],
                        [4.0, -2.0, 6.0],
                        [4.0, 2.0, -6.0],
                        [7.0, 8.0, 3.0],
                    ],
                    [
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                    ],
                    [[3, 1, 4], [3, 1, 4], [3, 2, 1], [4, 4, 3]],
                ),
                (
                    [
                        [7.0, 8.0, 3.0],
                        [7.0, -8.0, 3.0],
                        [7.0, 8.0, -3.0],
                        [-1.0, 2.0, -3.0],
                    ],
                    [
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                    ],
                    [[4, 4, 3], [4, 1, 3], [4, 4, 1], [1, 2, 1]],
                ),
            ],
            ["col1", "col2", "bucket_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, splits, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1",
                "bucket_col1",
                [2.0, 3.0, 5.0],
                "bucketizer_col1_expected",
            ),
            (
                "example_dataframe",
                "col2",
                "bucket_col2",
                [1.0, 5.0, 7.0, 7.5],
                "bucketizer_col2_expected",
            ),
            (
                "example_dataframe",
                "col1_col2_col3",
                "bucket_col1_col2_col3",
                [1.0, 5.0, 6.0, 7.0, 7.5],
                "bucketizer_col1_2_3_expected",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "bucket_col1",
                [2.0, 3.0, 5.0],
                "bucketizer_array_col1_expected",
            ),
        ],
    )
    def test_spark_bucketizer(
        self,
        input_dataframe,
        input_col,
        output_col,
        splits,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = BucketizeTransformer(
            inputCol=input_col,
            outputCol=output_col,
            splits=splits,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_spark_bucketizer_throws_error(self):
        with pytest.raises(ValueError):
            # Throws error because splits are not in ascending order
            transformer = BucketizeTransformer(
                inputCol="col1",
                outputCol="bucket_col1",
                splits=[10.0, 3.0, 5.0],
            )

    def test_bucketizer_defaults(self):
        # when
        bucketizer = BucketizeTransformer()
        # then
        assert bucketizer.getLayerName() == bucketizer.uid
        assert bucketizer.getOutputCol() == f"{bucketizer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, splits",
        [
            (tf.constant([1.0, 4.0, 7.0, 8.0]), "int", "bigint", [2.0, 3.0, 5.0]),
            (tf.constant([2.0, 5.0, 1.0]), "double", "string", [1.0, 5.0, 7.0, 7.5]),
            (tf.constant([-1.0, 7.0]), "float", "string", [-11.0, 1.0, 6.0, 7.0, 7.5]),
            (tf.constant([2.0, 5.0, 1.0, 5.0, 2.5]), None, None, [1.0]),
        ],
    )
    def test_bucketizer_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, splits
    ):
        # given
        transformer = BucketizeTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            splits=splits,
        )
        # when
        if input_tensor.dtype.name == "string":
            spark_df = spark_session.createDataFrame(
                [(v.decode("utf-8"),) for v in input_tensor.numpy().tolist()], ["input"]
            )
        else:
            spark_df = spark_session.createDataFrame(
                [(v,) for v in input_tensor.numpy().tolist()], ["input"]
            )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
