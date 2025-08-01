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

from kamae.spark.transformers import CosineSimilarityTransformer


class TestCosineSimilarity:
    @pytest.fixture(scope="class")
    def example_dataframe_with_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                    ],
                    [
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                    ],
                    [
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                    ],
                    [
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4"],
        )

    @pytest.fixture(scope="class")
    def cosine_similarity_transform_col1_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                    ],
                    [
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                    ],
                    [
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                    ],
                    [
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                    ],
                    [
                        [[-0.38593899785873664, -0.38593899785873664]],
                        [[-0.38593899785873664, -0.38593899785873664]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "cosine_similarity_col1_col2"],
        )

    @pytest.fixture(scope="class")
    def cosine_similarity_transform_col3_col4_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                    ],
                    [
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                    ],
                    [
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                    ],
                    [
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                    ],
                    [
                        [[0.3400380395655728, 0.3400380395655728]],
                        [[0.3400380395655728, 0.3400380395655728]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "cosine_similarity_col3_col4"],
        )

    @pytest.fixture(scope="class")
    def cosine_similarity_transform_col1_col3_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                    ],
                    [
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                    ],
                    [
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                    ],
                    [
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                    ],
                    [
                        [[0.40214351906110246, 0.40214351906110246]],
                        [[0.40214351906110246, 0.40214351906110246]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "cosine_similarity_col1_col3"],
        )

    @pytest.fixture(scope="class")
    def cosine_similarity_transform_col2_col4_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                        [[[1.456, -15.45, 20.890], [1.456, -15.45, 20.890]]],
                    ],
                    [
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                        [[[-6.0, 5.789, 0.678], [-6.0, 5.789, 0.678]]],
                    ],
                    [
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                        [[[0.12367, 0.456, 0.7896], [0.12367, 0.456, 0.7896]]],
                    ],
                    [
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                        [[[7.456, -1.45, 2.890], [7.456, -1.45, 2.890]]],
                    ],
                    [
                        [[-0.7527191440110375, -0.7527191440110375]],
                        [[-0.7527191440110375, -0.7527191440110375]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "cosine_similarity_col2_col4"],
        )

    @pytest.mark.parametrize(
        "input_cols, output_col, expected_dataframe",
        [
            (
                ["col1", "col2"],
                "cosine_similarity_col1_col2",
                "cosine_similarity_transform_col1_col2_expected",
            ),
            (
                ["col3", "col4"],
                "cosine_similarity_col3_col4",
                "cosine_similarity_transform_col3_col4_expected",
            ),
            (
                ["col1", "col3"],
                "cosine_similarity_col1_col3",
                "cosine_similarity_transform_col1_col3_expected",
            ),
            (
                ["col2", "col4"],
                "cosine_similarity_col2_col4",
                "cosine_similarity_transform_col2_col4_expected",
            ),
        ],
    )
    def test_spark_cosine_similarity_transform(
        self,
        example_dataframe_with_arrays,
        input_cols,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = CosineSimilarityTransformer(
            inputCols=input_cols,
            outputCol=output_col,
        )
        actual = transformer.transform(example_dataframe_with_arrays)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_cosine_similarity_transform_defaults(self):
        # when
        cosine_similarity_transform = CosineSimilarityTransformer()
        # then
        assert (
            cosine_similarity_transform.getLayerName()
            == cosine_similarity_transform.uid
        )
        assert (
            cosine_similarity_transform.getOutputCol()
            == f"{cosine_similarity_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype",
        [
            (
                [
                    tf.constant(
                        [
                            [45.78, 23.09, -45.90, -67.78, -90.0, 78.0],
                            [67.89, 12.34, -0.12, 91.07, 90.0, -180.0],
                            [0.056, 0.089, -0.90, -0.78, -0.0, 0.0],
                        ]
                    ),
                    tf.constant(
                        [
                            [23.45, 76.89, -89.0, 88.07, 9.87, -18.0],
                            [120.0, 120.34, -12.98, 9.07, 9.0, -180.0],
                            [0.045, 0.089, -0.90, -0.78, -0.0, 0.0],
                        ]
                    ),
                ],
                "float",
                "double",
            ),
            (
                [
                    tf.constant(
                        [
                            [-6.2964, -2.3],
                            [-567.90, 0.45],
                            [0.0, 0.0],
                            [8.567, -0.45],
                        ]
                    ),
                    tf.constant(
                        [
                            [9.0566, 0.0],
                            [0.0056, -2.456],
                            [0.0, 0.9],
                            [10.67, -2.5680],
                        ]
                    ),
                ],
                "double",
                "float",
            ),
            (
                [
                    tf.constant(
                        [
                            [8324.45, 23.789, 0.4566, -123.0, 0.0, 0.0],
                            [86.789, -89.78, -32.789, 0.0677, 0.999, 0.0],
                        ]
                    ),
                    tf.constant(
                        [
                            [8349.8, -35.8, 67.90, 89.02, 0.001, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                ],
                None,
                None,
            ),
        ],
    )
    def test_cosine_similarity_transform_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype
    ):
        # given
        transformer = CosineSimilarityTransformer(
            inputCols=[f"input_{i}" for i in range(len(input_tensors))],
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            zip(*[input_tensor.numpy().tolist() for input_tensor in input_tensors]),
            [f"input_{i}" for i in range(len(input_tensors))],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = (
            transformer.get_tf_layer()(input_tensors).numpy().flatten().tolist()
        )

        # then
        np.testing.assert_almost_equal(
            spark_values,
            tensorflow_values,
            decimal=6,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
