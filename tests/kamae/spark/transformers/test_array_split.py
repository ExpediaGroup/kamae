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

from kamae.spark.transformers import ArraySplitTransformer


class TestArraySplit:
    @pytest.fixture(scope="class")
    def vector_slicer_col1_col2_col3_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1, 2, 3),
                (4, 2, 6, "b", "c", [4, 2, 6], 4, 2, 6),
                (7, 8, 3, "a", "a", [7, 8, 3], 7, 8, 3),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "slice_col1",
                "slice_col2",
                "slice_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def array_split_nested_expected(self, spark_session):
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
                    [1.0, 1.0, 1.0, 4.0],
                    [-2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, -3.0, -6.0],
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
                    [4.0, 4.0, 4.0, 7.0],
                    [-2.0, -2.0, 2.0, 8.0],
                    [6.0, 6.0, -6.0, 3.0],
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
                    [7.0, 7.0, 7.0, -1.0],
                    [8.0, -8.0, 8.0, 2.0],
                    [3.0, 3.0, -3.0, -3.0],
                ),
            ],
            ["col1", "col2", "slice_1", "slice_2", "slice_3"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_cols, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1_col2_col3",
                ["slice_col1", "slice_col2", "slice_col3"],
                "vector_slicer_col1_col2_col3_expected",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                ["slice_1", "slice_2", "slice_3"],
                "array_split_nested_expected",
            ),
        ],
    )
    def test_spark_vector_slicer(
        self,
        input_dataframe,
        input_col,
        output_cols,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = ArraySplitTransformer(
            inputCol=input_col,
            outputCols=output_cols,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_vector_slicer_defaults(self):
        # when
        transformer = ArraySplitTransformer()
        # then
        assert transformer.getLayerName() == transformer.uid

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype",
        [
            (
                tf.constant(
                    [
                        [1.0, 6.0, 11.0],
                        [2.0, 7.0, 12.0],
                        [3.0, 8.0, 13.0],
                        [4.0, 9.0, 14.0],
                        [5.0, 10.0, 15.0],
                    ]
                ),
                "string",
                "double",
            ),
            (
                tf.constant(
                    [
                        [6.7, 4.7, 2.7, 45.7, 6.9],
                        [2.3, 5.3, 67.3, 3.3, 23.3],
                        [3.7, 3.7, 3.7, 3.7, 3.7],
                        [4.1, 6.1, 8.1, 8.1, 10.111],
                        [5.0111, 8.0111, 9.0111, 10.0111, 15.0111],
                    ]
                ),
                "double",
                None,
            ),
            (
                tf.constant(
                    [
                        [1.1, 6.05],
                        [2.0, 7.0],
                        [3.0, 8.0],
                        [4.0, 9.0],
                        [5.0, 10.0],
                        [7.90, 4567.0],
                        [345.890, 1000.0],
                    ]
                ),
                None,
                None,
            ),
        ],
    )
    def test_vector_slicer_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype
    ):
        col_names = [f"output{i}" for i in range(input_tensor.shape[1])]

        # given
        transformer = ArraySplitTransformer(
            inputCol="input",
            outputCols=col_names,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                (v.decode("utf-8") if isinstance(v, bytes) else v,)
                for v in input_tensor.numpy().tolist()
            ],
            ["input"],
        )

        spark_values = [
            transformer.transform(spark_df).select(c).rdd.map(lambda r: r[0]).collect()
            for c in col_names
        ]

        decoder = lambda x: x.decode("utf-8")
        vec_decoder = np.vectorize(decoder)
        tensorflow_values = [
            vec_decoder(v.numpy().tolist()) if isinstance(v[0], bytes) else v
            for v in transformer.get_tf_layer()(input_tensor)
        ]

        # then
        np.testing.assert_almost_equal(
            np.array(spark_values).flatten(),
            np.array(tensorflow_values).flatten(),
            decimal=6,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
