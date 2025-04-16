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
from pyspark.sql.types import DoubleType

from kamae.spark.transformers import ArrayConcatenateTransformer

from ..test_helpers import tensor_to_python_type


class TestArrayConcatenate:
    @pytest.fixture(scope="class")
    def example_dataframe_w_nested_arrays_to_concat(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [["t", "a"], ["c", "s"]],
                        [["d", "f"], ["l", "a"]],
                        [["v", "i"], ["r", "j"]],
                        [["r", "l"], ["d", "v"]],
                    ],
                    [
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                    ],
                    "5",
                ),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def vec_col1_col2_col3_nested_arrays_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [["t", "a"], ["c", "s"]],
                        [["d", "f"], ["l", "a"]],
                        [["v", "i"], ["r", "j"]],
                        [["r", "l"], ["d", "v"]],
                    ],
                    [
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                    ],
                    "5",
                    [
                        [["t", "a", "a", "b", "5"], ["c", "s", "c", "d", "5"]],
                        [["d", "f", "a", "b", "5"], ["l", "a", "c", "d", "5"]],
                        [["v", "i", "a", "b", "5"], ["r", "j", "c", "d", "5"]],
                        [["r", "l", "a", "b", "5"], ["d", "v", "c", "d", "5"]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "vec_col1_col2_col3"],
        )

    @pytest.fixture(scope="class")
    def vector_assembler_col1_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [1, 2]),
                (4, 2, 6, "b", "c", [4, 2, 6], [4, 2]),
                (7, 8, 3, "a", "a", [7, 8, 3], [7, 8]),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "vec_col1_col2"],
        )

    @pytest.fixture(scope="class")
    def vector_assembler_col1_col2_col3_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [1, 2, 3]),
                (4, 2, 6, "b", "c", [4, 2, 6], [4, 2, 6]),
                (7, 8, 3, "a", "a", [7, 8, 3], [7, 8, 3]),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "vec_col1_col2_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def vector_assembler_col4_col5_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], ["a", "c"]),
                (4, 2, 6, "b", "c", [4, 2, 6], ["b", "c"]),
                (7, 8, 3, "a", "a", [7, 8, 3], ["a", "a"]),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "vec_col4_col5"],
        )

    @pytest.fixture(scope="class")
    def vector_assembler_col1_col2_col3_col1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [1, 2, 3, 1]),
                (4, 2, 6, "b", "c", [4, 2, 6], [4, 2, 6, 4]),
                (7, 8, 3, "a", "a", [7, 8, 3], [7, 8, 3, 7]),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "vec_col1_col2_col3_col1",
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, expected_dataframe",
        [
            (
                "example_dataframe",
                ["col1", "col2"],
                "vec_col1_col2",
                "vector_assembler_col1_col2_expected",
            ),
            (
                "example_dataframe",
                ["col1", "col2", "col3"],
                "vec_col1_col2_col3",
                "vector_assembler_col1_col2_col3_expected",
            ),
            (
                "example_dataframe",
                ["col4", "col5"],
                "vec_col4_col5",
                "vector_assembler_col4_col5_expected",
            ),
            (
                "example_dataframe",
                ["col1_col2_col3", "col1"],
                "vec_col1_col2_col3_col1",
                "vector_assembler_col1_col2_col3_col1_expected",
            ),
            (
                "example_dataframe_w_nested_arrays_to_concat",
                ["col1", "col2", "col3"],
                "vec_col1_col2_col3",
                "vec_col1_col2_col3_nested_arrays_expected",
            ),
        ],
    )
    def test_spark_vector_assembler(
        self,
        input_dataframe,
        input_cols,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        example_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = ArrayConcatenateTransformer(
            inputCols=input_cols,
            outputCol=output_col,
        )
        actual = transformer.transform(example_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_vector_assembler_defaults(self):
        # when
        transformer = ArrayConcatenateTransformer()
        # then
        assert transformer.getLayerName() == transformer.uid
        assert transformer.getOutputCol() == f"{transformer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype",
        [
            (
                [
                    tf.constant([[1.0], [2.0], [3.0], [4.0], [5.0]]),
                    tf.constant([[6.0], [7.0], [8.0], [9.0], [10.0]]),
                    tf.constant([[11.0], [12.0], [13.0], [14.0], [15.0]]),
                ],
                "string",
                None,
            ),
            (
                [
                    tf.constant([[6.7], [2.3], [3.7], [4.1], [5.0111]]),
                    tf.constant([[4.7], [5.3], [3.7], [6.1], [8.0111]]),
                    tf.constant([[2.7], [67.3], [3.7], [8.1], [9.0111]]),
                    tf.constant([[45.7], [3.3], [3.7], [8.1], [10.0111]]),
                    tf.constant([[6.9], [23.3], [3.7], [10.111], [15.0111]]),
                ],
                "double",
                "float",
            ),
            (
                [
                    tf.constant([[1.1], [2.0], [3.0], [4.0], [5.0], [7.90], [345.890]]),
                    tf.constant(
                        [[6.05], [7.0], [8.0], [9.0], [10.0], [4567.0], [1000.0]]
                    ),
                ],
                None,
                None,
            ),
        ],
    )
    def test_vector_assembler_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]

        # given
        transformer = ArrayConcatenateTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                tuple([tensor_to_python_type(ti) for ti in t])
                for t in zip(*input_tensors)
            ],
            col_names,
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        decoder = lambda x: x.decode("utf-8")
        vec_decoder = np.vectorize(decoder)
        tensorflow_values = [
            vec_decoder(v) if isinstance(v[0], bytes) else v
            for v in transformer.get_tf_layer()(input_tensors).numpy().tolist()
        ]

        # then
        if isinstance(spark_values[0][0], str):
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
