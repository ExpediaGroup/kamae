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

from kamae.spark.transformers import StringArrayConstantTransformer


class TestStringArrayConstant:
    @pytest.fixture(scope="class")
    def transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], ["a", "b", "c"]),
                (4, 2, 6, "b", "c", [4, 2, 6], ["a", "b", "c"]),
                (7, 8, 3, "a", "a", [7, 8, 3], ["a", "b", "c"]),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "column_string_array",
            ],
        )

    @pytest.fixture(scope="class")
    def transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "c",
                    [1, 2, 3],
                    [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "c",
                    [4, 2, 6],
                    [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "a",
                    [7, 8, 3],
                    [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]],
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "column_string_array",
            ],
        )

    @pytest.fixture(scope="class")
    def string_array_constant_w_arrays_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["a", "b", "c"],
                        ["d", "e", "f"],
                        ["g", "h", "i"],
                        ["j", "k", "l"],
                    ],
                    [
                        ["m", "n", "o"],
                        ["p", "q", "r"],
                        ["s", "t", "u"],
                        ["v", "w", "x"],
                    ],
                    [
                        [
                            ["hello", "world", "again"],
                            ["hello", "world", "again"],
                            ["hello", "world", "again"],
                        ],
                        [
                            ["hello", "world", "again"],
                            ["hello", "world", "again"],
                            ["hello", "world", "again"],
                        ],
                        [
                            ["hello", "world", "again"],
                            ["hello", "world", "again"],
                            ["hello", "world", "again"],
                        ],
                        [
                            ["hello", "world", "again"],
                            ["hello", "world", "again"],
                            ["hello", "world", "again"],
                        ],
                    ],
                )
            ],
            ["col1", "col2", "string_array_constant_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, string_array, expected_dataframe",
        [
            (
                "example_dataframe",
                "col4",
                "column_string_array",
                ["a", "b", "c"],
                "transform_expected_1",
            ),
            # should be the same
            (
                "example_dataframe",
                "col1",
                "column_string_array",
                ["a", "b", "c"],
                "transform_expected_1",
            ),
            # should be the same but nested in a array
            (
                "example_dataframe",
                "col1_col2_col3",
                "column_string_array",
                ["a", "b", "c"],
                "transform_expected_2",
            ),
            (
                "example_dataframe_w_multiple_string_nested_arrays",
                "col1",
                "string_array_constant_col1",
                ["hello", "world", "again"],
                "string_array_constant_w_arrays_expected",
            ),
        ],
    )
    def test_spark_string_array_constant_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        string_array,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = StringArrayConstantTransformer(
            inputCol=input_col, outputCol=output_col, constantStringArray=string_array
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_string_array_constant_transform_defaults(self):
        # when
        transformer = StringArrayConstantTransformer()
        # then
        assert transformer.getLayerName() == transformer.uid
        assert transformer.getOutputCol() == f"{transformer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, string_array",
        [
            (
                tf.constant([[1.0, -5.0, -6.0, 7.0, -8.0, 9.0]]),
                None,
                "string",
                ["lala", "zzz"],
            ),
            (
                tf.constant([[-4.0, -5.0, -3.0, -47.0, -8.2, -11.0]]),
                "double",
                "string",
                ["!@£", "xxx"],
            ),
            (tf.constant([[1.0, -2.0, -3.0]]), None, None, ["random", "words"]),
            (tf.constant([[1.0, -2.0, -3.0]]), None, "bigint", ["1", "2", "3", "4"]),
            (tf.constant([[-56.4, 55.4]]), None, "boolean", ["True", "False"]),
            (tf.constant([[-76.4, -55.4]]), None, None, ["god", "speed"]),
        ],
    )
    def test_string_array_constant_transform_spark_tf_parity(
        self,
        example_dataframe,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        string_array,
    ):
        # given
        transformer = StringArrayConstantTransformer(
            inputCol="input1", outputCol="output", constantStringArray=string_array
        )
        t_list = input_tensor.numpy().T.tolist()
        spark_df = spark_session.createDataFrame(
            t_list,
            ["input1"],
        )
        # when
        spark_output_df = transformer.transform(spark_df)
        spark_values = (
            spark_output_df.select("output").rdd.map(lambda r: r[0]).collect()
        )
        # take first element (which is repeated for all rows)
        # (this drops first dimension)
        # and put it in a list to bring back the dimension
        spark_values_reshape = [spark_values[0]]
        tensorflow_values_np = transformer.get_tf_layer()(input_tensor).numpy()
        tensorflow_values = np.vectorize(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )(tensorflow_values_np).tolist()

        # then
        np.testing.assert_equal(
            spark_values_reshape,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
