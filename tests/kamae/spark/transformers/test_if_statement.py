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

from kamae.spark.transformers import IfStatementTransformer
from tests.kamae.spark.test_helpers import tensor_to_python_type


class TestIfStatement:
    @pytest.fixture(scope="class")
    def example_dataframe_with_three_string_columns(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", "c", "a"),
                ("b", "c", "b"),
                ("a", "a", "a"),
            ],
            [
                "col1",
                "col2",
                "col3",
            ],
        )

    @pytest.fixture(scope="class")
    def string_if_statement_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", "c", "a", "TRUE"),
                ("b", "c", "b", "FALSE"),
                ("a", "a", "a", "TRUE"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "string_if_statement_col1_w_constant",
            ],
        )

    @pytest.fixture(scope="class")
    def string_if_statement_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", "c", "a", "a"),
                ("b", "c", "b", "b"),
                ("a", "a", "a", "TRUE"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "string_if_statement_col1_col2_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def string_if_statement_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", "c", "a", "a"),
                ("b", "c", "b", "FALSE"),
                ("a", "a", "a", "a"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "string_if_statement_col1_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def string_if_statement_transform_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", "c", "a", "TRUE"),
                ("b", "c", "b", "TRUE"),
                ("a", "a", "a", "FALSE"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "string_if_statement_col2_w_constant",
            ],
        )

    @pytest.fixture(scope="class")
    def string_if_statement_transform_array_expected(self, spark_session):
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
                        ["TRUE", "FALSE", "FALSE"],
                        ["FALSE", "FALSE", "FALSE"],
                        ["FALSE", "FALSE", "FALSE"],
                        ["FALSE", "FALSE", "FALSE"],
                    ],
                )
            ],
            ["col1", "col2", "string_if_statement_col1_array_w_constant"],
        )

    @pytest.fixture(scope="class")
    def if_statement_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], -1.0),
                (4, 2, 6, "b", "c", [4, 2, 6], 1.0),
                (7, 8, 3, "a", "a", [7, 8, 3], 1.0),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "if_statement_col1_w_constants",
            ],
        )

    @pytest.fixture(scope="class")
    def if_statement_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1.0),
                (4, 2, 6, "b", "c", [4, 2, 6], 6.0),
                (7, 8, 3, "a", "a", [7, 8, 3], 1.0),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "if_statement_col1_col2_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def if_statement_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1.0),
                (4, 2, 6, "b", "c", [4, 2, 6], -1.0),
                (7, 8, 3, "a", "a", [7, 8, 3], 7.0),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "if_statement_col1_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def if_statement_transform_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], -1.0),
                (4, 2, 6, "b", "c", [4, 2, 6], -1.0),
                (7, 8, 3, "a", "a", [7, 8, 3], 1.0),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "if_statement_col2_w_constant",
            ],
        )

    @pytest.fixture(scope="class")
    def if_statement_transform_array_expected(self, spark_session):
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
                        [4.0, 5.0, -1.2],
                        [41.0, -89.45, 56.5],
                        [14.0, -6.0, 9.5],
                        [43.45, -2.0, 4.5],
                    ],
                    [
                        [-1.0, -1.0, 1.0],
                        [-1.0, 1.0, 1.0],
                        [-1.0, 1.0, -1.0],
                        [1.0, 1.0, -1.0],
                    ],
                )
            ],
            ["col1", "col2", "if_statement_col1_array_w_constant"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, input_cols, condition_operator, value_to_compare, result_if_true, result_if_false, output_col, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1",  # input
                None,
                "geq",  # condition_operator
                4.0,  # value_to_compare
                1.0,  # result_if_true
                -1.0,  # result_if_false
                "if_statement_col1_w_constant",
                "if_statement_transform_expected_1",
            ),
            (
                "example_dataframe",
                None,
                ["col1", "col2", "col3"],  # input, value_to_compare, result_if_false
                "lt",  # condition_operator
                None,
                1.0,  # result_if_true
                None,
                "if_statement_col1_col2_col3",
                "if_statement_transform_expected_2",
            ),
            (
                "example_dataframe",
                None,
                ["col3", "col1"],  # input, result_if_true
                "eq",  # condition_operator
                3.0,  # value_to_compare
                None,
                -1.0,  # result_if_false
                "if_statement_col1_col3",
                "if_statement_transform_expected_3",
            ),
            (
                "example_dataframe",
                "col2",  # input
                None,
                "gt",  # condition_operator
                5.0,  # value_to_compare
                1.0,  # result_if_true
                -1.0,  # result_if_false
                "if_statement_col2_w_constant",
                "if_statement_transform_expected_4",
            ),
            (
                "example_dataframe_w_multiple_numeric_nested_arrays",
                "col1",  # input
                None,
                "geq",  # condition_operator
                2.0,  # value_to_compare
                1.0,  # result_if_true
                -1.0,  # result_if_false
                "if_statement_col1_array_w_constant",
                "if_statement_transform_array_expected",
            ),
            (
                "example_dataframe_with_three_string_columns",
                "col1",  # input
                None,
                "eq",  # condition_operator
                "a",  # value_to_compare
                "TRUE",  # result_if_true
                "FALSE",  # result_if_false
                "string_if_statement_col1_w_constant",
                "string_if_statement_transform_expected_1",
            ),
            (
                "example_dataframe_with_three_string_columns",
                None,
                ["col1", "col2", "col3"],  # input, value_to_compare, result_if_false
                "eq",  # condition_operator
                None,
                "TRUE",  # result_if_true
                None,
                "string_if_statement_col1_col2_col3",
                "string_if_statement_transform_expected_2",
            ),
            (
                "example_dataframe_with_three_string_columns",
                None,
                ["col3", "col1"],  # input, result_if_true
                "eq",  # condition_operator
                "a",  # value_to_compare
                None,
                "FALSE",  # result_if_false
                "string_if_statement_col1_col3",
                "string_if_statement_transform_expected_3",
            ),
            (
                "example_dataframe_with_three_string_columns",
                "col2",  # input
                None,
                "eq",  # condition_operator
                "c",  # value_to_compare
                "TRUE",  # result_if_true
                "FALSE",  # result_if_false
                "string_if_statement_col2_w_constant",
                "string_if_statement_transform_expected_4",
            ),
            (
                "example_dataframe_w_multiple_string_nested_arrays",
                "col1",  # input
                None,
                "eq",  # condition_operator
                "a",  # value_to_compare
                "TRUE",  # result_if_true
                "FALSE",  # result_if_false
                "string_if_statement_col1_array_w_constant",
                "string_if_statement_transform_array_expected",
            ),
        ],
    )
    def test_spark_if_statement_transform(
        self,
        input_dataframe,
        input_col,
        input_cols,
        condition_operator,
        value_to_compare,
        result_if_true,
        result_if_false,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = IfStatementTransformer(
            conditionOperator=condition_operator,
            outputCol=output_col,
            inputCol=input_col,
            inputCols=input_cols,
            valueToCompare=value_to_compare,
            resultIfTrue=result_if_true,
            resultIfFalse=result_if_false,
        )

        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_if_statement_transform_defaults(self):
        # when
        if_statement_transformer = IfStatementTransformer()
        # then
        assert if_statement_transformer.getLayerName() == if_statement_transformer.uid
        assert (
            if_statement_transformer.getOutputCol()
            == f"{if_statement_transformer.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, condition_operator, value_to_compare, result_if_true, result_if_false",
        [
            (tf.constant([1.0, 2.0, 3.0]), None, None, "geq", 4.0, 1.0, -1.0),
            (
                tf.constant([10, 2, 3], dtype="int32"),
                "double",
                None,
                "lt",
                4.0,
                1.0,
                -1.0,
            ),
            (tf.constant([1.0, 2.0, 3.0]), "float", None, "eq", 3.0, 10.0, -10.0),
            (tf.constant([1.0, 2.0, 3.0]), None, None, "neq", 1.0, 100.0, -100.0),
            (tf.constant(["a", "b", "c"]), None, "string", "eq", "a", "TRUE", "FALSE"),
            (tf.constant([1, 2, 3]), "string", None, "eq", "2", "TRUE", "FALSE"),
            (
                tf.constant([True, False, False]),
                "string",
                "boolean",
                "eq",
                "False",
                "TRUE",
                "FALSE",
            ),
        ],
    )
    def test_if_statement_transform_single_input_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        condition_operator,
        value_to_compare,
        result_if_true,
        result_if_false,
    ):
        # given
        transformer = IfStatementTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            conditionOperator=condition_operator,
            valueToCompare=value_to_compare,
            resultIfTrue=result_if_true,
            resultIfFalse=result_if_false,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                (v.decode("utf-8") if isinstance(v, bytes) else v,)
                for v in input_tensor.numpy().tolist()
            ],
            ["input"],
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

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, condition_operator",
        [
            (
                [
                    tf.constant([1.0, -5.0, -6.0, 7.0, -8.0, 9.0], dtype=tf.float32),
                    tf.constant(
                        [10.0, -56.0, -6.0, 14.0, -3.2, 9.56], dtype=tf.float32
                    ),
                    tf.constant([1.0, -5.0, -3.0, 2.0, -4.25, -1.5], dtype=tf.float32),
                    tf.constant(
                        [2.0, -52.0, -32.0, 22.0, -42.25, -12.5], dtype=tf.float32
                    ),
                ],
                "double",
                None,
                "leq",
            ),
            (
                [
                    tf.constant(
                        [-4.0, -5.0, -3.0, -47.0, -8.0, -11.0], dtype=tf.float32
                    ),
                    tf.constant(
                        [-2.0, -17.0, -3.45, -4.2, -0.1, -1.0], dtype=tf.float32
                    ),
                    tf.constant(
                        [-4.0, -5.0, -3.0, -47.0, -8.0, -11.0], dtype=tf.float32
                    ),
                    tf.constant(
                        [-2.45, -11.0, -300.45, -40.2, -0.01, -10.0], dtype=tf.float32
                    ),
                ],
                "float",
                "double",
                "geq",
            ),
            (
                [
                    tf.constant([1.0, -2.0, -3.0], dtype=tf.float32),
                    tf.constant([1.0, -20.0, -3.0], dtype=tf.float32),
                    tf.constant([-1.0, -1.0, -1.0], dtype=tf.float32),
                    tf.constant([1.0, 1.0, 1.0], dtype=tf.float32),
                ],
                None,
                None,
                "neq",
            ),
            (
                [
                    tf.constant(["THIS", "IS", "A", "TEST"], dtype=tf.string),
                    tf.constant(["THIS", "IS", "A", "FAIL"], dtype=tf.string),
                    tf.constant(
                        ["TRUE", "TRUE x1", "TRUE x2", "TRUE x3"], dtype=tf.string
                    ),
                    tf.constant(
                        ["FALSE", "FALSE x1", "FALSE x2", "FALSE x3"], dtype=tf.string
                    ),
                ],
                None,
                None,
                "eq",
            ),
            (
                [
                    tf.constant([True, False, False, True, False, True]),
                    tf.constant([1, 0, 1, 1, 1, 1]),
                    tf.constant(
                        ["TRUE", "TRUE x1", "TRUE x2", "TRUE x3", "TRUE x4", "TRUE x5"],
                        dtype=tf.string,
                    ),
                    tf.constant(
                        [
                            1,
                            2,
                            9,
                            3,
                            6,
                            8,
                        ],
                    ),
                ],
                "string",
                None,
                "eq",
            ),
        ],
    )
    def test_if_statement_transform_multiple_input_spark_tf_parity(
        self,
        spark_session,
        input_tensors,
        input_dtype,
        output_dtype,
        condition_operator,
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = IfStatementTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            conditionOperator=condition_operator,
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
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensors).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
