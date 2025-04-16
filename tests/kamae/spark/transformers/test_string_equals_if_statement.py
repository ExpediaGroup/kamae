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
from pyspark.sql.types import StringType

from kamae.spark.transformers import StringEqualsIfStatementTransformer

from ..test_helpers import tensor_to_python_type


class TestStringEqualsIfStatement:
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

    @pytest.mark.parametrize(
        "input_dataframe, input_col, input_cols, value_to_compare, result_if_true, result_if_false, output_col, expected_dataframe",
        [
            (
                "example_dataframe_with_three_string_columns",
                "col1",
                None,
                "a",
                "TRUE",
                "FALSE",
                "string_if_statement_col1_w_constant",
                "string_if_statement_transform_expected_1",
            ),
            (
                "example_dataframe_with_three_string_columns",
                None,
                ["col1", "col2", "col3"],
                None,
                "TRUE",
                None,
                "string_if_statement_col1_col2_col3",
                "string_if_statement_transform_expected_2",
            ),
            (
                "example_dataframe_with_three_string_columns",
                None,
                ["col3", "col1"],
                "a",
                None,
                "FALSE",
                "string_if_statement_col1_col3",
                "string_if_statement_transform_expected_3",
            ),
            (
                "example_dataframe_with_three_string_columns",
                "col2",
                None,
                "c",
                "TRUE",
                "FALSE",
                "string_if_statement_col2_w_constant",
                "string_if_statement_transform_expected_4",
            ),
            (
                "example_dataframe_w_multiple_string_nested_arrays",
                "col1",
                None,
                "a",
                "TRUE",
                "FALSE",
                "string_if_statement_col1_array_w_constant",
                "string_if_statement_transform_array_expected",
            ),
        ],
    )
    def test_spark_string_if_statement_transform(
        self,
        input_dataframe,
        input_col,
        input_cols,
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
        transformer = StringEqualsIfStatementTransformer(
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

    def test_string_if_statement_transform_defaults(self):
        # when
        string_if_statement_transformer = StringEqualsIfStatementTransformer()
        # then
        assert (
            string_if_statement_transformer.getLayerName()
            == string_if_statement_transformer.uid
        )
        assert (
            string_if_statement_transformer.getOutputCol()
            == f"{string_if_statement_transformer.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, value_to_compare, result_if_true, result_if_false",
        [
            (tf.constant(["a", "b", "c"]), None, "string", "a", "TRUE", "FALSE"),
            (tf.constant([1, 2, 3]), "string", None, "2", "TRUE", "FALSE"),
            (
                tf.constant([True, False, False]),
                "string",
                "boolean",
                "False",
                "TRUE",
                "FALSE",
            ),
        ],
    )
    def test_string_if_statement_transform_single_input_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        value_to_compare,
        result_if_true,
        result_if_false,
    ):
        # given
        transformer = StringEqualsIfStatementTransformer(
            inputCol="input",
            outputCol="output",
            valueToCompare=value_to_compare,
            resultIfTrue=result_if_true,
            resultIfFalse=result_if_false,
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
        "input_tensors, input_dtype, output_dtype",
        [
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
            ),
        ],
    )
    def test_string_if_statement_transform_multiple_input_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = StringEqualsIfStatementTransformer(
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
