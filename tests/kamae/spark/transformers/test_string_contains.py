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

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
import numpy as np
import pytest
import tensorflow as tf

from kamae.spark.transformers import StringContainsTransformer

from ..test_helpers import tensor_to_python_type


class TestStringContains:
    @pytest.fixture(scope="class")
    def example_dataframe_w_arrays_contains(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [["string]", "ing]", "str.ng"]],
                        [["thing", "ng", "th.*"]],
                        [["bling", "bi", "bl..g"]],
                        [["", "", ".*"]],
                        [["nonempty", "", ".*"]],
                    ],
                    [
                        [["bling", "bi", "bl..g"]],
                        [["", "", ".*"]],
                        [["nonempty", "", ".*"]],
                        [["string]", "ing]", "str.ng"]],
                        [["thing", "ng", "th.*"]],
                    ],
                )
            ],
            ["col1", "col2"],
        )

    @pytest.fixture(scope="class")
    def string_contains_base_0(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string]", "ing]", "str.ng"),
                ("thing", "ng", "th.*"),
                ("bling", "bi", "bl..g"),
                ("", "", ".*"),
                ("nonempty", "", ".*"),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def string_contains_expected_0(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string]", "ing]", "str.ng", True),
                ("thing", "ng", "th.*", True),
                ("bling", "bi", "bl..g", False),
                ("", "", ".*", True),
                ("nonempty", "", ".*", False),
            ],
            ["col1", "col2", "col3", "col2_in_col1"],
        )

    @pytest.fixture(scope="class")
    def string_contains_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string]", "ing]", "str.ng", True),
                ("thing", "ng", "th.*", True),
                ("bling", "bi", "bl..g", True),
                ("", "", ".*", False),
                ("nonempty", "", ".*", False),
            ],
            ["col1", "col2", "col3", "in_in_col1"],
        )

    @pytest.fixture(scope="class")
    def string_contains_array_w_constant_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [["string]", "ing]", "str.ng"]],
                        [["thing", "ng", "th.*"]],
                        [["bling", "bi", "bl..g"]],
                        [["", "", ".*"]],
                        [["nonempty", "", ".*"]],
                    ],
                    [
                        [["bling", "bi", "bl..g"]],
                        [["", "", ".*"]],
                        [["nonempty", "", ".*"]],
                        [["string]", "ing]", "str.ng"]],
                        [["thing", "ng", "th.*"]],
                    ],
                    [
                        [[True, True, False]],
                        [[True, False, False]],
                        [[True, False, False]],
                        [[False, False, False]],
                        [[False, False, False]],
                    ],
                )
            ],
            ["col1", "col2", "col1_string_array_contains_constant"],
        )

    @pytest.fixture(scope="class")
    def string_contains_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [["string]", "ing]", "str.ng"]],
                        [["thing", "ng", "th.*"]],
                        [["bling", "bi", "bl..g"]],
                        [["", "", ".*"]],
                        [["nonempty", "", ".*"]],
                    ],
                    [
                        [["bling", "bi", "bl..g"]],
                        [["", "", ".*"]],
                        [["nonempty", "", ".*"]],
                        [["string]", "ing]", "str.ng"]],
                        [["thing", "ng", "th.*"]],
                    ],
                    [
                        [[False, False, False]],
                        [[False, False, False]],
                        [[False, False, False]],
                        [[False, False, False]],
                        [[False, False, True]],
                    ],
                )
            ],
            ["col1", "col2", "col2_in_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, input_cols, string_constant, output_col, negation, expected_dataframe",
        [
            (
                "string_contains_base_0",
                None,
                ["col1", "col2"],
                None,
                "col2_in_col1",
                False,
                "string_contains_expected_0",
            ),
            (
                "string_contains_base_0",
                "col1",
                None,
                "in",
                "in_in_col1",
                False,
                "string_contains_expected_1",
            ),
            (
                "example_dataframe_w_arrays_contains",
                "col1",
                None,
                "in",
                "col1_string_array_contains_constant",
                False,
                "string_contains_array_w_constant_expected",
            ),
            (
                "example_dataframe_w_arrays_contains",
                None,
                ["col2", "col1"],
                None,
                "col2_in_col1",
                False,
                "string_contains_array_expected",
            ),
        ],
    )
    def test_string_contains_transform_layer(
        self,
        input_dataframe,
        input_col,
        input_cols,
        string_constant,
        output_col,
        negation,
        expected_dataframe,
        request,
    ):
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected_dataframe = request.getfixturevalue(expected_dataframe)

        if string_constant is None:
            layer = StringContainsTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                negation=negation,
            )
        else:
            layer = StringContainsTransformer(
                inputCol=input_col,
                stringConstant=string_constant,
                outputCol=output_col,
                negation=negation,
            )

        assert layer.getNegation() == negation
        assert layer.getOutputCol() == output_col
        actual = layer.transform(input_dataframe)
        diff = actual.exceptAll(expected_dataframe)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, string_constant, negation",
        [
            (
                [
                    tf.constant(["hello", "there", "friend"]),
                    tf.constant(["hel", "th", "fri"]),
                ],
                None,
                None,
                None,
                False,
            ),
            (
                [tf.constant(["hello1", "", "friend2"]), tf.constant([1, 0, 2])],
                "string",
                None,
                None,
                False,
            ),
            (
                [tf.constant([True, False, False])],
                "string",
                "string",
                "u",
                False,
            ),
        ],
    )
    def test_string_contains_spark_tf_parity(
        self,
        spark_session,
        input_tensors,
        input_dtype,
        output_dtype,
        string_constant,
        negation,
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = StringContainsTransformer(
            outputCol="output",
            stringConstant=string_constant,
            negation=negation,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        if len(col_names) == 1:
            transformer = transformer.setInputCol(col_names[0])
        else:
            transformer = transformer.setInputCols(col_names)

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
            .rdd.map(lambda x: x[0])
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
