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

from kamae.spark.transformers import StringReplaceTransformer

from ..test_helpers import tensor_to_python_type


class TestStringReplace:
    # Test for constant, non-regex match and constant replace
    @pytest.fixture(scope="class")
    def expected_df_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("abc", "", "", "zbc"),
                ("a", None, None, "z"),
                ("b", None, None, "b"),
                ("z", None, None, "z"),
                ("b", None, None, "b"),
                ("c", None, None, "c"),
                ("", None, None, ""),
            ],
            ["col1", "col2", "col3", "output"],
        )

    # Test for column-based non-regex match and replace, with special chars
    @pytest.fixture(scope="class")
    def expected_df_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("abc", "a", "x.", "x.bc"),
                ("a", "b", "y!", "a"),
                ("b", "b", "\\\\z", "\\z"),
                ("z", "b", "\\\\z", "z"),
                ("b", "b", "*z", "*z"),
                ("c", "d", "x*", "c"),
                ("", "", "", ""),
                ("a.b.c", "a.", "x", "xb.c"),
                ("a?", "a?", "y", "y"),
                ("b!b", "!b", "z", "bz"),
                ("\\z", "!b", "x", "\\z"),
                ("b!b", "!b", "z", "bz"),
                ("c*c", "c*", "x", "xc"),
                ("", "", "x", "x"),
                ("", ".*", "x", ""),
                ("", "^$", "x", ""),
                ("", ".*", "x", ""),
            ],
            ["col1", "col2", "col3", "output"],
        )

    # Test for column-based, regex match and replace, with special chars
    @pytest.fixture(scope="class")
    def expected_df_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a.b.c", "a.*c", "x.", "x."),
                ("a?", "a.", "y!", "y!"),
                ("b!b", "b.", "\\\\z", "\\zb"),
                ("\\z", "\\\\z", "\\\\z", "\\z"),
                ("b!b", "b.", "*z", "*zb"),
                ("c*c", "c.*c", "x*", "x*"),
                ("", ".*", "x", "x"),
                ("", "^$", "x", "x"),
                ("", ".*", "x", "x"),
            ],
            ["col1", "col2", "col3", "output"],
        )

    # Test for constant regex match and column replace
    @pytest.fixture(scope="class")
    def expected_df_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("abc", "", "", "bc"),
                ("a", None, "w", "w"),
                ("b", None, "e", "b"),
                ("z", None, "f", "z"),
                ("b", None, "c", "b"),
                ("c", None, "o", "c"),
                ("", None, "a", ""),
            ],
            ["col1", "col2", "col3", "output"],
        )

    # Test for column non-regex match and constant replace
    @pytest.fixture(scope="class")
    def expected_df_5(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("abc", "b", "", "axc"),
                ("a", "a", None, "x"),
                ("b", "b.", None, "b"),
                ("z", "\\\\z", None, "z"),
                ("b", "b.", None, "b"),
                ("c", "c.*c", None, "c"),
                ("", ".*", None, ""),
            ],
            ["col1", "col2", "col3", "output"],
        )

    @pytest.fixture(scope="class")
    def string_replace_col1_array_w_constant(self, spark_session):
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
                        ["REPLACE", "REPLACE", "REPLACE"],
                        ["d", "e", "f"],
                        ["g", "h", "i"],
                        ["j", "k", "l"],
                    ],
                )
            ],
            ["col1", "col2", "output"],
        )

    @pytest.mark.parametrize(
        "input_df, input_cols, string_match_constant, string_replace_constant, output_col, regex",
        [
            ("expected_df_1", ["col1"], "a", "z", "output", False),
            (
                "expected_df_2",
                ["col1", "col2", "col3"],
                None,
                None,
                "output",
                False,
            ),
            ("expected_df_3", ["col1", "col2", "col3"], None, None, "output", True),
            ("expected_df_4", ["col1", "col3"], "a", None, "output", False),
            ("expected_df_5", ["col1", "col2"], None, "x", "output", False),
            (
                "string_replace_col1_array_w_constant",
                ["col1"],
                "a|b|c|s|t|u",
                "REPLACE",
                "output",
                True,
            ),
        ],
    )
    def test_string_replace_transform_layer(
        self,
        input_df,
        input_cols,
        string_match_constant,
        string_replace_constant,
        output_col,
        regex,
        request,
    ):
        full_dataframe = request.getfixturevalue(input_df)
        input_dataframe = full_dataframe.drop("output")
        expected_dataframe = full_dataframe

        if len(input_cols) == 1:
            input_col = input_cols[0]
            layer = StringReplaceTransformer(
                inputCol=input_col,
                outputCol=output_col,
                stringMatchConstant=string_match_constant,
                stringReplaceConstant=string_replace_constant,
                regex=regex,
            )
        else:
            layer = StringReplaceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                regex=regex,
            )
            if string_match_constant is not None:
                layer = layer.setStringMatchConstant(string_match_constant)
            if string_replace_constant is not None:
                layer = layer.setStringReplaceConstant(string_replace_constant)

        assert layer.getRegex() == regex
        assert layer.getOutputCol() == output_col
        actual = layer.transform(input_dataframe)
        diff = actual.exceptAll(expected_dataframe)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, string_match_constant, string_replace_constant, regex",
        [
            (
                [
                    tf.constant(["hello", "there", "friend"]),
                    tf.constant(["hel", "th", "fri"]),
                    tf.constant(["llo", "ere", "end"]),
                ],
                None,
                None,
                None,
                None,
                False,
            ),
            (
                [
                    tf.constant([True, False, False]),
                    tf.constant(["T.e", "Fa.*", "Fa.*"]),
                    tf.constant(["llo", "ere", "end"]),
                ],
                "string",
                None,
                None,
                None,
                True,
            ),
            (
                [
                    tf.constant(["", "", ""]),
                    tf.constant(["^$", "", ".*"]),
                    tf.constant(["a", "b", "c"]),
                ],
                None,
                None,
                None,
                None,
                True,
            ),
            (
                [
                    tf.constant(["hello", "", "friend"]),
                    tf.constant(["hel", "", "fri"]),
                    tf.constant(["llo*", "x", "end"]),
                ],
                "string",
                None,
                None,
                None,
                False,
            ),
            # Constants set
            (
                [tf.constant([100, 200]), tf.constant([0, 0])],
                "string",
                "bigint",
                None,
                "",
                False,
            ),
            (
                [tf.constant(["hello", "friend"]), tf.constant(["hel", "fri"])],
                None,
                None,
                "o",
                None,
                False,
            ),
        ],
    )
    def test_string_replace_spark_tf_parity_no_constants(
        self,
        spark_session,
        input_tensors,
        input_dtype,
        output_dtype,
        string_match_constant,
        string_replace_constant,
        regex,
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = StringReplaceTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            regex=regex,
            stringMatchConstant=string_match_constant,
            stringReplaceConstant=string_replace_constant,
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
