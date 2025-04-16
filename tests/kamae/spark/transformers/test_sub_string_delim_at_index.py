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

from kamae.spark.transformers import SubStringDelimAtIndexTransformer


class TestSubStringDelimAtIndex:
    @pytest.fixture(scope="class")
    def example_dataframe_w_nested_string_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["a", "b?c", "c?d"],
                        ["d?e", "e?f", "f?g"],
                        ["g?h", "h?i", "i?j"],
                        ["j?k", "k?l?", "l?m?n?o"],
                    ],
                )
            ],
            ["col1"],
        )

    @pytest.fixture(scope="class")
    def example_dataframe_with_strings_to_split(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("hello world", "en_US", ["en_US", "en_US", "en_US"]),
                ("adventure time", "en_CA", ["en_CA", "ch_CH", "es_US"]),
                ("begins", "es_US", ["es_BR", "ch_CH", "es_US"]),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def sub_string_delim_transform_col1_indx_0_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("hello world", "en_US", ["en_US", "en_US", "en_US"], "hello"),
                ("adventure time", "en_CA", ["en_CA", "ch_CH", "es_US"], "adventure"),
                ("begins", "es_US", ["es_BR", "ch_CH", "es_US"], "begins"),
            ],
            ["col1", "col2", "col3", "sub_str_col1_idx_0"],
        )

    @pytest.fixture(scope="class")
    def sub_string_delim_transform_col1_indx_1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("hello world", "en_US", ["en_US", "en_US", "en_US"], "world"),
                ("adventure time", "en_CA", ["en_CA", "ch_CH", "es_US"], "time"),
                ("begins", "es_US", ["es_BR", "ch_CH", "es_US"], "NOT_FOUND"),
            ],
            ["col1", "col2", "col3", "sub_str_col1_idx_1"],
        )

    @pytest.fixture(scope="class")
    def sub_string_delim_transform_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("hello world", "en_US", ["en_US", "en_US", "en_US"], "US"),
                ("adventure time", "en_CA", ["en_CA", "ch_CH", "es_US"], "CA"),
                ("begins", "es_US", ["es_BR", "ch_CH", "es_US"], "US"),
            ],
            ["col1", "col2", "col3", "sub_str_col2"],
        )

    @pytest.fixture(scope="class")
    def sub_string_delim_transform_col3_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "hello world",
                    "en_US",
                    ["en_US", "en_US", "en_US"],
                    ["en", "en", "en"],
                ),
                (
                    "adventure time",
                    "en_CA",
                    ["en_CA", "ch_CH", "es_US"],
                    ["en", "ch", "es"],
                ),
                ("begins", "es_US", ["es_BR", "ch_CH", "es_US"], ["es", "ch", "es"]),
            ],
            ["col1", "col2", "col3", "sub_str_col3"],
        )

    @pytest.fixture(scope="class")
    def sub_string_delim_transform_col1_char_index_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("hello world", "en_US", ["en_US", "en_US", "en_US"], "l"),
                ("adventure time", "en_CA", ["en_CA", "ch_CH", "es_US"], "e"),
                ("begins", "es_US", ["es_BR", "ch_CH", "es_US"], "i"),
            ],
            ["col1", "col2", "col3", "char_at_index_col1"],
        )

    @pytest.fixture(scope="class")
    def sub_string_delim_at_index_nested_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["a", "b?c", "c?d"],
                        ["d?e", "e?f", "f?g"],
                        ["g?h", "h?i", "i?j"],
                        ["j?k", "k?l?", "l?m?n?o"],
                    ],
                    [
                        ["DEFAULT", "c", "d"],
                        ["e", "f", "g"],
                        ["h", "i", "j"],
                        ["k", "l", "m"],
                    ],
                )
            ],
            ["col1", "sub_string_delim_col1_at_1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, delimiter, index, default_value, expected_dataframe",
        [
            (
                "example_dataframe_with_strings_to_split",
                "col1",
                "sub_str_col1_idx_0",
                " ",
                0,
                "NOT_FOUND",
                "sub_string_delim_transform_col1_indx_0_expected",
            ),
            (
                "example_dataframe_with_strings_to_split",
                "col1",
                "sub_str_col1_idx_1",
                " ",
                1,
                "NOT_FOUND",
                "sub_string_delim_transform_col1_indx_1_expected",
            ),
            (
                "example_dataframe_with_strings_to_split",
                "col2",
                "sub_str_col2",
                "_",
                1,
                "NOT_FOUND",
                "sub_string_delim_transform_col2_expected",
            ),
            (
                "example_dataframe_with_strings_to_split",
                "col3",
                "sub_str_col3",
                "_",
                0,
                "NOT_FOUND",
                "sub_string_delim_transform_col3_expected",
            ),
            (
                "example_dataframe_with_strings_to_split",
                "col1",
                "char_at_index_col1",
                "",
                3,
                "NOT_FOUND",
                "sub_string_delim_transform_col1_char_index_expected",
            ),
            (
                "example_dataframe_w_nested_string_arrays",
                "col1",
                "sub_string_delim_col1_at_1",
                "?",
                1,
                "DEFAULT",
                "sub_string_delim_at_index_nested_arrays",
            ),
        ],
    )
    def test_spark_sub_string_delim_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        delimiter,
        index,
        default_value,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = SubStringDelimAtIndexTransformer(
            inputCol=input_col,
            outputCol=output_col,
            delimiter=delimiter,
            index=index,
            defaultValue=default_value,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_sub_string_delim_transform_defaults(self):
        # when
        sub_string_delim_transform = SubStringDelimAtIndexTransformer()
        # then
        assert (
            sub_string_delim_transform.getLayerName() == sub_string_delim_transform.uid
        )
        assert sub_string_delim_transform.getDelimiter() == "_"
        assert sub_string_delim_transform.getIndex() == 0
        assert sub_string_delim_transform.getDefaultValue() == ""
        assert (
            sub_string_delim_transform.getOutputCol()
            == f"{sub_string_delim_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, delimiter, index, default_value",
        [
            (
                tf.constant(["1 world", "2 time", "begins"]),
                None,
                "bigint",
                " ",
                0,
                "-999",
            ),
            (
                tf.constant(["hello world", "adventure time", "begins"]),
                None,
                None,
                " ",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["hello world", "adventure time", "begins"]),
                None,
                None,
                "",
                4,
                "NOT_FOUND",
            ),
            (
                tf.constant(["en_True", "en_False", "es_True"]),
                None,
                "boolean",
                "_",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["(1995", "(2005", "2006"]),
                None,
                None,
                "(",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["Question? I don't think so", "Another question?"]),
                None,
                None,
                "?",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["What about a \ backslash delimiter"]),
                None,
                None,
                "\\",
                -1,
                "NOT_FOUND",
            ),
            # Add more test cases for regex special characters
            (
                tf.constant(["String special^ character"]),
                None,
                None,
                "^",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String special$ character"]),
                None,
                None,
                "$",
                -1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String special* character"]),
                None,
                None,
                "*",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String special] character"]),
                None,
                None,
                "]",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String special{ character"]),
                None,
                None,
                "{",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String special| character"]),
                None,
                None,
                "|",
                0,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String special| character"]),
                None,
                None,
                "|",
                -1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String +with {multiple special| characters"]),
                None,
                None,
                "|",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String +with {multiple special| characters"]),
                None,
                None,
                "|",
                -1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String with trailing delimiter-"]),
                None,
                None,
                "-",
                1,
                "NOT_FOUND",
            ),
            (
                tf.constant(["_String with leading delimiter"]),
                None,
                None,
                "_",
                0,
                "NOT_FOUND",
            ),
            (
                tf.constant(["String with multiple __ delimiters"]),
                None,
                None,
                "_",
                -200,
                "NOT_FOUND",
            ),
        ],
    )
    def test_sub_string_delim_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        delimiter,
        index,
        default_value,
    ):
        # given
        transformer = SubStringDelimAtIndexTransformer(
            inputCol="input",
            outputCol="output",
            delimiter=delimiter,
            index=index,
            defaultValue=default_value,
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
