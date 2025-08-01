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

from kamae.spark.transformers import StringToStringListTransformer


class TestStringToStringList:
    @pytest.fixture(scope="class")
    def example_dataframe_w_nested_string_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["a", "b-c", "c-d"],
                        ["d-e", "e-f", "f-g"],
                        ["g-h", "h-i", "i-j"],
                        ["j-k", "k-l", "l-m-n-o"],
                    ],
                )
            ],
            ["col1"],
        )

    @pytest.fixture(scope="class")
    def example_dataframe_with_long_strings(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "Hello World|adventure time|let's go",
                    "1.067,-67.8,0.0,0.0",
                    "split^me^up",
                ),
                ("ADVENTURE Time|is upon|us", "0.0,0.0,0.0,0.0", "split^me^up^again"),
                (
                    "time|to|begin|again",
                    "-1.0,6.789,3.067,456.078",
                    "split^me^up^again^again",
                ),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def string_to_string_list_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "Hello World|adventure time|let's go",
                    "1.067,-67.8,0.0,0.0",
                    "split^me^up",
                    ["Hello World", "adventure time", "let's go"],
                ),
                (
                    "ADVENTURE Time|is upon|us",
                    "0.0,0.0,0.0,0.0",
                    "split^me^up^again",
                    ["ADVENTURE Time", "is upon", "us"],
                ),
                (
                    "time|to|begin|again",
                    "-1.0,6.789,3.067,456.078",
                    "split^me^up^again^again",
                    ["time", "to", "begin"],
                ),
            ],
            ["col1", "col2", "col3", "col1_string_list"],
        )

    @pytest.fixture(scope="class")
    def string_to_string_list_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "Hello World|adventure time|let's go",
                    "1.067,-67.8,0.0,0.0",
                    "split^me^up",
                    ["1.067", "-67.8", "0.0", "0.0"],
                ),
                (
                    "ADVENTURE Time|is upon|us",
                    "0.0,0.0,0.0,0.0",
                    "split^me^up^again",
                    ["0.0", "0.0", "0.0", "0.0"],
                ),
                (
                    "time|to|begin|again",
                    "-1.0,6.789,3.067,456.078",
                    "split^me^up^again^again",
                    ["-1.0", "6.789", "3.067", "456.078"],
                ),
            ],
            ["col1", "col2", "col3", "col2_string_list"],
        )

    @pytest.fixture(scope="class")
    def string_to_string_list_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "Hello World|adventure time|let's go",
                    "1.067,-67.8,0.0,0.0",
                    "split^me^up",
                    ["split", "me", "up", "DEFAULT"],
                ),
                (
                    "ADVENTURE Time|is upon|us",
                    "0.0,0.0,0.0,0.0",
                    "split^me^up^again",
                    ["split", "me", "up", "again"],
                ),
                (
                    "time|to|begin|again",
                    "-1.0,6.789,3.067,456.078",
                    "split^me^up^again^again",
                    ["split", "me", "up", "again"],
                ),
            ],
            ["col1", "col2", "col3", "col3_string_list"],
        )

    @pytest.fixture(scope="class")
    def string_to_string_list_nested_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["a", "b-c", "c-d"],
                        ["d-e", "e-f", "f-g"],
                        ["g-h", "h-i", "i-j"],
                        ["j-k", "k-l", "l-m-n-o"],
                    ],
                    [
                        [["a", "DEFAULT"], ["b", "c"], ["c", "d"]],
                        [["d", "e"], ["e", "f"], ["f", "g"]],
                        [["g", "h"], ["h", "i"], ["i", "j"]],
                        [["j", "k"], ["k", "l"], ["l", "m"]],
                    ],
                )
            ],
            ["col1", "string_to_string_list_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, separator, default_value, list_length, expected_dataframe",
        [
            (
                "example_dataframe_with_long_strings",
                "col1",
                "col1_string_list",
                "|",
                "DEFAULT",
                3,
                "string_to_string_list_transform_expected_1",
            ),
            (
                "example_dataframe_with_long_strings",
                "col2",
                "col2_string_list",
                ",",
                "DEFAULT",
                4,
                "string_to_string_list_transform_expected_2",
            ),
            (
                "example_dataframe_with_long_strings",
                "col3",
                "col3_string_list",
                "^",
                "DEFAULT",
                4,
                "string_to_string_list_transform_expected_3",
            ),
            (
                "example_dataframe_w_nested_string_arrays",
                "col1",
                "string_to_string_list_col1",
                "-",
                "DEFAULT",
                2,
                "string_to_string_list_nested_arrays",
            ),
        ],
    )
    def test_spark_string_to_string_list_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        separator,
        default_value,
        list_length,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = StringToStringListTransformer(
            inputCol=input_col,
            outputCol=output_col,
            separator=separator,
            defaultValue=default_value,
            listLength=list_length,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_string_to_string_list_transform_defaults(self):
        # when
        string_to_string_list_transform = StringToStringListTransformer()
        # then
        assert (
            string_to_string_list_transform.getLayerName()
            == string_to_string_list_transform.uid
        )
        assert string_to_string_list_transform.getSeparator() == ","
        assert string_to_string_list_transform.getDefaultValue() == ""
        assert string_to_string_list_transform.getListLength() == 1
        assert (
            string_to_string_list_transform.getOutputCol()
            == f"{string_to_string_list_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, separator, list_length, default_value",
        [
            (
                tf.constant(
                    [
                        "take,this,string,apart",
                        "and,this,one,too",
                        "and,this,one,too too",
                    ]
                ),
                None,
                "string",
                ",",
                4,
                "DEFAULT",
            ),
            (
                tf.constant(["0=>1", "2=>3"]),
                "string",
                "int",
                "=>",
                2,
                "DEFAULT",
            ),
            (
                tf.constant(["0=>1", "2=>3"]),
                "string",
                "float",
                "=>",
                2,
                "DEFAULT",
            ),
            (
                tf.constant(
                    [
                        "how|about|a|longer|set|of|strings",
                        "lets|see|how|this|goes|then",
                        "also|lets|add|another|string",
                    ],
                ),
                None,
                "string",
                "|",
                7,
                "DEFAULT",
            ),
        ],
    )
    def test_string_to_string_list_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        separator,
        list_length,
        default_value,
    ):
        # given
        transformer = StringToStringListTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            separator=separator,
            listLength=list_length,
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

        decoder = lambda x: x.decode("utf-8")
        vec_decoder = np.vectorize(decoder)
        tensorflow_values = [
            vec_decoder(v) if isinstance(v[0], bytes) else v
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
