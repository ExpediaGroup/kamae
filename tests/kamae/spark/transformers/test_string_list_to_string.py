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

from kamae.spark.transformers import StringListToStringTransformer


class TestStringListToString:
    @pytest.fixture(scope="class")
    def example_dataframe_with_list_strings(self, spark_session):
        return spark_session.createDataFrame(
            [
                (["Hello World", "adventure time", "let's go"], 1),
                (["ADVENTURE Time", "is upon", "us"], 2),
                (["time", "to", "begin"], 3),
            ],
            # Had to add another column otherwise the array gets split out into
            # separate columns
            ["col1", "index"],
        )

    @pytest.fixture(scope="class")
    def string_list_to_string_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    ["Hello World", "adventure time", "let's go"],
                    1,
                    "Hello World adventure time let's go",
                ),
                (["ADVENTURE Time", "is upon", "us"], 2, "ADVENTURE Time is upon us"),
                (["time", "to", "begin"], 3, "time to begin"),
            ],
            ["col1", "index", "col1_string"],
        )

    @pytest.fixture(scope="class")
    def string_list_to_string_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    ["Hello World", "adventure time", "let's go"],
                    1,
                    "Hello World=>adventure time=>let's go",
                ),
                (["ADVENTURE Time", "is upon", "us"], 2, "ADVENTURE Time=>is upon=>us"),
                (["time", "to", "begin"], 3, "time=>to=>begin"),
            ],
            ["col1", "index", "col1_string"],
        )

    @pytest.fixture(scope="class")
    def string_list_to_string_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    ["Hello World", "adventure time", "let's go"],
                    1,
                    "Hello WorldSEPARATORadventure timeSEPARATORlet's go",
                ),
                (
                    ["ADVENTURE Time", "is upon", "us"],
                    2,
                    "ADVENTURE TimeSEPARATORis uponSEPARATORus",
                ),
                (["time", "to", "begin"], 3, "timeSEPARATORtoSEPARATORbegin"),
            ],
            ["col1", "index", "col1_string"],
        )

    @pytest.fixture(scope="class")
    def string_list_to_string_array_col1(self, spark_session):
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
                        "a-b-c",
                        "d-e-f",
                        "g-h-i",
                        "j-k-l",
                    ],
                )
            ],
            ["col1", "col2", "string_list_as_string_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, separator, expected_dataframe",
        [
            (
                "example_dataframe_with_list_strings",
                "col1",
                "col1_string",
                " ",
                "string_list_to_string_transform_expected_1",
            ),
            (
                "example_dataframe_with_list_strings",
                "col1",
                "col1_string",
                "=>",
                "string_list_to_string_transform_expected_2",
            ),
            (
                "example_dataframe_with_list_strings",
                "col1",
                "col1_string",
                "SEPARATOR",
                "string_list_to_string_transform_expected_3",
            ),
            (
                "example_dataframe_w_multiple_string_nested_arrays",
                "col1",
                "string_list_as_string_col1",
                "-",
                "string_list_to_string_array_col1",
            ),
        ],
    )
    def test_spark_string_list_to_string_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        separator,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = StringListToStringTransformer(
            inputCol=input_col,
            outputCol=output_col,
            separator=separator,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_string_list_to_string_transform_defaults(self):
        # when
        string_list_to_string_transform = StringListToStringTransformer()
        # then
        assert (
            string_list_to_string_transform.getLayerName()
            == string_list_to_string_transform.uid
        )
        assert string_list_to_string_transform.getSeparator() == ""
        assert (
            string_list_to_string_transform.getOutputCol()
            == f"{string_list_to_string_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, separator",
        [
            (
                tf.constant(
                    [
                        ["put", "this", "string", "together"],
                        ["and", "this", "one", "too"],
                        ["and", "this", "one", "too too"],
                    ]
                ),
                "string",
                None,
                " ",
            ),
            (
                tf.constant([[True, False], [False, True]]),
                "string",
                "string",
                "=>",
            ),
            (
                tf.constant(
                    [
                        ["how", "about", "a", "longer", "set", "of", "strings"],
                        ["lets", "see", "how", "this", "goes", "then", "alright"],
                        ["also", "lets", "add", "another", "row", "of", "strings"],
                        ["and", "why not", "another", "one", "for", "good", "measure"],
                    ]
                ),
                None,
                None,
                "SEPARATOR",
            ),
        ],
    )
    def test_string_list_to_string_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, separator
    ):
        # given
        transformer = StringListToStringTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            separator=separator,
        )
        vec_decoder = np.vectorize(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )
        # when
        spark_df = spark_session.createDataFrame(
            [(vec_decoder(v).tolist(),) for v in input_tensor.numpy().tolist()],
            ["input"],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        tensorflow_values = vec_decoder(
            transformer.get_tf_layer()(input_tensor).numpy().flatten()
        ).tolist()

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
