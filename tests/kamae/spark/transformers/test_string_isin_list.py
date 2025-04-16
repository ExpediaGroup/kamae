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

from kamae.spark.transformers import StringIsInListTransformer


class TestStringIsInList:
    @pytest.fixture(scope="class")
    def string_isin_list_dataframe(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string",),
                ("String",),
                ("STRING",),
                ("",),
            ],
            [
                "col1",
            ],
        )

    @pytest.fixture(scope="class")
    def string_isin_list_expected_dataframe(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string", True),
                ("String", False),
                ("STRING", False),
                ("", True),
            ],
            ["col1", "string_in_col1"],
        )

    @pytest.fixture(scope="class")
    def string_isin_list_expected_dataframe_numeric(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string", 1.0),
                ("String", 0.0),
                ("STRING", 0.0),
                ("", 1.0),
            ],
            ["col1", "string_in_col1"],
        )

    @pytest.fixture(scope="class")
    def string_isin_list_expected_dataframe_inverted(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string", False),
                ("String", True),
                ("STRING", True),
                ("", False),
            ],
            ["col1", "string_in_col1_inverted"],
        )

    @pytest.fixture(scope="class")
    def string_isin_list_array_expected_dataframe(self, spark_session):
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
                        [True, True, True],
                        [False, True, False],
                        [False, False, False],
                        [True, False, True],
                    ],
                )
            ],
            ["col1", "col2", "col1_string_isin_list"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, input_dtype, output_dtpye, string_constant_list, output_col, negation, expected_dataframe",
        [
            (
                "string_isin_list_dataframe",
                "col1",
                "string",
                "boolean",
                ["string", ""],
                "string_in_col1",
                False,
                "string_isin_list_expected_dataframe",
            ),
            (
                "string_isin_list_dataframe",
                "col1",
                "string",
                "float",
                ["string", ""],
                "string_in_col1",
                False,
                "string_isin_list_expected_dataframe_numeric",
            ),
            (
                "string_isin_list_dataframe",
                "col1",
                "string",
                "boolean",
                ["string", ""],
                "string_in_col1_inverted",
                True,
                "string_isin_list_expected_dataframe_inverted",
            ),
            (
                "example_dataframe_w_multiple_string_nested_arrays",
                "col1",
                "string",
                "boolean",
                ["a", "b", "c", "e", "j", "l"],
                "col1_string_isin_list",
                False,
                "string_isin_list_array_expected_dataframe",
            ),
        ],
    )
    def test_string_isin_list_transform_layer(
        self,
        input_dataframe,
        input_col,
        input_dtype,
        output_dtpye,
        string_constant_list,
        output_col,
        negation,
        expected_dataframe,
        request,
    ):
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected_dataframe = request.getfixturevalue(expected_dataframe)

        layer = StringIsInListTransformer(
            inputCol=input_col,
            constantStringArray=string_constant_list,
            outputCol=output_col,
            inputDtype=input_dtype,
            outputDtype=output_dtpye,
            negation=negation,
        )

        assert layer.getNegation() == negation
        assert layer.getOutputCol() == output_col
        actual = layer.transform(input_dataframe)
        diff = actual.exceptAll(expected_dataframe)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_col, input_tensor, string_constant_list, negation",
        [
            (
                "col1",
                tf.constant(["string", "String", "STRING", "", "", ""]),
                ["string", ""],
                False,
            ),
            (
                "col1",
                tf.constant(["string", "String1", "STRING", "", "", ""]),
                ["string", ""],
                True,
            ),
        ],
    )
    def test_string_isin_list_spark_tf_parity(
        self, spark_session, input_col, input_tensor, string_constant_list, negation
    ):
        # given
        transformer = (
            StringIsInListTransformer()
            .setOutputCol("output_col")
            .setConstantStringArray(string_constant_list)
            .setNegation(negation)
            .setInputCol(input_col)
        )

        # when
        spark_df = spark_session.createDataFrame(
            [(t.decode("utf-8"),) for t in input_tensor.numpy().tolist()],
            [input_col],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output_col")
            .rdd.map(lambda x: x[0])
            .collect()
        )
        tensorflow_values = transformer.get_tf_layer()(input_tensor).numpy()

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
