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

import traceback

import numpy as np
import pytest
import tensorflow as tf

from kamae.spark.transformers.ordinal_array_encode import OrdinalArrayEncodeTransformer


class TestOrdinalArrayEncoder:
    @pytest.fixture(scope="class")
    def ordinal_array_encoder_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    ["-1", "a", "b", "-1"],
                    [-1, 0, 1, -1],
                ),
                (
                    4,
                    ["a", "a", "b", "c"],
                    [0, 0, 1, 2],
                ),
                (
                    7,
                    ["b", "b", "b", "a"],
                    [0, 0, 0, 1],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def ordinal_array_encoder_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    ["-1", "a", "b", "-1"],
                    [0, 1, 2, 0],
                ),
                (
                    4,
                    ["a", "a", "b", "c"],
                    [0, 0, 1, 2],
                ),
                (
                    7,
                    ["b", "b", "b", "a"],
                    [0, 0, 0, 1],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def ordinal_array_encoder_array_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, [["-1", "a", "b", "-1"]], [[-1, 0, 1, -1]]),
                (4, [["a", "a", "b", "c"]], [[0, 0, 1, 2]]),
                (7, [["b", "b", "b", "a"]], [[0, 0, 0, 1]]),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def ordinal_array_encoder_array_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, [["-1", "a", "b", "-1"]], [[0, 1, 2, 0]]),
                (4, [["a", "a", "b", "c"]], [[0, 0, 1, 2]]),
                (7, [["b", "b", "b", "a"]], [[0, 0, 0, 1]]),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, input_dtype, output_dtype, pad_value, expected_dataframe",
        [
            (
                "example_dataframe_with_string_array",
                "col2",
                "col2_diff",
                "string",
                "double",
                "-1",
                "ordinal_array_encoder_expected_1",
            ),
            (
                "example_dataframe_with_string_array",
                "col2",
                "col2_diff",
                "string",
                "double",
                None,
                "ordinal_array_encoder_expected_2",
            ),
            (
                "example_dataframe_with_nested_string_array",
                "col2",
                "col2_diff",
                "string",
                "double",
                "-1",
                "ordinal_array_encoder_array_expected_1",
            ),
            (
                "example_dataframe_with_nested_string_array",
                "col2",
                "col2_diff",
                "string",
                "double",
                None,
                "ordinal_array_encoder_array_expected_2",
            ),
        ],
    )
    def test_spark_ordinal_array_encoder_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        input_dtype,
        output_dtype,
        pad_value,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        ordinal_array_encoder = OrdinalArrayEncodeTransformer(
            inputCol=input_col,
            outputCol=output_col,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            padValue=pad_value,
        )
        actual = ordinal_array_encoder.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)

        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensor,pad_value",
        [
            (
                tf.constant([["-1", "a", "b", "-1"]]),
                "-1",
            ),
            (
                tf.constant([["a", "a", "b", "-1"]]),
                None,
            ),
            (
                tf.constant([["a", "a", "b", "-1"], ["a", "a", "b", "c"]]),
                "-1",
            ),
            (
                tf.constant([["a", "a", "a"], ["b", "b", "b"]]),
                "-1",
            ),
        ],
    )
    def test_ordinal_encoding_spark_tf_parity(
        self, spark_session, input_tensor, pad_value
    ):
        # given
        transformer = (
            OrdinalArrayEncodeTransformer()
            .setInputCol("input")
            .setOutputCol("output")
            .setInputDtype("string")
            .setOutputDtype("double")
            .setPadValue(pad_value)
        )

        spark_df = spark_session.createDataFrame(
            [
                ([v.decode("utf-8") for v in input_row],)
                for input_row in input_tensor.numpy().tolist()
            ],
            ["input"],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = transformer.get_tf_layer()(input_tensor).numpy().tolist()

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
