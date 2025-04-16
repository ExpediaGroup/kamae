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

from kamae.spark.transformers import StringContainsListTransformer


# TODO: rename and repurpose
class TestStringContainsList:
    @pytest.fixture(scope="class")
    def string_contains_list_base_0(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string]",),
                ("thing",),
                ("bling",),
                ("nonempty",),
                ("spe.cial*cha|rs",),
                ("bar|bar",),
                ("aaaa-bbbb-cccc-dddd",),
            ],
            [
                "col1",
            ],
        )

    @pytest.fixture(scope="class")
    def string_contains_list_expected_0(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string]", False),
                ("thing", False),
                ("bling", False),
                ("nonempty", False),
                ("spe.cial*cha|rs", True),
                ("bar|bar", True),
                ("aaaa-bbbb-cccc-dddd", False),
            ],
            ["col1", "special_in_col1"],
        )

    @pytest.fixture(scope="class")
    def string_contains_list_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("string]", True),
                ("thing", True),
                ("bling", True),
                ("nonempty", False),
                ("spe.cial*cha|rs", True),
                ("bar|bar", False),
                ("aaaa-bbbb-cccc-dddd", True),
            ],
            ["col1", "list_in_col1"],
        )

    @pytest.fixture(scope="class")
    def string_contains_list_array_expected_dataframe(self, spark_session):
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
            ["col1", "col2", "col1_string_contains_list"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, string_constant_list, output_col, negation, expected_dataframe",
        [
            (
                "string_contains_list_base_0",
                "col1",
                [".", "*", "|"],
                "special_in_col1",
                False,
                "string_contains_list_expected_0",
            ),
            (
                "string_contains_list_base_0",
                "col1",
                ["in", "|rs", "aaaa-bbbb-cccc-dddd"],
                "list_in_col1",
                False,
                "string_contains_list_expected_1",
            ),
            (
                "example_dataframe_w_multiple_string_nested_arrays",
                "col1",
                ["a", "b", "c", "e", "j", "l"],
                "col1_string_contains_list",
                False,
                "string_contains_list_array_expected_dataframe",
            ),
        ],
    )
    def test_string_contains_list_transform_layer(
        self,
        input_dataframe,
        input_col,
        string_constant_list,
        output_col,
        negation,
        expected_dataframe,
        request,
    ):
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected_dataframe = request.getfixturevalue(expected_dataframe)

        layer = StringContainsListTransformer(
            inputCol=input_col,
            constantStringArray=string_constant_list,
            outputCol=output_col,
            negation=negation,
        )

        assert layer.getNegation() == negation
        assert layer.getOutputCol() == output_col
        actual = layer.transform(input_dataframe)
        diff = actual.exceptAll(expected_dataframe)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, string_constant_list, negation",
        [
            (
                tf.constant(["hello", "there", "friend", "whacky*special|chars"]),
                None,
                "string",
                [".", "*", "|"],
                False,
            ),
            (
                tf.constant(["aaa", "bbbb-", "-dddd", "eeee", "bbbb-cccc-dddd", "\\"]),
                None,
                None,
                ["aaaa", "bbbb-cccc", "dddd", "\\"],
                False,
            ),
            (
                tf.constant([True, False]),
                "string",
                "string",
                ["T", "e"],
                False,
            ),
        ],
    )
    def test_string_contains_list_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        string_constant_list,
        negation,
    ):
        # given
        transformer = StringContainsListTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            constantStringArray=string_constant_list,
            negation=negation,
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
            .rdd.map(lambda x: x[0])
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
