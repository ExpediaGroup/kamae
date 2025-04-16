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
from pyspark.sql.types import DoubleType

from kamae.spark.transformers import BinTransformer


class TestBin:
    @pytest.fixture(scope="class")
    def bin_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], "less_than_equal_to_2"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], "default"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], "greater_than_6"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "bin_col1",
            ],
        )

    @pytest.fixture(scope="class")
    def bin_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], "equal_to_2"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], "equal_to_2"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], "not_equal_to_10"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "bin_col2",
            ],
        )

    @pytest.fixture(scope="class")
    def bin_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], 1.0),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], -1.0),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], 1.0),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "bin_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def bin_transform_array_expected_3(self, spark_session):
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
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                    ],
                    [
                        ["less_than_3", "less_than_3", "equal_to_3"],
                        ["less_than_3", "less_than_3", "equal_to_3"],
                        ["less_than_3", "less_than_3", "less_than_3"],
                        ["default", "less_than_3", "less_than_3"],
                    ],
                ),
                (
                    [
                        [4.0, -2.0, 6.0],
                        [4.0, -2.0, 6.0],
                        [4.0, 2.0, -6.0],
                        [7.0, 8.0, 3.0],
                    ],
                    [
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                    ],
                    [
                        ["default", "less_than_3", "default"],
                        ["default", "less_than_3", "default"],
                        ["default", "less_than_3", "less_than_3"],
                        ["greater_than_6", "greater_than_6", "equal_to_3"],
                    ],
                ),
                (
                    [
                        [7.0, 8.0, 3.0],
                        [7.0, -8.0, 3.0],
                        [7.0, 8.0, -3.0],
                        [-1.0, 2.0, -3.0],
                    ],
                    [
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                    ],
                    [
                        ["greater_than_6", "greater_than_6", "equal_to_3"],
                        ["greater_than_6", "less_than_3", "equal_to_3"],
                        ["greater_than_6", "greater_than_6", "less_than_3"],
                        ["less_than_3", "less_than_3", "less_than_3"],
                    ],
                ),
            ],
            ["col1", "col2", "bin_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, condition_operators, bin_values, bin_labels, default_label, output_col, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1",
                ["leq", "gt"],
                [2, 6],
                ["less_than_equal_to_2", "greater_than_6"],
                "default",
                "bin_col1",
                "bin_transform_expected_1",
            ),
            (
                "example_dataframe",
                "col2",
                ["eq", "neq"],
                [2, 10],
                ["equal_to_2", "not_equal_to_10"],
                "default",
                "bin_col2",
                "bin_transform_expected_2",
            ),
            (
                "example_dataframe",
                "col3",
                ["lt", "eq", "gt"],
                [3, 3, 6],
                [0.0, 1.0, 2.0],
                -1.0,
                "bin_col3",
                "bin_transform_expected_3",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                ["lt", "eq", "gt"],
                [3, 3, 6],
                ["less_than_3", "equal_to_3", "greater_than_6"],
                "default",
                "bin_col1",
                "bin_transform_array_expected_3",
            ),
        ],
    )
    def test_spark_bin_transform(
        self,
        input_dataframe,
        input_col,
        condition_operators,
        bin_values,
        bin_labels,
        default_label,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = BinTransformer(
            conditionOperators=condition_operators,
            outputCol=output_col,
            inputCol=input_col,
            binValues=bin_values,
            binLabels=bin_labels,
            defaultLabel=default_label,
        )

        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_bin_transform_defaults(self):
        # when
        bin_transformer = BinTransformer()
        # then
        assert bin_transformer.getLayerName() == bin_transformer.uid
        assert bin_transformer.getOutputCol() == f"{bin_transformer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, condition_operators, bin_values, bin_labels, default_label",
        [
            (
                tf.constant([1, 7, 14, 31, 256, 45, 3, 3, 1, 8], dtype=tf.float32),
                None,
                None,
                ["leq", "leq", "leq", "leq", "leq", "leq", "leq", "leq", "gt"],
                [1, 3, 5, 7, 10, 14, 25, 30, 50],
                [
                    "less_than_equal_to_1",
                    "less_than_equal_to_3",
                    "less_than_equal_to_5",
                    "less_than_equal_to_7",
                    "less_than_equal_to_10",
                    "less_than_equal_to_14",
                    "less_than_equal_to_25",
                    "less_than_equal_to_30",
                    "greater_than_50",
                ],
                "default",
            ),
            (
                tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float32),
                "bigint",
                "string",
                ["eq", "neq"],
                [2, 10],
                ["equal_to_2", "not_equal_to_10"],
                "default",
            ),
            (
                tf.constant(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]),
                "double",
                None,
                ["lt", "eq", "gt"],
                [3, 3, 6],
                ["less_than_3", "equal_to_3", "greater_than_6"],
                "default",
            ),
            (
                tf.constant(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]),
                "double",
                None,
                ["lt", "eq", "gt"],
                [3, 3, 6],
                [0, 1, 2],
                -1,
            ),
        ],
    )
    def test_bin_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        condition_operators,
        bin_values,
        bin_labels,
        default_label,
    ):
        # given
        transformer = BinTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            conditionOperators=condition_operators,
            binValues=bin_values,
            binLabels=bin_labels,
            defaultLabel=default_label,
        )
        if input_tensor.dtype.name == "string":
            spark_df = spark_session.createDataFrame(
                [(v.decode("utf-8"),) for v in input_tensor.numpy().tolist()], ["input"]
            )
        else:
            spark_df = spark_session.createDataFrame(
                [(v,) for v in input_tensor.numpy().tolist()], ["input"]
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
