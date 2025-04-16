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

from kamae.spark.transformers import ArraySubtractMinimumTransformer


class TestArraySubtractMinimum:
    @pytest.fixture(scope="class")
    def array_subtract_minimum_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    [3.0, 2.0, 1.0, -1.0],
                    [2.0, 1.0, 0.0, -1.0],
                ),
                (
                    4,
                    [100.0, 6.0, 4.0, -1.0],
                    [96.0, 2.0, 0.0, -1.0],
                ),
                (
                    7,
                    [12.0, 8.0, -1.0, -1.0],
                    [4.0, 0.0, -1.0, -1.0],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def array_subtract_minimum_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    [3.0, 2.0, 1.0, -1.0],
                    [4.0, 3.0, 2.0, 0.0],
                ),
                (
                    4,
                    [100.0, 6.0, 4.0, -1.0],
                    [101.0, 7.0, 5.0, 0.0],
                ),
                (
                    7,
                    [12.0, 8.0, -1.0, -1.0],
                    [13.0, 9.0, 0.0, 0.0],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def array_subtract_minimum_nested_expected_1(self, spark_session):
        example_df = spark_session.createDataFrame(
            [
                (
                    [
                        [
                            [100.0, 98.0, 2.0, 5.0, -1.0],
                            [1000.0, 67.0, 84.0, -1.0, -1.0],
                            [1000.0, 67.0, 84.0, -1.0, -1.0],
                        ]
                    ],
                    [[3.0, 2.0, 1.0, -1.0], [3.0, 2.0, 1.0, -1.0]],
                    [
                        [
                            [98.0, 96.0, 0.0, 3.0, -1.0],
                            [933.0, 0.0, 17.0, -1.0, -1.0],
                            [933.0, 0.0, 17.0, -1.0, -1.0],
                        ]
                    ],
                ),
                (
                    [
                        [
                            [167.0, 9.0, 2.0, 5.0, -1.0],
                            [10.0, 6.0, 8.0, -1.0, -1.0],
                            [100.0, 7.0, 4.0, -1.0, -1.0],
                        ]
                    ],
                    [[100.0, 6.0, 4.0, -1.0], [100.0, 6.0, 4.0, -1.0]],
                    [
                        [
                            [165.0, 7.0, 0.0, 3.0, -1.0],
                            [4.0, 0.0, 2.0, -1.0, -1.0],
                            [96.0, 3.0, 0.0, -1.0, -1.0],
                        ]
                    ],
                ),
            ],
            ["col1", "col2", "col1_diff"],
        )
        return example_df

    @pytest.fixture(scope="class")
    def array_subtract_minimum_nested_expected_2(self, spark_session):
        example_df = spark_session.createDataFrame(
            [
                (
                    [
                        [
                            [100.0, 98.0, 2.0, 5.0, -1.0],
                            [1000.0, 67.0, 84.0, -1.0, -1.0],
                            [1000.0, 67.0, 84.0, -1.0, -1.0],
                        ]
                    ],
                    [[3.0, 2.0, 1.0, -1.0], [3.0, 2.0, 1.0, -1.0]],
                    [[2.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]],
                ),
                (
                    [
                        [
                            [167.0, 9.0, 2.0, 5.0, -1.0],
                            [10.0, 6.0, 8.0, -1.0, -1.0],
                            [100.0, 7.0, 4.0, -1.0, -1.0],
                        ]
                    ],
                    [[100.0, 6.0, 4.0, -1.0], [100.0, 6.0, 4.0, -1.0]],
                    [[96.0, 2.0, 0.0, -1.0], [96.0, 2.0, 0.0, -1.0]],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )
        return example_df

    @pytest.mark.parametrize(
        "input_dataframe, input_col,output_col,pad_value,expected_dataframe",
        [
            (
                "example_dataframe_with_padding_no_nulls",
                "col2",
                "col2_diff",
                -1.0,
                "array_subtract_minimum_expected_1",
            ),
            (
                "example_dataframe_with_padding_no_nulls",
                "col2",
                "col2_diff",
                None,
                "array_subtract_minimum_expected_2",
            ),
            (
                "example_dataframe_with_padding_no_nulls",
                "col2",
                "col2_diff",
                0.0,
                "array_subtract_minimum_expected_2",
            ),
            (
                "example_dataframe_with_nested_arrays_padding_no_nulls",
                "col1",
                "col1_diff",
                -1.0,
                "array_subtract_minimum_nested_expected_1",
            ),
            (
                "example_dataframe_with_nested_arrays_padding_no_nulls",
                "col2",
                "col2_diff",
                -1.0,
                "array_subtract_minimum_nested_expected_2",
            ),
        ],
    )
    def test_spark_array_subtract_minimum_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        pad_value,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        array_subtract_minimum_model = ArraySubtractMinimumTransformer(
            inputCol=input_col,
            outputCol=output_col,
            padValue=pad_value,
        )
        actual = array_subtract_minimum_model.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)

        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, pad_value",
        [
            (
                tf.constant([[11.0, 5.0, 2.0, -1.0], [11.0, 5.0, 2.0, -1.0]]),
                None,
                None,
                -1.0,
            ),
            (
                tf.constant([[11002.0, 5.0, 2.0, -1.0], [11.0, 5.0, -1.0, -1.0]]),
                "bigint",
                "float",
                -1.0,
            ),
            (
                tf.constant([[11002.0, 5.0, 2.0, -1.0], [11.0, 5.0, 2.0, -1.0]]),
                "double",
                "string",
                None,
            ),
            (
                tf.constant(
                    [["11002.0", "5.0", "2.0", "0.0"], ["11.0", "5.0", "2.0", "-1.0"]]
                ),
                "float",
                None,
                0.0,
            ),
            (
                tf.constant([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                "smallint",
                "float",
                0.0,
            ),
        ],
    )
    def test_array_subtract_minimum_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, pad_value
    ):
        # given
        transformer = ArraySubtractMinimumTransformer(
            inputCol="input",
            outputCol="output",
            padValue=pad_value,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        array_decoder = np.vectorize(lambda x: x.decode("utf-8"))

        if input_tensor.dtype.name == "string":
            spark_df = spark_session.createDataFrame(
                [(array_decoder(v).tolist(),) for v in input_tensor.numpy().tolist()],
                ["input"],
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
            array_decoder(v) if isinstance(v[0], bytes) else v
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
