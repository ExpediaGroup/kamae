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

from kamae.spark.transformers import RoundToDecimalTransformer


class TestRoundToDecimal:
    @pytest.fixture(scope="class")
    def example_dataframe_w_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [
                            [1.5464, -2.90890, 3.78865],
                            [1.546, 2.768789, 3.6574],
                            [1.2332, 2.67868, -3.43534],
                            [4.879, 2.454, -6.2424],
                        ],
                        [
                            [4.546, 5.234, -1.4531],
                            [41.351485, -89.45, 56.5485],
                            [14.654, -6.06354, 9.56546],
                            [43.45541, -2.654156, 4.568456],
                        ],
                    ],
                )
            ],
            ["col1"],
        )

    @pytest.fixture(scope="class")
    def example_dataframe_with_floats(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.53465, 2.345, 3.67678),
                (4.243242, 2.234324234, 6.43546),
                (7.7978, 8.547, 3.24234),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def round_to_decimal_transform_col1_decimals_2_ceil_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.53465, 2.345, 3.67678, 1.53),
                (4.243242, 2.234324234, 6.43546, 4.24),
                (7.7978, 8.547, 3.24234, 7.80),
            ],
            ["col1", "col2", "col3", "round_to_decimal_col1"],
        )

    @pytest.fixture(scope="class")
    def round_to_decimal_transform_col2_decimals_3_type_floor_expected(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (1.53465, 2.345, 3.67678, 2.345),
                (4.243242, 2.234324234, 6.43546, 2.234),
                (7.7978, 8.547, 3.24234, 8.547),
            ],
            ["col1", "col2", "col3", "round_to_decimal_col2"],
        )

    @pytest.fixture(scope="class")
    def round_to_decimal_transform_col3_decimals_5_round_to_decimal_expected(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (1.53465, 2.345, 3.67678, 3.67678),
                (4.243242, 2.234324234, 6.43546, 6.43546),
                (7.7978, 8.547, 3.24234, 3.24234),
            ],
            ["col1", "col2", "col3", "round_to_decimal_col3"],
        )

    @pytest.fixture(scope="class")
    def round_to_decimal_transform_array(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [
                            [1.5464, -2.90890, 3.78865],
                            [1.546, 2.768789, 3.6574],
                            [1.2332, 2.67868, -3.43534],
                            [4.879, 2.454, -6.2424],
                        ],
                        [
                            [4.546, 5.234, -1.4531],
                            [41.351485, -89.45, 56.5485],
                            [14.654, -6.06354, 9.56546],
                            [43.45541, -2.654156, 4.568456],
                        ],
                    ],
                    [
                        [[2, -3, 4], [2, 3, 4], [1, 3, -3], [5, 2, -6]],
                        [
                            [5, 5, -1],
                            [41, -89, 57],
                            [15, -6, 10],
                            [43, -3, 5],
                        ],
                    ],
                )
            ],
            ["col1", "round_to_decimal_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, decimals, expected_dataframe",
        [
            (
                "example_dataframe_with_floats",
                "col1",
                "round_to_decimal_col1",
                2,
                "round_to_decimal_transform_col1_decimals_2_ceil_expected",
            ),
            (
                "example_dataframe_with_floats",
                "col2",
                "round_to_decimal_col2",
                3,
                "round_to_decimal_transform_col2_decimals_3_type_floor_expected",
            ),
            (
                "example_dataframe_with_floats",
                "col3",
                "round_to_decimal_col3",
                5,
                "round_to_decimal_transform_col3_decimals_5_round_to_decimal_expected",
            ),
            (
                "example_dataframe_w_arrays",
                "col1",
                "round_to_decimal_col1",
                0,
                "round_to_decimal_transform_array",
            ),
        ],
    )
    def test_spark_round_to_decimal_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        decimals,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = RoundToDecimalTransformer(
            inputCol=input_col,
            outputCol=output_col,
            decimals=decimals,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_round_to_decimal_transform_defaults(self):
        # when
        round_to_decimal_transform = RoundToDecimalTransformer()
        # then
        assert (
            round_to_decimal_transform.getLayerName() == round_to_decimal_transform.uid
        )
        assert (
            round_to_decimal_transform.getOutputCol()
            == f"{round_to_decimal_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, decimals",
        [
            (tf.constant([1.345345, 4.768768, 7.2324, 8.56567]), "double", "float", 2),
            (tf.constant([2.2345, 5.6787567, 1.23424]), "float", "bigint", 0),
            (tf.constant([-1.90567, 7.12313]), None, None, 1),
            (tf.constant([0.3245, 6.45657, 3.2344]), "float", "double", 4),
            (tf.constant([2.574568, 5.63464, 1.56758, 5.3453, 2.7899]), None, None, 3),
        ],
    )
    def test_round_to_decimal_transform_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, decimals
    ):
        # given
        transformer = RoundToDecimalTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            decimals=decimals,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [(v,) for v in input_tensor.numpy().tolist()], ["input"]
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = transformer.get_tf_layer()(input_tensor).numpy().tolist()

        # then
        np.testing.assert_almost_equal(
            spark_values,
            tensorflow_values,
            decimal=6,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
