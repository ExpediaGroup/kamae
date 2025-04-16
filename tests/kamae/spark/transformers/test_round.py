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

from kamae.spark.transformers import RoundTransformer


class TestRound:
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
    def round_transform_col1_round_type_ceil_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.53465, 2.345, 3.67678, 2),
                (4.243242, 2.234324234, 6.43546, 5),
                (7.7978, 8.547, 3.24234, 8),
            ],
            ["col1", "col2", "col3", "round_col1"],
        )

    @pytest.fixture(scope="class")
    def round_transform_col2_round_type_floor_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.53465, 2.345, 3.67678, 2),
                (4.243242, 2.234324234, 6.43546, 2),
                (7.7978, 8.547, 3.24234, 8),
            ],
            ["col1", "col2", "col3", "round_col2"],
        )

    @pytest.fixture(scope="class")
    def round_transform_col3_round_type_round_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.53465, 2.345, 3.67678, 4),
                (4.243242, 2.234324234, 6.43546, 6),
                (7.7978, 8.547, 3.24234, 3),
            ],
            ["col1", "col2", "col3", "round_col3"],
        )

    @pytest.fixture(scope="class")
    def round_transform_array(self, spark_session):
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
            ["col1", "round_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, round_type, expected_dataframe",
        [
            (
                "example_dataframe_with_floats",
                "col1",
                "round_col1",
                "ceil",
                "round_transform_col1_round_type_ceil_expected",
            ),
            (
                "example_dataframe_with_floats",
                "col2",
                "round_col2",
                "floor",
                "round_transform_col2_round_type_floor_expected",
            ),
            (
                "example_dataframe_with_floats",
                "col3",
                "round_col3",
                "round",
                "round_transform_col3_round_type_round_expected",
            ),
            (
                "example_dataframe_w_arrays",
                "col1",
                "round_col1",
                "round",
                "round_transform_array",
            ),
        ],
    )
    def test_spark_round_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        round_type,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = RoundTransformer(
            inputCol=input_col,
            outputCol=output_col,
            roundType=round_type,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_round_transform_defaults(self):
        # when
        round_transform = RoundTransformer()
        # then
        assert round_transform.getLayerName() == round_transform.uid
        assert round_transform.getOutputCol() == f"{round_transform.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, round_type",
        [
            (tf.constant([1.345345, 4.768768, 7.2324, 8.56567]), None, None, "ceil"),
            (tf.constant([2.2345, 5.6787567, 1.23424]), "float", "double", "floor"),
            (tf.constant([-1.90567, 7.12313]), "double", "bigint", "round"),
            (tf.constant([0.3245, 6.45657, 3.2344]), None, "smallint", "round"),
            (
                tf.constant([2.574568, 5.63464, 1.56758, 5.3453, 2.7899]),
                "double",
                None,
                "ceil",
            ),
        ],
    )
    def test_round_transform_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, round_type
    ):
        # given
        transformer = RoundTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            roundType=round_type,
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
