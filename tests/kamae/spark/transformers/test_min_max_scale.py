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

from kamae.spark.transformers import MinMaxScaleTransformer


class TestMinMaxScale:
    @pytest.fixture(scope="class")
    def min_max_scale_expected_nested(self, spark_session):
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
                    # Take the columns and do the following transformation: (x - min) / (max - min)
                    # Where min = [-1.0, -8.0, -6.0] and max = [7.0, 8.0, 6.0]
                    # scale = [1.0, 8.0, 6.0] and shift = [8.0, 16.0, 12.0]
                    [
                        [0.25, 0.375, 0.75],
                        [0.25, 0.625, 0.75],
                        [0.25, 0.625, 0.25],
                        [0.625, 0.625, 0.0],
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
                        [0.625, 0.375, 1.0],
                        [0.625, 0.375, 1.0],
                        [0.625, 0.625, 0.0],
                        [1.0, 1.0, 0.75],
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
                        [1.0, 1.0, 0.75],
                        [1.0, 0.0, 0.75],
                        [1.0, 1.0, 0.25],
                        [0.0, 0.625, 0.25],
                    ],
                ),
            ],
            ["col1", "col2", "scaled_col1"],
        )

    @pytest.fixture(scope="class")
    def min_max_scaler_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "c",
                    [1, 2, 3],
                    [0.0, 0.0, 0.0],
                ),
                (4, 2, 6, "b", "c", [4, 2, 6], [0.5, 0.0, 1.0]),
                (
                    7,
                    8,
                    3,
                    "a",
                    "a",
                    [7, 8, 3],
                    [1.0, 1.0, 0.0],
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "scaled_features",
            ],
        )

    @pytest.fixture(scope="class")
    def min_max_scale_w_scalar_input_col1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "c",
                    [1.0, 2.0, 3.0],
                    0.0,
                ),
                (
                    4.0,
                    2.0,
                    6.0,
                    "b",
                    "c",
                    [4.0, 2.0, 6.0],
                    0.5,
                ),
                (
                    7.0,
                    8.0,
                    3.0,
                    "a",
                    "a",
                    [7.0, 8.0, 3.0],
                    1.0,
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col1_scaled",
            ],
        )

    @pytest.fixture(scope="class")
    def min_max_scale_w_scalar_input_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "c",
                    [1.0, 2.0, 3.0],
                    0.0,
                ),
                (
                    4.0,
                    2.0,
                    6.0,
                    "b",
                    "c",
                    [4.0, 2.0, 6.0],
                    0.0,
                ),
                (
                    7.0,
                    8.0,
                    3.0,
                    "a",
                    "a",
                    [7.0, 8.0, 3.0],
                    1.0,
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col2_scaled",
            ],
        )

    @pytest.fixture(scope="class")
    def min_max_scaler_expected_with_masking(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    None,
                    2,
                    3,
                    [4, 2, 3, -1, -1],
                    [
                        0.5714285714285714,
                        0.6000000000000001,
                        1.0,
                        -1.0,
                        -1.0,
                    ],
                ),
                (
                    4,
                    None,
                    6,
                    [4, 3, -1, -1, -1],
                    [0.5714285714285714, 0.65, -1.0, -1.0, -1.0],
                ),
                (
                    7,
                    8,
                    None,
                    [7, -1, -1, -1, -1],
                    [1.0, -1.0, -1.0, -1.0, -1.0],
                ),
                (
                    7,
                    8,
                    None,
                    [7, 8, 1, 9, 0],
                    [
                        1.0,
                        0.9,
                        0.6000000000000001,
                        0.0,
                        0.0,
                    ],
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "scaled_features",
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, min, max, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1_col2_col3",
                "scaled_features",
                [1.0, 2.0, 3.0],
                [7.0, 8.0, 6.0],
                "min_max_scaler_expected",
            ),
            (
                "example_dataframe",
                "col1",
                "col1_scaled",
                [1.0],
                [7.0],
                "min_max_scale_w_scalar_input_col1_expected",
            ),
            (
                "example_dataframe",
                "col2",
                "col2_scaled",
                [2.0],
                [8.0],
                "min_max_scale_w_scalar_input_col2_expected",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "scaled_col1",
                [-1.0, -8.0, -6.0],
                [7.0, 8.0, 6.0],
                "min_max_scale_expected_nested",
            ),
        ],
    )
    def test_spark_min_max_scaler_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        min,
        max,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        min_max_scaler_model = MinMaxScaleTransformer(
            inputCol=input_col,
            outputCol=output_col,
            min=min,
            max=max,
        )
        actual = min_max_scaler_model.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_col, output_col, min, max, expected_dataframe",
        [
            (
                "col4",
                "scaled_features",
                [0.0, -10.0, -2.0, 9.0, 0.0],
                [7.0, 10.0, 3.0, 15.0, 10.0],
                "min_max_scaler_expected_with_masking",
            ),
        ],
    )
    def test_spark_min_max_scaler_transform_with_masking(
        self,
        example_dataframe_with_padding,
        input_col,
        output_col,
        min,
        max,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        min_max_scaler_model = MinMaxScaleTransformer(
            inputCol=input_col,
            outputCol=output_col,
            min=min,
            max=max,
            maskValue=-1,
        )
        actual = min_max_scaler_model.transform(example_dataframe_with_padding)
        actual.show(20, False)
        expected.show(20, False)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_min_max_scaler_model_defaults(self):
        # when
        min_max_scaler_model = MinMaxScaleTransformer(
            min=[1.0, 2.0, 3.0], max=[4.0, 5.0, 6.0]
        )
        # then
        assert min_max_scaler_model.getLayerName() == min_max_scaler_model.uid
        assert (
            min_max_scaler_model.getOutputCol() == f"{min_max_scaler_model.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, min, max",
        [
            (
                tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]),
                None,
                None,
                [3.0, 10.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 3.0, 4.0],
            ),
            (
                tf.constant(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0],
                        [-1.0, 51.0, 12.89, 0.0, 1.0],
                    ]
                ),
                "double",
                "float",
                [3.0, 10.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 3.0, 4.0],
            ),
            (
                tf.constant([[-1.0, -2.0, 3.0, 5.0], [6.0, -7.0, -9.0, 10.0]]),
                "float",
                None,
                [3.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 4.0],
            ),
            (
                tf.constant([[1.0, 2.0], [6.0, 10.0]]),
                "double",
                "double",
                [-1.0, 4.0],
                [2.0, 4.0],
            ),
        ],
    )
    def test_min_max_scaler_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, min, max
    ):
        # given
        transformer = MinMaxScaleTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            min=min,
            max=max,
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
