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

from kamae.spark.estimators import StandardScaleEstimator
from kamae.spark.transformers import StandardScaleTransformer


class TestStandardScale:
    @pytest.fixture(scope="class")
    def standard_scale_expected_nested(self, spark_session):
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
                        [-0.3278688524590164, -0.8670520231213874, -2.8901734104046244],
                        [-0.3278688524590164, 0.28901734104046245, -2.8901734104046244],
                        [-0.3278688524590164, 0.28901734104046245, -6.358381502890174],
                        [0.6557377049180328, 0.28901734104046245, -8.092485549132949],
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
                        [0.6557377049180328, -0.8670520231213874, -1.1560693641618498],
                        [0.6557377049180328, -0.8670520231213874, -1.1560693641618498],
                        [0.6557377049180328, 0.28901734104046245, -8.092485549132949],
                        [1.639344262295082, 2.023121387283237, -2.8901734104046244],
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
                        [1.639344262295082, 2.023121387283237, -2.8901734104046244],
                        [1.639344262295082, -2.601156069364162, -2.8901734104046244],
                        [1.639344262295082, 2.023121387283237, -6.358381502890174],
                        [-0.9836065573770493, 0.28901734104046245, -6.358381502890174],
                    ],
                ),
            ],
            ["col1", "col2", "scaled_col1"],
        )

    @pytest.fixture(scope="class")
    def standard_scaler_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "c",
                    [1, 2, 3],
                    [-0.3278688524590164, 0.28901734104046245, -2.8901734104046244],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "c",
                    [4, 2, 6],
                    [0.6557377049180328, 0.28901734104046245, -1.1560693641618498],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "a",
                    [7, 8, 3],
                    [1.639344262295082, 2.023121387283237, -2.8901734104046244],
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
    def standard_scale_w_scalar_input_col1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "c",
                    [1.0, 2.0, 3.0],
                    -0.3278688524590164,
                ),
                (
                    4.0,
                    2.0,
                    6.0,
                    "b",
                    "c",
                    [4.0, 2.0, 6.0],
                    0.6557377049180328,
                ),
                (
                    7.0,
                    8.0,
                    3.0,
                    "a",
                    "a",
                    [7.0, 8.0, 3.0],
                    1.639344262295082,
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
    def standard_scale_w_scalar_input_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "c",
                    [1.0, 2.0, 3.0],
                    0.28901734104046245,
                ),
                (
                    4.0,
                    2.0,
                    6.0,
                    "b",
                    "c",
                    [4.0, 2.0, 6.0],
                    0.28901734104046245,
                ),
                (
                    7.0,
                    8.0,
                    3.0,
                    "a",
                    "a",
                    [7.0, 8.0, 3.0],
                    2.023121387283237,
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
    def standard_scaler_expected_with_masking(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    None,
                    2,
                    3,
                    [4, 2, 3, -1, -1],
                    [
                        -0.3076923076923076,
                        -1.0769230769230766,
                        -0.6923076923076922,
                        -1.0,
                        -1.0,
                    ],
                ),
                (
                    4,
                    None,
                    6,
                    [4, 3, -1, -1, -1],
                    [-0.3076923076923076, -0.6923076923076922, -1.0, -1.0, -1.0],
                ),
                (
                    7,
                    8,
                    None,
                    [7, -1, -1, -1, -1],
                    [0.8461538461538461, -1.0, -1.0, -1.0, -1.0],
                ),
                (
                    7,
                    8,
                    None,
                    [7, 8, 1, 9, 0],
                    [
                        0.8461538461538461,
                        1.2307692307692308,
                        -1.4615384615384612,
                        1.6153846153846154,
                        -1.8461538461538458,
                    ],
                ),
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, mean, stddev, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1_col2_col3",
                "scaled_features",
                [2.0, 1.0, 8.0],
                [3.05, 3.46, 1.73],
                "standard_scaler_expected",
            ),
            (
                "example_dataframe",
                "col1",
                "col1_scaled",
                [2.0],
                [3.05],
                "standard_scale_w_scalar_input_col1_expected",
            ),
            (
                "example_dataframe",
                "col2",
                "col2_scaled",
                [1.0],
                [3.46],
                "standard_scale_w_scalar_input_col2_expected",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "scaled_col1",
                [2.0, 1.0, 8.0],
                [3.05, 3.46, 1.73],
                "standard_scale_expected_nested",
            ),
        ],
    )
    def test_spark_standard_scaler_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        mean,
        stddev,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        standard_scaler_model = StandardScaleTransformer(
            inputCol=input_col,
            outputCol=output_col,
            mean=mean,
            stddev=stddev,
        )
        actual = standard_scaler_model.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_col, output_col, mean, stddev, expected_dataframe",
        [
            (
                "col4",
                "scaled_features",
                [4.8, 4.8, 4.8, 4.8, 4.8],
                [2.6, 2.6, 2.6, 2.6, 2.6],
                "standard_scaler_expected_with_masking",
            ),
        ],
    )
    def test_spark_standard_scaler_transform_with_padding(
        self,
        example_dataframe_with_padding,
        input_col,
        output_col,
        mean,
        stddev,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        standard_scaler_model = StandardScaleTransformer(
            inputCol=input_col,
            outputCol=output_col,
            mean=mean,
            stddev=stddev,
            maskValue=-1,
        )
        actual = standard_scaler_model.transform(example_dataframe_with_padding)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_standard_scaler_model_defaults(self):
        # when
        standard_scaler_model = StandardScaleTransformer(
            mean=[1.0, 2.0, 3.0], stddev=[4.0, 5.0, 6.0]
        )
        # then
        assert standard_scaler_model.getLayerName() == standard_scaler_model.uid
        assert (
            standard_scaler_model.getOutputCol()
            == f"{standard_scaler_model.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, mean, stddev",
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
    def test_standard_scaler_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, mean, stddev
    ):
        # given
        transformer = StandardScaleTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            mean=mean,
            stddev=stddev,
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

    @pytest.mark.parametrize(
        "input_col, output_col",
        [
            (
                "col4",
                "scaled_features",
            ),
        ],
    )
    def test_spark_standard_scaler_raises_error_when_mean_contains_none(
        self,
        example_dataframe_with_padding_2,
        input_col,
        output_col,
    ):
        # when
        standard_scaler = StandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
            maskValue=-1,
        )
        with pytest.raises(ValueError):
            _ = standard_scaler.fit(example_dataframe_with_padding_2)
            tb = traceback.format_exc()
            assert "Mean values cannot be None" in tb
