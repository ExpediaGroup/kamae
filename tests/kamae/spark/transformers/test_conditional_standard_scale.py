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

from kamae.spark.estimators import ConditionalStandardScaleEstimator
from kamae.spark.transformers import ConditionalStandardScaleTransformer


class TestConditionalStandardScale:
    @pytest.fixture(scope="class")
    def cond_standard_scale_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "c",
                    [1.0, 2.0, 3.0],
                    [-0.3278688524590164, 0.28901734104046245, -2.8901734104046244],
                ),
                (
                    4.0,
                    2.0,
                    6.0,
                    "b",
                    "c",
                    [4.0, 2.0, 6.0],
                    [0.6557377049180328, 0.28901734104046245, -1.1560693641618498],
                ),
                (
                    7.0,
                    8.0,
                    3.0,
                    "a",
                    "a",
                    [7.0, 8.0, 3.0],
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
    def cond_standard_scale_w_scalar_input_col1_expected(self, spark_session):
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
    def cond_standard_scale_w_scalar_input_col2_expected(self, spark_session):
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
    def cond_standard_scale_expected_all_zeros(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "b",
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                ),
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "b",
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                ),
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "b",
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
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
    def cond_standard_scale_expected_nested(self, spark_session):
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

    @pytest.mark.parametrize(
        "example_dataframe_name, input_col, output_col, mean, stddev, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1_col2_col3",
                "scaled_features",
                [2.0, 1.0, 8.0],
                [3.05, 3.46, 1.73],
                "cond_standard_scale_expected",
            ),
            (
                "example_dataframe",
                "col1",
                "col1_scaled",
                [2.0],
                [3.05],
                "cond_standard_scale_w_scalar_input_col1_expected",
            ),
            (
                "example_dataframe",
                "col2",
                "col2_scaled",
                [1.0],
                [3.46],
                "cond_standard_scale_w_scalar_input_col2_expected",
            ),
            (
                "example_dataframe_equal_rows",
                "col1_col2_col3",
                "scaled_features",
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                "cond_standard_scale_expected_all_zeros",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "scaled_col1",
                [2.0, 1.0, 8.0],
                [3.05, 3.46, 1.73],
                "cond_standard_scale_expected_nested",
            ),
        ],
    )
    def test_spark_cond_standard_scaler_transform(
        self,
        example_dataframe_name,
        input_col,
        output_col,
        mean,
        stddev,
        expected_dataframe,
        request,
    ):
        # given
        df = request.getfixturevalue(example_dataframe_name)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        standard_scaler_model = ConditionalStandardScaleTransformer(
            inputCol=input_col,
            outputCol=output_col,
            mean=mean,
            stddev=stddev,
        )
        actual = standard_scaler_model.transform(df)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, mean, stddev, skip_zeros, epsilon",
        [
            (
                tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]),
                None,
                None,
                [3.0, 10.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 3.0, 4.0],
                False,
                0.0,
            ),
            (
                tf.constant(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 13.0, 10.0],
                        [-1.0, 51.0, 12.0, 4.0, 1.0],
                    ]
                ),
                None,
                "string",
                [3.0, 10.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 3.0, 4.0],
                False,
                0.0,
            ),
            (
                tf.constant([[-1.0, -2.0, 3.0, 5.0], [6.0, -7.0, -9.0, 10.0]]),
                "double",
                None,
                [3.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 4.0],
                False,
                0.0,
            ),
            (
                tf.constant([[-1.0, -2.0, 3.0, 5.0], [6.0, -7.0, -9.0, 10.0]]),
                "float",
                "double",
                [3.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 4.0],
                False,
                0.0,
            ),
            (
                tf.constant([[1.0, 2.0, 0.0], [6.0, 10.0, 0.0]]),
                None,
                "string",
                [-1.0, 4.0, 10.0],
                [2.0, 4.0, 10.0],
                False,
                0.0,
            ),
            (
                tf.constant([[1.0, 2.0, 0.0], [6.0, 10.0, 0.0]]),
                None,
                None,
                [-1.0, 4.0, 10.0],
                [2.0, 4.0, 10.0],
                True,
                0.0,
            ),
            (
                tf.constant([[1.0, 2.0, 0.00001], [6.0, 10.0, 0.00001]]),
                None,
                None,
                [-1.0, 4.0, 10.0],
                [2.0, 4.0, 10.0],
                True,
                0.0001,
            ),
        ],
    )
    def test_cond_standard_scaler_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        skip_zeros,
        epsilon,
        mean,
        stddev,
    ):
        # given
        transformer = ConditionalStandardScaleTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            mean=mean,
            stddev=stddev,
            skipZeros=skip_zeros,
            epsilon=epsilon,
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
        if isinstance(spark_values[0][0], str):
            np.testing.assert_equal(
                spark_values,
                tensorflow_values,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
        else:
            np.testing.assert_almost_equal(
                spark_values,
                tensorflow_values,
                decimal=5,
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
    def test_spark_cond_standard_scaler_raises_error_when_no_data(
        self,
        example_dataframe_with_padding_2,
        input_col,
        output_col,
    ):
        # when
        standard_scaler = ConditionalStandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
            maskCols=["col1"],
            maskOperators=["gt"],
            maskValues=[1000],
        )
        with pytest.raises(ValueError):
            _ = standard_scaler.fit(example_dataframe_with_padding_2)
            tb = traceback.format_exc()
            assert "No data left after application of mask conditions." in tb
