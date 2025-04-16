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
import pandas as pd
import pytest
import tensorflow as tf

from kamae.sklearn.estimators import StandardScaleEstimator


class TestStandardScale:
    @pytest.fixture(scope="class")
    def standard_scaler_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "scaled_features": [
                    [-0.3278688524590164, 0.2886751345948129, -2.886751345948129],
                    [0.6557377049180328, 0.2886751345948129, -1.1547005383792517],
                    [1.639344262295082, 2.0207259421636903, -2.886751345948129],
                ],
            }
        )

    @pytest.mark.parametrize(
        "input_col, output_col, expected_mean, expected_var",
        [
            (
                "col1_col2_col3",
                "scaled_features",
                [4.0, 4.0, 4.0],
                [6.0, 8.0, 2.0],
            ),
        ],
    )
    def test_sklearn_standard_scaler_fit(
        self,
        example_dataframe,
        input_col,
        output_col,
        expected_mean,
        expected_var,
    ):
        # when
        standard_scaler = StandardScaleEstimator(
            input_col=input_col,
            output_col=output_col,
            layer_name="standard_scaler",
        )
        actual = standard_scaler.fit(example_dataframe)
        # then
        actual_mean, actual_var = actual.mean_, actual.var_
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(np.array(actual_var), np.array(expected_var))

    @pytest.mark.parametrize(
        "input_col, output_col, expected_mean, expected_var",
        [
            (
                "col1_col2_col3",
                "scaled_features",
                [6.0, 6.0, 4.5],
                [2.0, 8.0, 2.25],
            ),
        ],
    )
    def test_sklearn_standard_scaler_fit_with_nulls(
        self,
        example_dataframe_with_nulls,
        input_col,
        output_col,
        expected_mean,
        expected_var,
    ):
        # when
        standard_scaler = StandardScaleEstimator(
            input_col=input_col,
            output_col=output_col,
            layer_name="standard_scaler",
        )
        actual = standard_scaler.fit(example_dataframe_with_nulls)
        # then
        actual_mean, actual_var = actual.mean_, actual.var_
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(np.array(actual_var), np.array(expected_var))

    @pytest.mark.parametrize(
        "input_col, output_col, mean, var, expected_dataframe",
        [
            (
                "col1_col2_col3",
                "scaled_features",
                [2.0, 1.0, 8.0],
                [9.3025, 12.0, 3.0],
                "standard_scaler_expected",
            ),
        ],
    )
    def test_sklearn_standard_scaler_transform(
        self,
        example_dataframe,
        input_col,
        output_col,
        mean,
        var,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        standard_scaler_model = StandardScaleEstimator(
            input_col=input_col,
            output_col=output_col,
            layer_name="standard_scaler",
        )
        standard_scaler_model.mean_ = mean
        standard_scaler_model.var_ = var
        standard_scaler_model.scale_ = np.sqrt(var)
        actual = standard_scaler_model.transform(example_dataframe)
        # then
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "input_tensor, mean, stddev",
        [
            (
                tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]),
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
                [3.0, 10.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 3.0, 4.0],
            ),
            (
                tf.constant([[-1.0, -2.0, 3.0, 5.0], [6.0, -7.0, -9.0, 10.0]]),
                [3.0, -1.0, 4.0, 2.0],
                [2.0, 2.0, 1.0, 4.0],
            ),
            (
                tf.constant([[1.0, 2.0], [6.0, 10.0]]),
                [-1.0, 4.0],
                [2.0, 4.0],
            ),
        ],
    )
    def test_standard_scaler_spark_tf_parity(self, input_tensor, mean, stddev):
        # given
        transformer = StandardScaleEstimator(
            input_col="input",
            output_col="output",
            layer_name="standard_scaler",
        )
        transformer.mean_ = mean
        transformer.var_ = np.power(stddev, 2)
        transformer.scale_ = stddev

        # when
        pd_df = pd.DataFrame(
            {
                "input": input_tensor.numpy().tolist(),
            }
        )
        pd_values = transformer.transform(pd_df)["output"].values.tolist()
        tensorflow_values = transformer.get_tf_layer()(input_tensor).numpy().tolist()

        # then
        np.testing.assert_almost_equal(
            pd_values,
            tensorflow_values,
            decimal=6,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
