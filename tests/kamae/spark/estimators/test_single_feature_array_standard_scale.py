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

from kamae.spark.estimators import SingleFeatureArrayStandardScaleEstimator
from kamae.spark.transformers import StandardScaleTransformer


class TestSingleFeatureArrayStandardScale:
    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, expected_mean, expected_stddev",
        [
            (
                "example_dataframe",
                "col1_col2_col3",
                "scaled_features",
                [4.0, 4.0, 4.0],
                [2.3094010767585, 2.3094010767585, 2.3094010767585],
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "scaled_features",
                [2, 2, 2],
                [4.163332, 4.163332, 4.163332],
            ),
        ],
    )
    def test_spark_standard_scaler_fit_with_flat_axis(
        self,
        input_dataframe,
        input_col,
        output_col,
        expected_mean,
        expected_stddev,
        request,
    ):
        # when
        input_dataframe = request.getfixturevalue(input_dataframe)
        standard_scaler = SingleFeatureArrayStandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = standard_scaler.fit(input_dataframe)
        # then
        actual_mean, actual_stddev = actual.getMean(), actual.getStddev()
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(
            np.array(actual_stddev), np.array(expected_stddev)
        )
        assert isinstance(actual, StandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    @pytest.mark.parametrize(
        "input_col, output_col, expected_mean, expected_stddev",
        [
            (
                "col4",
                "scaled_features",
                [4.3636364, 4.3636364, 4.3636364, 4.3636364, 4.3636364],
                [2.8371794, 2.8371794, 2.8371794, 2.8371794, 2.8371794],
            ),
        ],
    )
    def test_spark_standard_scaler_fit_with_masking(
        self,
        example_dataframe_with_padding,
        input_col,
        output_col,
        expected_mean,
        expected_stddev,
    ):
        # when
        standard_scaler = SingleFeatureArrayStandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
            maskValue=-1,
        )
        actual = standard_scaler.fit(example_dataframe_with_padding)
        # then
        actual_mean, actual_stddev = actual.getMean(), actual.getStddev()
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(
            np.array(actual_stddev), np.array(expected_stddev)
        )
        assert isinstance(actual, StandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    @pytest.mark.parametrize(
        "input_col, output_col, expected_mean, expected_stddev",
        [
            (
                "col1_col2_col3",
                "scaled_features",
                [5.625, 5.625, 5.625],
                [2.1758619, 2.1758619, 2.1758619],
            ),
        ],
    )
    def test_spark_standard_scaler_fit_with_nulls(
        self,
        example_dataframe_with_nulls,
        input_col,
        output_col,
        expected_mean,
        expected_stddev,
    ):
        # when
        standard_scaler = SingleFeatureArrayStandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = standard_scaler.fit(example_dataframe_with_nulls)
        # then
        actual_mean, actual_stddev = actual.getMean(), actual.getStddev()
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(
            np.array(actual_stddev), np.array(expected_stddev)
        )
        assert isinstance(actual, StandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    def test_standard_scaler_defaults(self):
        # when
        standard_scaler = SingleFeatureArrayStandardScaleEstimator()
        # then
        assert standard_scaler.getLayerName() == standard_scaler.uid
        assert standard_scaler.getOutputCol() == f"{standard_scaler.uid}__output"
