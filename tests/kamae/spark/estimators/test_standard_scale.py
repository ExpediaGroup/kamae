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

from kamae.spark.estimators import StandardScaleEstimator
from kamae.spark.transformers import StandardScaleTransformer


class TestStandardScale:
    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, expected_mean, expected_variance",
        [
            (
                "example_dataframe",
                "col1_col2_col3",
                "scaled_features",
                [4.0, 4.0, 4.0],
                [6.0, 8.0, 2.0],
            ),
            (
                "example_dataframe",
                "col1",
                "scaled_features",
                [4.0],
                [6.0],
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "scaled_features",
                [3.8333333, 1.6666667, 0.5],
                [7.3055556, 21.2222222, 17.75],
            ),
        ],
    )
    def test_spark_standard_scaler_fit(
        self,
        input_dataframe,
        input_col,
        output_col,
        expected_mean,
        expected_variance,
        request,
    ):
        # when
        input_dataframe = request.getfixturevalue(input_dataframe)
        standard_scaler = StandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = standard_scaler.fit(input_dataframe)
        # then
        actual_mean, actual_variance = actual.getMean(), actual.getVariance()
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(
            np.array(actual_variance), np.array(expected_variance)
        )
        assert isinstance(actual, StandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    @pytest.mark.parametrize(
        "input_col, output_col, expected_mean, expected_variance",
        [
            (
                "col4",
                "scaled_features",
                [5.5, 4.333333333333333, 2.0, 9.0, 0.0],
                [2.25, 6.8888889, 1.0, 0.0, 0.0],
            ),
        ],
    )
    def test_spark_standard_scaler_fit_with_masking(
        self,
        example_dataframe_with_padding,
        input_col,
        output_col,
        expected_mean,
        expected_variance,
    ):
        # when
        standard_scaler = StandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
            maskValue=-1,
        )
        actual = standard_scaler.fit(example_dataframe_with_padding)
        # then
        actual_mean, actual_variance = actual.getMean(), actual.getVariance()
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(
            np.array(actual_variance), np.array(expected_variance)
        )
        assert isinstance(actual, StandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    @pytest.mark.parametrize(
        "input_col, output_col, expected_mean, expected_variance",
        [
            (
                "col1_col2_col3",
                "scaled_features",
                [6.0, 6.0, 4.5],
                [2.0, 8.0, 2.25],
            ),
        ],
    )
    def test_spark_standard_scaler_fit_with_nulls(
        self,
        example_dataframe_with_nulls,
        input_col,
        output_col,
        expected_mean,
        expected_variance,
    ):
        # when
        standard_scaler = StandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = standard_scaler.fit(example_dataframe_with_nulls)
        # then
        actual_mean, actual_variance = actual.getMean(), actual.getVariance()
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(
            np.array(actual_variance), np.array(expected_variance)
        )
        assert isinstance(actual, StandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    def test_standard_scaler_defaults(self):
        # when
        standard_scaler = StandardScaleEstimator()
        # then
        assert standard_scaler.getLayerName() == standard_scaler.uid
        assert standard_scaler.getOutputCol() == f"{standard_scaler.uid}__output"

    def test_standard_scaler_default_sample_fraction(self):
        scaler = StandardScaleEstimator()
        assert scaler.getSampleFraction() is None

    def test_standard_scaler_sample_fraction_round_trip(self):
        scaler = StandardScaleEstimator(sampleFraction=0.5)
        assert scaler.getSampleFraction() == 0.5

    @pytest.mark.parametrize("invalid_fraction", [-0.1, 0.0, 1.0, 1.5, 2.0, -1.0])
    def test_standard_scaler_invalid_sample_fraction(self, invalid_fraction):
        scaler = StandardScaleEstimator()
        with pytest.raises(ValueError):
            scaler.setSampleFraction(invalid_fraction)

    def test_standard_scaler_fit_with_sample_fraction(self, example_dataframe_large):
        scaler = StandardScaleEstimator(
            inputCol="col1",
            outputCol="scaled_features",
            sampleFraction=0.8,
        )
        result = scaler.fit(example_dataframe_large)
        assert isinstance(result, StandardScaleTransformer)
        assert result.getInputCol() == "col1"
        assert result.getOutputCol() == "scaled_features"
        assert all(isinstance(v, float) for v in result.getMean())
        assert all(isinstance(v, float) for v in result.getVariance())
