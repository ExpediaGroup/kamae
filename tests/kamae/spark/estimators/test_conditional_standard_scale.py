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

from kamae.spark.estimators import ConditionalStandardScaleEstimator
from kamae.spark.transformers import ConditionalStandardScaleTransformer


class TestConditionalStandardScale:
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
    def test_spark_conditional_standard_scaler_fit(
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
        standard_scaler = ConditionalStandardScaleEstimator(
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
        assert isinstance(actual, ConditionalStandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    @pytest.mark.parametrize(
        "dataframe_name, input_col, mask_cols, mask_operators, mask_values, relevance_col, scaling_function, skip_zeros, epsilon, nan_fill_value, output_col, expected_mean, expected_variance",
        [
            (
                "example_dataframe_with_padding",
                "col4",
                ["col1"],
                ["gt"],
                [0.0],
                None,
                "standard",
                False,
                None,
                None,
                "scaled_features",
                [6.0, 3.3333333, -0.3333333, 2.3333333, -0.6666667],
                [2.0, 13.5555556, 0.8888889, 22.2222222, 0.2222222],
            ),
            (
                "example_dataframe_bool",
                "col4",
                ["col1"],
                ["gt"],
                [0.0],
                "col2",
                "binary",
                False,
                None,
                None,
                "scaled_features",
                [1.0, 0.0, 0.0, 1.0, 0.3333333],
                [0.0, 0.0, 0.0, 0.0, 0.3333333],
            ),
            # Moments calculated by skipping the Nones in input
            (
                "example_dataframe_with_nulls",
                "col1_col2_col3",
                ["col1"],
                ["gt"],
                [0.0],
                None,
                "standard",
                False,
                None,
                None,
                "scaled_features",
                [6.0, 8.0, 6.0],
                [2.0, 0.0, 0.0],
            ),
            # Moments cannot be calculated since nothing left after the conditions
            (
                "example_dataframe_with_padding",
                "col4",
                ["col1"],
                ["gt"],
                [0.0],
                None,
                "standard",
                True,  # Skip zeros with eps
                100.0,  # Nothing left after epsilon condition
                0.0,  # But nanFillValue is 0, so results is 0
                "scaled_features",
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            # Binary moments when nothing left on conditions are mean=1 and variance=0
            (
                "example_dataframe_bool",
                "col4",
                ["col1"],
                ["gt"],
                [4.0],
                "col3",
                "binary",
                False,
                None,
                None,
                "scaled_features",
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ],
    )
    def test_spark_conditional_standard_scaler_fit_with_masking(
        self,
        dataframe_name,
        input_col,
        mask_cols,
        mask_operators,
        mask_values,
        relevance_col,
        scaling_function,
        skip_zeros,
        epsilon,
        nan_fill_value,
        output_col,
        expected_mean,
        expected_variance,
        request,
    ):
        df = request.getfixturevalue(dataframe_name)
        # when
        standard_scaler = ConditionalStandardScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
            maskCols=mask_cols,
            maskOperators=mask_operators,
            maskValues=mask_values,
            scalingFunction=scaling_function,
            relevanceCol=relevance_col,
            skipZeros=skip_zeros,
            epsilon=epsilon,
            nanFillValue=nan_fill_value,
        )
        actual = standard_scaler.fit(df)
        # then
        actual_mean, actual_variance = actual.getMean(), actual.getVariance()
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(
            np.array(actual_variance), np.array(expected_variance)
        )
        assert isinstance(actual, ConditionalStandardScaleTransformer)
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
    def test_spark_conditional_standard_scaler_fit_with_nulls(
        self,
        example_dataframe_with_nulls,
        input_col,
        output_col,
        expected_mean,
        expected_variance,
    ):
        # when
        standard_scaler = ConditionalStandardScaleEstimator(
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
        assert isinstance(actual, ConditionalStandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    def test_conditional_standard_scaler_defaults(self):
        # when
        standard_scaler = ConditionalStandardScaleEstimator()
        # then
        assert standard_scaler.getLayerName() == standard_scaler.uid
        assert standard_scaler.getOutputCol() == f"{standard_scaler.uid}__output"

    def test_conditional_standard_scaler_default_sample_fraction(self):
        scaler = ConditionalStandardScaleEstimator()
        assert scaler.getSampleFraction() is None

    def test_conditional_standard_scaler_sample_fraction_round_trip(self):
        scaler = ConditionalStandardScaleEstimator(sampleFraction=0.5)
        assert scaler.getSampleFraction() == 0.5

    @pytest.mark.parametrize("invalid_fraction", [-0.1, 0.0, 1.0, 1.5, 2.0, -1.0])
    def test_conditional_standard_scaler_invalid_sample_fraction(
        self, invalid_fraction
    ):
        scaler = ConditionalStandardScaleEstimator()
        with pytest.raises(ValueError):
            scaler.setSampleFraction(invalid_fraction)

    def test_conditional_standard_scaler_fit_with_sample_fraction(
        self, example_dataframe_large
    ):
        scaler = ConditionalStandardScaleEstimator(
            inputCol="col1",
            outputCol="scaled_features",
            sampleFraction=0.8,
        )
        result = scaler.fit(example_dataframe_large)
        assert isinstance(result, ConditionalStandardScaleTransformer)
        assert result.getInputCol() == "col1"
        assert result.getOutputCol() == "scaled_features"
        assert all(isinstance(v, float) for v in result.getMean())
        assert all(isinstance(v, float) for v in result.getVariance())
