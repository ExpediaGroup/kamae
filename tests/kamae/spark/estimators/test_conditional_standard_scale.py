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
        "input_dataframe, input_col, output_col, expected_mean, expected_stddev",
        [
            (
                "example_dataframe",
                "col1_col2_col3",
                "scaled_features",
                [4.0, 4.0, 4.0],
                [2.449489742783178, 2.8284271247461903, 1.4142135623730951],
            ),
            (
                "example_dataframe",
                "col1",
                "scaled_features",
                [4.0],
                [2.449489742783178],
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "scaled_features",
                [3.8333333, 1.6666667, 0.5],
                [2.7028791, 4.6067583, 4.2130749],
            ),
        ],
    )
    def test_spark_conditional_standard_scaler_fit(
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
        standard_scaler = ConditionalStandardScaleEstimator(
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
        assert isinstance(actual, ConditionalStandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    @pytest.mark.parametrize(
        "dataframe_name, input_col, mask_cols, mask_operators, mask_values, relevance_col, scaling_function, skip_zeros, epsilon, nan_fill_value, output_col, expected_mean, expected_stddev",
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
                [1.4142135, 3.681787, 0.942809, 4.7140452, 0.4714045],
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
                [0.0, 0.0, 0.0, 0.0, 0.5773503],
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
                [1.4142136, 0.0, 0.0],
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
            # Binary moments when nothing left on conditions are mean=1 and stddev=0
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
        expected_stddev,
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
        actual_mean, actual_stddev = actual.getMean(), actual.getStddev()
        np.testing.assert_almost_equal(np.array(actual_mean), np.array(expected_mean))
        np.testing.assert_almost_equal(
            np.array(actual_stddev), np.array(expected_stddev)
        )
        assert isinstance(actual, ConditionalStandardScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == standard_scaler.uid

    @pytest.mark.parametrize(
        "input_col, output_col, expected_mean, expected_stddev",
        [
            (
                "col1_col2_col3",
                "scaled_features",
                [6.0, 6.0, 4.5],
                [1.4142135623730951, 2.8284271247461903, 1.5],
            ),
        ],
    )
    def test_spark_conditional_standard_scaler_fit_with_nulls(
        self,
        example_dataframe_with_nulls,
        input_col,
        output_col,
        expected_mean,
        expected_stddev,
    ):
        # when
        standard_scaler = ConditionalStandardScaleEstimator(
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
