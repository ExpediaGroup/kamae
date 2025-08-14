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

from kamae.spark.estimators import MinMaxScaleEstimator
from kamae.spark.transformers import MinMaxScaleTransformer


class TestMinMaxScale:
    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, expected_min, expected_max",
        [
            (
                "example_dataframe",
                "col1_col2_col3",
                "scaled_features",
                [1.0, 2.0, 3.0],
                [7.0, 8.0, 6.0],
            ),
            (
                "example_dataframe",
                "col1",
                "scaled_features",
                [1.0],
                [7.0],
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "scaled_features",
                [-1.0, -8.0, -6.0],
                [7.0, 8.0, 6.0],
            ),
        ],
    )
    def test_spark_min_max_scaler_fit(
        self,
        input_dataframe,
        input_col,
        output_col,
        expected_min,
        expected_max,
        request,
    ):
        # when
        input_dataframe = request.getfixturevalue(input_dataframe)
        min_max_scaler = MinMaxScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = min_max_scaler.fit(input_dataframe)
        # then
        actual_min, actual_max = actual.getMin(), actual.getMax()
        np.testing.assert_almost_equal(np.array(actual_min), np.array(expected_min))
        np.testing.assert_almost_equal(np.array(actual_max), np.array(expected_max))
        assert isinstance(actual, MinMaxScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == min_max_scaler.uid

    @pytest.mark.parametrize(
        "input_col, output_col, expected_min, expected_max",
        [
            (
                "col4",
                "scaled_features",
                [4.0, 2.0, 1.0, 9.0, 0.0],
                [7.0, 8.0, 3.0, 9.0, 0.0],
            ),
        ],
    )
    def test_spark_min_max_scaler_fit_with_masking(
        self,
        example_dataframe_with_padding,
        input_col,
        output_col,
        expected_min,
        expected_max,
    ):
        # when
        min_max_scaler = MinMaxScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
            maskValue=-1,
        )
        actual = min_max_scaler.fit(example_dataframe_with_padding)
        # then
        actual_min, actual_max = actual.getMin(), actual.getMax()
        np.testing.assert_almost_equal(np.array(actual_min), np.array(expected_min))
        np.testing.assert_almost_equal(np.array(actual_max), np.array(expected_max))
        assert isinstance(actual, MinMaxScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == min_max_scaler.uid

    @pytest.mark.parametrize(
        "input_col, output_col, expected_min, expected_max",
        [
            (
                "col1_col2_col3",
                "scaled_features",
                [4.0, 2.0, 3.0],
                [7.0, 8.0, 6.0],
            ),
        ],
    )
    def test_spark_min_max_scaler_fit_with_nulls(
        self,
        example_dataframe_with_nulls,
        input_col,
        output_col,
        expected_min,
        expected_max,
    ):
        # when
        min_max_scaler = MinMaxScaleEstimator(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = min_max_scaler.fit(example_dataframe_with_nulls)
        # then
        actual_min, actual_max = actual.getMin(), actual.getMax()
        np.testing.assert_almost_equal(np.array(actual_min), np.array(expected_min))
        np.testing.assert_almost_equal(np.array(actual_max), np.array(expected_max))
        assert isinstance(actual, MinMaxScaleTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == min_max_scaler.uid

    def test_min_max_scaler_defaults(self):
        # when
        min_max_scaler = MinMaxScaleEstimator()
        # then
        assert min_max_scaler.getLayerName() == min_max_scaler.uid
        assert min_max_scaler.getOutputCol() == f"{min_max_scaler.uid}__output"
