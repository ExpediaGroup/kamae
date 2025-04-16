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

from kamae.spark.estimators import ImputeEstimator
from kamae.spark.transformers import ImputeTransformer


class TestImpute:
    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, impute_method, mask_value, expected_median",
        [
            ("example_dataframe", "col1", "col1_median_imputed", "median", -999.0, 4.0),
            ("example_dataframe", "col1", "col1_median_imputed", "median", 7.0, 2.5),
            ("example_dataframe", "col1", "col1_mean_imputed", "mean", -999.0, 4.0),
            ("example_dataframe", "col1", "col1_mean_imputed", "mean", 7.0, 2.5),
            (
                "example_dataframe_with_nulls",
                "col1",
                "col1_mean_imputed",
                "mean",
                4.0,
                7.0,
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "col1_mean_imputed",
                "mean",
                -999.0,
                2.0,
            ),
        ],
    )
    def test_spark_impute_fit(
        self,
        input_dataframe,
        input_col,
        output_col,
        impute_method,
        mask_value,
        expected_median,
        request,
    ):
        input_dataframe = request.getfixturevalue(input_dataframe)
        imputer = ImputeEstimator(
            inputCol=input_col,
            outputCol=output_col,
            maskValue=mask_value,
            imputeMethod=impute_method,
        )

        actual = imputer.fit(input_dataframe)

        actual_median = actual.getImputeValue()

        np.testing.assert_almost_equal(
            np.array(actual_median), np.array(expected_median)
        )

        assert isinstance(actual, ImputeTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == imputer.uid
