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

from kamae.spark.estimators import SharedOneHotEncodeEstimator
from kamae.spark.transformers import SharedOneHotEncodeTransformer


class TestSharedOneHotEncode:
    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_cols, max_num_labels, expected_labels_array",
        [
            (
                "example_dataframe",
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                None,
                ["a", "c", "b"],
            ),
            (
                "example_dataframe",
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                None,
                ["a", "c", "b"],
            ),
            (
                "example_dataframe",
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                2,
                ["a", "c"],
            ),
            (
                "example_dataframe",
                ["col4"],
                ["col4_indexed"],
                None,
                ["a", "b"],
            ),
            (
                "example_index_input_with_string_arrays",
                ["col4"],
                ["col4_indexed"],
                None,
                ["a", "c", "d", "h", "l", "o", "p", "s", "t", "w", "x"],
            ),
            (
                "example_index_input_with_string_arrays",
                ["col4"],
                ["col4_indexed"],
                15,
                ["a", "c", "d", "h", "l", "o", "p", "s", "t", "w", "x"],
            ),
        ],
    )
    def test_spark_shared_one_hot_encoder_fit(
        self,
        input_dataframe,
        input_cols,
        output_cols,
        max_num_labels,
        expected_labels_array,
        request,
    ):
        # when
        example_dataframe = request.getfixturevalue(input_dataframe)
        oh_encoder = SharedOneHotEncodeEstimator(
            inputCols=input_cols,
            outputCols=output_cols,
            maxNumLabels=max_num_labels,
        )
        actual = oh_encoder.fit(example_dataframe)
        # then
        actual_labels_array = actual.getLabelsArray()
        np.testing.assert_equal(
            np.sort(np.array(actual_labels_array)),
            np.sort(np.array(expected_labels_array)),
        )
        assert isinstance(actual, SharedOneHotEncodeTransformer)
        assert actual.getInputCols() == input_cols
        assert actual.getOutputCols() == output_cols
        assert actual.getLayerName() == oh_encoder.uid

    def test_shared_one_hot_encoder_defaults(self):
        # when
        oh_encoder = SharedOneHotEncodeEstimator()
        # then
        assert oh_encoder.getLayerName() == oh_encoder.uid
