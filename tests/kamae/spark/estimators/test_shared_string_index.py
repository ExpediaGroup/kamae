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

from kamae.spark.estimators import SharedStringIndexEstimator
from kamae.spark.transformers import SharedStringIndexTransformer


class TestSharedStringIndex:
    @pytest.mark.parametrize(
        "input_cols, output_cols, string_order_type, max_num_labels, expected_labels_array",
        [
            (
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                "frequencyAsc",
                None,
                ["b", "c", "a"],
            ),
            (
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                "frequencyDesc",
                None,
                ["a", "c", "b"],
            ),
            (
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                "alphabeticalAsc",
                None,
                ["a", "b", "c"],
            ),
            (
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                "alphabeticalDesc",
                None,
                ["c", "b", "a"],
            ),
            (
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                "alphabeticalDesc",
                2,
                ["c", "b"],
            ),
            (
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                "alphabeticalDesc",
                20,
                ["c", "b", "a"],
            ),
        ],
    )
    def test_spark_shared_string_indexer_fit(
        self,
        example_dataframe,
        input_cols,
        output_cols,
        string_order_type,
        max_num_labels,
        expected_labels_array,
    ):
        # when
        shared_string_indexer = SharedStringIndexEstimator(
            inputCols=input_cols,
            outputCols=output_cols,
            stringOrderType=string_order_type,
            maxNumLabels=max_num_labels,
        )
        actual = shared_string_indexer.fit(example_dataframe)
        # then
        actual_labels_array = actual.getLabelsArray()
        np.testing.assert_equal(
            np.array(actual_labels_array), np.array(expected_labels_array)
        )
        assert isinstance(actual, SharedStringIndexTransformer)
        assert actual.getInputCols() == input_cols
        assert actual.getOutputCols() == output_cols
        assert actual.getLayerName() == shared_string_indexer.uid

    def test_shared_string_indexer_defaults(self):
        # when
        shared_string_indexer = SharedStringIndexEstimator()
        # then
        assert shared_string_indexer.getStringOrderType() == "frequencyDesc"
        assert shared_string_indexer.getLayerName() == shared_string_indexer.uid
