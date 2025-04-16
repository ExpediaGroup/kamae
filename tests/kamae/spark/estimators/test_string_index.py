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

from kamae.spark.estimators import StringIndexEstimator
from kamae.spark.transformers import StringIndexTransformer


class TestStringIndex:
    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, string_order_type, max_num_labels, expected_labels_array",
        [
            (
                "example_dataframe",
                "col4",
                "col4_indexed",
                "frequencyAsc",
                None,
                ["b", "a"],
            ),
            ("example_dataframe", "col4", "col4_indexed", "frequencyAsc", 1, ["b"]),
            (
                "example_dataframe",
                "col4",
                "col4_indexed",
                "frequencyDesc",
                None,
                ["a", "b"],
            ),
            ("example_dataframe", "col4", "col4_indexed", "frequencyDesc", 1, ["a"]),
            (
                "example_dataframe",
                "col5",
                "col5_indexed",
                "alphabeticalAsc",
                None,
                ["a", "c"],
            ),
            (
                "example_dataframe",
                "col5",
                "col5_indexed",
                "alphabeticalDesc",
                None,
                ["c", "a"],
            ),
            (
                "example_index_input_with_string_arrays",
                "col4",
                "col4_indexed",
                "alphabeticalAsc",
                None,
                ["a", "c", "d", "h", "l", "o", "p", "s", "t", "w", "x"],
            ),
        ],
    )
    def test_spark_string_indexer_fit(
        self,
        input_dataframe,
        input_col,
        output_col,
        string_order_type,
        max_num_labels,
        expected_labels_array,
        request,
    ):
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        string_indexer = StringIndexEstimator(
            inputCol=input_col,
            outputCol=output_col,
            stringOrderType=string_order_type,
            maxNumLabels=max_num_labels,
        )
        actual = string_indexer.fit(input_dataframe)
        # then
        actual_labels_array = actual.getLabelsArray()
        np.testing.assert_equal(
            np.array(actual_labels_array), np.array(expected_labels_array)
        )
        assert isinstance(actual, StringIndexTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == string_indexer.uid

    @pytest.mark.parametrize(
        "input_col, output_col, string_order_type, max_num_labels, expected_labels_array",
        [
            ("col4", "col4_indexed", "frequencyAsc", None, ["b", "a"]),
            ("col4", "col4_indexed", "frequencyDesc", None, ["a", "b"]),
            ("col5", "col5_indexed", "alphabeticalAsc", None, ["a", "c"]),
            ("col5", "col5_indexed", "alphabeticalDesc", None, ["c", "a"]),
            ("col5", "col5_indexed", "alphabeticalDesc", 1, ["c"]),
        ],
    )
    def test_spark_string_indexer_fit_with_nulls(
        self,
        example_dataframe_with_nulls,
        input_col,
        output_col,
        string_order_type,
        max_num_labels,
        expected_labels_array,
    ):
        # when
        string_indexer = StringIndexEstimator(
            inputCol=input_col,
            outputCol=output_col,
            stringOrderType=string_order_type,
            maxNumLabels=max_num_labels,
        )
        actual = string_indexer.fit(example_dataframe_with_nulls)
        # then
        actual_labels_array = actual.getLabelsArray()
        np.testing.assert_equal(
            np.array(actual_labels_array), np.array(expected_labels_array)
        )
        assert isinstance(actual, StringIndexTransformer)
        assert actual.getInputCol() == input_col
        assert actual.getOutputCol() == output_col
        assert actual.getLayerName() == string_indexer.uid

    def test_string_indexer_defaults(self):
        # when
        string_indexer = StringIndexEstimator()
        # then
        assert string_indexer.getStringOrderType() == "frequencyDesc"
        assert string_indexer.getLayerName() == string_indexer.uid
        assert string_indexer.getOutputCol() == f"{string_indexer.uid}__output"
