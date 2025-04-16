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

from kamae.spark.estimators import OneHotEncodeEstimator
from kamae.spark.transformers import OneHotEncodeTransformer


class TestOneHotEncode:
    @pytest.fixture(scope="class")
    def ohe_example_dataframe(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a",),
                ("b",),
                ("a",),
                ("c",),
                ("d",),
            ],
            ["col1"],
        )

    @pytest.fixture(scope="class")
    def ohe_example_dataframe_frequency(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a",),
                ("a",),
                ("b",),
                ("b",),
                ("a",),
                ("c",),
                ("d",),
                ("c",),
                ("a",),
                ("c",),
            ],
            ["col1"],
        )

    @pytest.fixture(scope="class")
    def expected_output_dataframe_oov0(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", [1, 0, 0, 0]),
                ("b", [0, 1, 0, 0]),
                ("a", [1, 0, 0, 0]),
                ("c", [0, 0, 1, 0]),
                ("d", [0, 0, 0, 1]),
            ],
            ["col1", "col1_ohe"],
        )

    @pytest.fixture(scope="class")
    def expected_output_dataframe_oov1(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", [0, 1, 0, 0, 0]),
                ("b", [0, 0, 1, 0, 0]),
                ("a", [0, 1, 0, 0, 0]),
                ("c", [0, 0, 0, 1, 0]),
                ("d", [0, 0, 0, 0, 1]),
            ],
            ["col1", "col1_ohe"],
        )

    @pytest.fixture(scope="class")
    def one_hot_encoder_col4_array_drop_unseen_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        ],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                    [
                        [
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col4_encoded"],
        )

    @pytest.fixture(scope="class")
    def expected_output_dataframe_max_num_labels(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", [1, 0]),
                ("b", [0, 1]),
                ("a", [1, 0]),
                ("c", [0, 0]),
                ("d", [0, 0]),
            ],
            ["col1", "col1_ohe"],
        )

    @pytest.fixture(scope="class")
    def expected_output_dataframe_frequency_desc_max_num_labels(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", [1, 0, 0]),
                ("a", [1, 0, 0]),
                ("a", [1, 0, 0]),
                ("b", [0, 0, 1]),
                ("b", [0, 0, 1]),
                ("a", [1, 0, 0]),
                ("c", [0, 1, 0]),
                ("d", [0, 0, 0]),
                ("c", [0, 1, 0]),
                ("a", [1, 0, 0]),
                ("c", [0, 1, 0]),
            ],
            ["col1", "col1_ohe"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, string_order_type, expected_labels_array, num_oov_indices, drop_unseen, "
        "max_num_labels, expected_dataframe_name",
        [
            (
                "ohe_example_dataframe",
                "col1",
                "col1_ohe",
                "alphabeticalAsc",
                ["a", "b", "c", "d"],
                1,
                True,
                None,
                "expected_output_dataframe_oov0",
            ),
            (
                "ohe_example_dataframe",
                "col1",
                "col1_ohe",
                "alphabeticalAsc",
                ["a", "b", "c", "d"],
                1,
                False,
                None,
                "expected_output_dataframe_oov1",
            ),
            (
                "example_index_input_with_string_arrays",
                "col4",
                "col4_indexed",
                "alphabeticalAsc",
                ["a", "c", "d", "h", "l", "o", "p", "s", "t", "w", "x"],
                1,
                True,
                None,
                "one_hot_encoder_col4_array_drop_unseen_expected",
            ),
            (
                "ohe_example_dataframe",
                "col1",
                "col1_ohe",
                "alphabeticalAsc",
                ["a", "b"],
                1,
                True,
                2,
                "expected_output_dataframe_max_num_labels",
            ),
            (
                "ohe_example_dataframe",
                "col1",
                "col1_ohe",
                "alphabeticalAsc",
                ["a", "b"],
                1,
                True,
                2,
                "expected_output_dataframe_max_num_labels",
            ),
            (
                "ohe_example_dataframe_frequency",
                "col1",
                "col1_ohe",
                "frequencyDesc",
                ["a", "b", "c"],
                1,
                True,
                3,
                "expected_output_dataframe_frequency_desc_max_num_labels",
            ),
        ],
    )
    def test_spark_one_hot_encoder_fit(
        self,
        input_dataframe,
        input_col,
        output_col,
        string_order_type,
        expected_labels_array,
        num_oov_indices,
        drop_unseen,
        max_num_labels,
        expected_dataframe_name,
        request,
    ):
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected_dataframe = request.getfixturevalue(expected_dataframe_name)
        # when
        oh_encoder = OneHotEncodeEstimator(
            inputCol=input_col,
            outputCol=output_col,
            stringOrderType=string_order_type,
            numOOVIndices=num_oov_indices,
            dropUnseen=drop_unseen,
            maxNumLabels=max_num_labels,
        )
        oh_transformer = oh_encoder.fit(input_dataframe)
        actual = oh_transformer.transform(input_dataframe)
        # then
        actual_labels_array = oh_transformer.getLabelsArray()
        np.testing.assert_equal(
            np.sort(np.array(actual_labels_array)),
            np.sort(np.array(expected_labels_array)),
        )
        assert isinstance(oh_transformer, OneHotEncodeTransformer)
        assert oh_transformer.getInputCol() == input_col
        assert oh_transformer.getOutputCol() == output_col
        assert oh_transformer.getLayerName() == oh_encoder.uid
        diff = actual.exceptAll(expected_dataframe)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_one_hot_encoder_defaults(self):
        # when
        oh_encoder = OneHotEncodeEstimator()
        # then
        assert oh_encoder.getLayerName() == oh_encoder.uid
        assert oh_encoder.getOutputCol() == f"{oh_encoder.uid}__output"
