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
import tensorflow as tf

from kamae.spark.transformers import SharedStringIndexTransformer


class TestSharedStringIndex:
    @pytest.fixture(scope="class")
    def string_index_col4_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]],
                    [[1, 2, 2], [1, 2, 2], [1, 1, 1]],
                ),
                (
                    4,
                    2,
                    6,
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                    [[1, 0, 2], [1, 0, 0], [0, 0, 0]],
                ),
                (
                    7,
                    8,
                    3,
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                    [[0, 2, 2], [1, 0, 2], [1, 0, 1]],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col4_encoded"],
        )

    @pytest.fixture(scope="class")
    def shared_string_indexer_expected_0(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 3, 2),
                (4, 2, 6, "b", "c", [4, 2, 6], 1, 2),
                (7, 8, 3, "a", "a", [7, 8, 3], 3, 3),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_indexed",
                "col5_indexed",
            ],
        )

    @pytest.fixture(scope="class")
    def shared_string_indexer_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1, 2),
                (4, 2, 6, "b", "c", [4, 2, 6], 3, 2),
                (7, 8, 3, "a", "a", [7, 8, 3], 1, 1),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_indexed",
                "col5_indexed",
            ],
        )

    @pytest.fixture(scope="class")
    def shared_string_indexer_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1, 3),
                (4, 2, 6, "b", "c", [4, 2, 6], 2, 3),
                (7, 8, 3, "a", "a", [7, 8, 3], 1, 1),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_indexed",
                "col5_indexed",
            ],
        )

    @pytest.fixture(scope="class")
    def shared_string_indexer_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 3, 1),
                (4, 2, 6, "b", "c", [4, 2, 6], 2, 1),
                (7, 8, 3, "a", "a", [7, 8, 3], 3, 3),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_indexed",
                "col5_indexed",
            ],
        )

    @pytest.fixture(scope="class")
    def shared_string_indexer_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 3),
                (4, 2, 6, "b", "c", [4, 2, 6], 2),
                (7, 8, 3, "a", "a", [7, 8, 3], 3),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col4_indexed"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_cols, labels_array, expected_dataframe",
        [
            (
                "example_index_input_with_string_arrays",
                ["col4"],
                ["col4_indexed"],
                ["a", "c"],
                "string_index_col4_array_expected",
            ),
            (
                "example_dataframe",
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                ["b", "c", "a"],
                "shared_string_indexer_expected_0",
            ),
            (
                "example_dataframe",
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                ["a", "c", "b"],
                "shared_string_indexer_expected_1",
            ),
            (
                "example_dataframe",
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                ["a", "b", "c"],
                "shared_string_indexer_expected_2",
            ),
            (
                "example_dataframe",
                ["col4", "col5"],
                ["col4_indexed", "col5_indexed"],
                ["c", "b", "a"],
                "shared_string_indexer_expected_3",
            ),
            (
                "example_dataframe",
                ["col4"],
                ["col4_indexed"],
                ["c", "b", "a"],
                "shared_string_indexer_expected_4",
            ),
        ],
    )
    def test_spark_shared_string_indexer_transform(
        self,
        input_dataframe,
        input_cols,
        output_cols,
        labels_array,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        shared_string_indexer_model = SharedStringIndexTransformer(
            inputCols=input_cols,
            outputCols=output_cols,
            labelsArray=labels_array,
        )
        actual = shared_string_indexer_model.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_shared_string_indexer_model_defaults(self):
        # when
        shared_string_indexer_model = SharedStringIndexTransformer(
            labelsArray=["a", "b", "c"],
        )
        # then
        assert (
            shared_string_indexer_model.getLayerName()
            == shared_string_indexer_model.uid
        )
        assert shared_string_indexer_model.getNumOOVIndices() == 1
        assert shared_string_indexer_model.getStringOrderType() == "frequencyDesc"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, labels_array, mask_token, num_oov_indices",
        [
            (
                tf.constant(["a", "b", "c", "d"]),
                None,
                None,
                ["a", "b", "c", "d"],
                None,
                2,
            ),
            (tf.constant([1, 2, 3]), "string", "string", ["1"], "d", 1),
            (tf.constant(["e", "f", "g"]), None, "string", ["f", "e", "g"], "h", 1),
            (tf.constant(["h", "i", "j"]), None, None, ["f", "g", "h"], "i", 5),
            (
                tf.constant(["k", "l", "m"]),
                None,
                "smallint",
                ["k", "n", "m", "l"],
                "t",
                1,
            ),
        ],
    )
    def test_shared_string_indexer_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        labels_array,
        mask_token,
        num_oov_indices,
    ):
        # given
        transformer = SharedStringIndexTransformer(
            inputCols=["input"],
            outputCols=["output"],
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            labelsArray=labels_array,
            maskToken=mask_token,
            numOOVIndices=num_oov_indices,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                (v.decode("utf-8") if isinstance(v, bytes) else v,)
                for v in input_tensor.numpy().tolist()
            ],
            ["input"],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()[0](input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
