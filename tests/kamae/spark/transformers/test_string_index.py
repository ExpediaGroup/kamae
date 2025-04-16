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
from pyspark.errors.exceptions.captured import PythonException

from kamae.spark.transformers import StringIndexTransformer


class TestStringIndex:
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
    def string_indexer_expected_0(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1),
                (4, 2, 6, "b", "c", [4, 2, 6], 2),
                (7, 8, 3, "a", "a", [7, 8, 3], 1),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col4_indexed"],
        )

    @pytest.fixture(scope="class")
    def string_indexer_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 2),
                (4, 2, 6, "b", "c", [4, 2, 6], 1),
                (7, 8, 3, "a", "a", [7, 8, 3], 2),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col4_indexed"],
        )

    @pytest.fixture(scope="class")
    def string_indexer_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 2),
                (4, 2, 6, "b", "c", [4, 2, 6], 2),
                (7, 8, 3, "a", "a", [7, 8, 3], 1),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col5_indexed"],
        )

    @pytest.fixture(scope="class")
    def string_indexer_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1),
                (4, 2, 6, "b", "c", [4, 2, 6], 1),
                (7, 8, 3, "a", "a", [7, 8, 3], 0),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col4_indexed"],
        )

    @pytest.fixture(scope="class")
    def string_indexer_w_nulls_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (None, 2, 3, "a", "c", [None, 2, 3], 6),
                (4, None, 6, "b", None, [4, None, 6], 0),
                (7, 8, None, None, "a", [7, 8, None], 0),
                (7, 8, None, "a", "a", [7, 8, None], 6),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col4_indexed"],
        )

    @pytest.fixture(scope="class")
    def string_indexer_w_nulls_expected_5(self, spark_session):
        return spark_session.createDataFrame(
            [
                (None, 2, 3, "a", "c", [None, 2, 3], 5),
                (4, None, 6, "b", None, [4, None, 6], 0),
                (7, 8, None, None, "a", [7, 8, None], 4),
                (7, 8, None, "a", "a", [7, 8, None], 4),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col5_indexed"],
        )

    @pytest.mark.parametrize(
        "input_col, output_col, labels_array, mask_token, num_oov_indices, input_dataframe, expected_dataframe",
        [
            (
                "col4",
                "col4_indexed",
                ["a", "c"],
                None,
                1,
                "example_index_input_with_string_arrays",
                "string_index_col4_array_expected",
            ),
            (
                "col4",
                "col4_indexed",
                ["a", "b"],
                None,
                1,
                "example_dataframe",
                "string_indexer_expected_0",
            ),
            (
                "col4",
                "col4_indexed",
                ["b", "a"],
                None,
                1,
                "example_dataframe",
                "string_indexer_expected_1",
            ),
            (
                "col5",
                "col5_indexed",
                ["a", "c"],
                None,
                1,
                "example_dataframe",
                "string_indexer_expected_2",
            ),
            (
                "col5",
                "col5_indexed",
                ["c"],
                None,
                1,
                "example_dataframe",
                "string_indexer_expected_3",
            ),
            (
                "col4",
                "col4_indexed",
                ["a"],
                "b",
                5,
                "example_dataframe_with_nulls",
                "string_indexer_w_nulls_expected_4",
            ),
            (
                "col5",
                "col5_indexed",
                ["a", "c"],
                "mask",
                3,
                "example_dataframe_with_nulls",
                "string_indexer_w_nulls_expected_5",
            ),
        ],
    )
    def test_spark_string_indexer_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        labels_array,
        mask_token,
        num_oov_indices,
        expected_dataframe,
        request,
    ):
        # given
        example_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        string_indexer_model = StringIndexTransformer(
            inputCol=input_col,
            outputCol=output_col,
            labelsArray=labels_array,
            maskToken=mask_token,
            numOOVIndices=num_oov_indices,
        )
        actual = string_indexer_model.transform(example_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_string_indexer_model_defaults(self):
        # when
        string_indexer_model = StringIndexTransformer(
            labelsArray=["a", "b", "c"],
        )
        # then
        assert string_indexer_model.getLayerName() == string_indexer_model.uid
        assert string_indexer_model.getStringOrderType() == "frequencyDesc"
        assert string_indexer_model.getNumOOVIndices() == 1
        assert (
            string_indexer_model.getOutputCol() == f"{string_indexer_model.uid}__output"
        )

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
            (tf.constant(["a", "b", "c", "d"]), "string", "string", ["a"], "b", 5),
            (tf.constant([True, False, True]), "string", "float", ["True"], None, 1),
            (tf.constant(["e", "f", "g"]), None, "bigint", ["f", "e", "g"], "h", 1),
            (tf.constant([1, 2, 3]), "string", None, ["2", "3", "7"], "3, 4", 5),
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
    def test_string_indexer_spark_tf_parity(
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
        transformer = StringIndexTransformer(
            inputCol="input",
            outputCol="output",
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
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col",
        [
            ("example_dataframe_w_null_characters", "col1", "hash_col1"),
            ("example_dataframe_w_null_characters", "col2", "hash_col2"),
        ],
    )
    def test_string_indexer_w_null_characters_raises_error(
        self, input_dataframe, input_col, output_col, request
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        transformer = StringIndexTransformer(
            inputCol=input_col, outputCol=output_col, labelsArray=["a", "b", "c"]
        )
        with pytest.raises(PythonException):
            transformer.transform(input_dataframe).show()
