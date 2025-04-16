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

from kamae.spark.transformers import OneHotEncodeTransformer


class TestOneHotEncode:
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
                        [[1, 0, 0], [0, 0, 1], [0, 0, 1]],
                        [[1, 0, 0], [0, 0, 1], [0, 0, 1]],
                        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                    [
                        [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                    [
                        [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
                        [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                        [[1, 0, 0], [0, 0, 0], [1, 0, 0]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col4_encoded"],
        )

    @pytest.fixture(scope="class")
    def one_hot_encoder_col4_array_keep_unseen_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]],
                    [
                        [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
                        [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
                        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                    [
                        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
                        [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                        [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                    [
                        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
                        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
                        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col4_encoded"],
        )

    @pytest.fixture(scope="class")
    def ohe_input_dataframe(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a",),
                ("b",),
                ("a",),
                ("c",),
                ("d",),
                ("e",),
            ],
            ["col1"],
        )

    @pytest.fixture(scope="class")
    def expected_output_dataframe_drop_unseen(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", [1, 0, 0, 0]),
                ("b", [0, 1, 0, 0]),
                ("a", [1, 0, 0, 0]),
                ("c", [0, 0, 1, 0]),
                ("d", [0, 0, 0, 1]),
                ("e", [0, 0, 0, 0]),
            ],
            ["col1", "col1_ohe"],
        )

    @pytest.fixture(scope="class")
    def expected_output_dataframe_keep_unseen(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", [0, 1, 0, 0, 0]),
                ("b", [0, 0, 1, 0, 0]),
                ("a", [0, 1, 0, 0, 0]),
                ("c", [0, 0, 0, 1, 0]),
                ("d", [0, 0, 0, 0, 1]),
                ("e", [1, 0, 0, 0, 0]),
            ],
            ["col1", "col1_ohe"],
        )

    @pytest.fixture(scope="class")
    def expected_output_dataframe_mask_token(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("a", [0, 0, 1, 0, 0, 0]),
                ("b", [0, 0, 0, 1, 0, 0]),
                ("a", [0, 0, 1, 0, 0, 0]),
                ("c", [0, 0, 0, 0, 1, 0]),
                ("d", [0, 0, 0, 0, 0, 1]),
                ("e", [1, 0, 0, 0, 0, 0]),
            ],
            ["col1", "col1_ohe"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, labels_array, num_oov, drop_unseen, mask_token, expected_dataframe_name",
        [
            (
                "example_index_input_with_string_arrays",
                "col4",
                "col4_encoded",
                ["a", "b", "c"],
                1,
                False,
                None,
                "one_hot_encoder_col4_array_keep_unseen_expected",
            ),
            (
                "example_index_input_with_string_arrays",
                "col4",
                "col4_encoded",
                ["a", "b", "c"],
                4,
                True,
                None,
                "one_hot_encoder_col4_array_drop_unseen_expected",
            ),
            (
                "ohe_input_dataframe",
                "col1",
                "col1_ohe",
                ["a", "b", "c", "d"],
                3,
                True,
                None,
                "expected_output_dataframe_drop_unseen",
            ),
            (
                "ohe_input_dataframe",
                "col1",
                "col1_ohe",
                ["a", "b", "c", "d"],
                1,
                False,
                None,
                "expected_output_dataframe_keep_unseen",
            ),
            (
                "ohe_input_dataframe",
                "col1",
                "col1_ohe",
                ["a", "b", "c", "d"],
                1,
                False,
                "e",
                "expected_output_dataframe_mask_token",
            ),
        ],
    )
    def test_spark_one_hot_encoder_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        labels_array,
        num_oov,
        drop_unseen,
        mask_token,
        expected_dataframe_name,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe_name)
        # when
        oh_encoder_model = OneHotEncodeTransformer(
            inputCol=input_col,
            outputCol=output_col,
            labelsArray=labels_array,
            numOOVIndices=num_oov,
            dropUnseen=drop_unseen,
            maskToken=mask_token,
        )
        actual = oh_encoder_model.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_one_hot_encoder_model_defaults(self):
        # when
        one_hot_encoder_model = OneHotEncodeTransformer(
            labelsArray=["a", "b", "c"],
        )
        # then
        assert one_hot_encoder_model.getLayerName() == one_hot_encoder_model.uid
        assert (
            one_hot_encoder_model.getOutputCol()
            == f"{one_hot_encoder_model.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, oov_indices, drop_unseen, mask_token, labels_array",
        [
            (
                tf.constant(["a", "b", "c", "a", "b", "c", "a", "b"]),
                None,
                None,
                1,
                False,
                "d",
                ["a", "b", "c"],
            ),
            (
                tf.constant([1, 2, 3, 4, 5, 5, 1, 2], dtype="int32"),
                "string",
                "string",
                1,
                False,
                "hello",
                ["1", "3"],
            ),
            (
                tf.constant(["a", "b", "c", "a", "b", "c", "a", "b", "z"]),
                None,
                "string",
                1,
                False,
                None,
                ["z", "a"],
            ),
            (
                tf.constant(["a", "b", "c", "a", "b", "c", "a", "b", "z"]),
                None,
                "string",
                1,
                True,
                "s",
                ["z", "a"],
            ),
            (
                tf.constant(["a", "b", "c", "a", "b", "c", "a", "b", "z"]),
                None,
                "string",
                2,
                True,
                "d",
                ["z", "a"],
            ),
            (
                tf.constant(["a", "b", "c", "a", "b", "c", "a", "b", "z"]),
                None,
                "string",
                2,
                False,
                None,
                ["z", "a"],
            ),
        ],
    )
    def test_one_hot_encoder_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        oov_indices,
        drop_unseen,
        mask_token,
        labels_array,
    ):
        # given
        transformer = OneHotEncodeTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            labelsArray=labels_array,
            numOOVIndices=oov_indices,
            maskToken=mask_token,
            dropUnseen=drop_unseen,
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
        decoder = lambda x: x.decode("utf-8")
        vec_decoder = np.vectorize(decoder)
        tensorflow_values = [
            vec_decoder(v) if isinstance(v[0], bytes) else v
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
    def test_one_hot_encoder_w_null_characters_raises_error(
        self, input_dataframe, input_col, output_col, request
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        transformer = OneHotEncodeTransformer(
            inputCol=input_col, outputCol=output_col, labelsArray=["a", "b", "c"]
        )
        with pytest.raises(PythonException):
            transformer.transform(input_dataframe).show()
