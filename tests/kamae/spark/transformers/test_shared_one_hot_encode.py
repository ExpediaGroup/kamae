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

from kamae.spark.transformers import SharedOneHotEncodeTransformer


class TestSharedOneHotEncode:
    @pytest.fixture(scope="class")
    def shared_one_hot_encoder_col4_array_drop_unseen_expected(self, spark_session):
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
    def shared_one_hot_encoder_col4_array_keep_unseen_expected(self, spark_session):
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
    def shared_one_hot_encoder_expected_0(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [0, 1, 0]),
                (4, 2, 6, "b", "c", [4, 2, 6], [0, 0, 1]),
                (7, 8, 3, "a", "a", [7, 8, 3], [0, 1, 0]),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col4_encoded"],
        )

    @pytest.fixture(scope="class")
    def shared_one_hot_encoder_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [0, 1]),
                (4, 2, 6, "b", "c", [4, 2, 6], [0, 1]),
                (7, 8, 3, "a", "a", [7, 8, 3], [1, 0]),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col5_encoded"],
        )

    @pytest.fixture(scope="class")
    def shared_one_hot_encoder_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [0, 1, 0, 0], [0, 0, 1, 0]),
                (4, 2, 6, "b", "c", [4, 2, 6], [0, 0, 0, 1], [0, 0, 1, 0]),
                (7, 8, 3, "a", "a", [7, 8, 3], [0, 1, 0, 0], [0, 1, 0, 0]),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_encoded",
                "col5_encoded",
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_cols, labels_array, drop_unseen, expected_dataframe",
        [
            (
                "example_index_input_with_string_arrays",
                ["col4"],
                ["col4_encoded"],
                ["a", "b", "c"],
                False,
                "shared_one_hot_encoder_col4_array_keep_unseen_expected",
            ),
            (
                "example_index_input_with_string_arrays",
                ["col4"],
                ["col4_encoded"],
                ["a", "b", "c"],
                True,
                "shared_one_hot_encoder_col4_array_drop_unseen_expected",
            ),
            (
                "example_dataframe",
                ["col4"],
                ["col4_encoded"],
                ["a", "b"],
                False,
                "shared_one_hot_encoder_expected_0",
            ),
            (
                "example_dataframe",
                ["col5"],
                ["col5_encoded"],
                ["a", "c"],
                True,
                "shared_one_hot_encoder_expected_1",
            ),
            (
                "example_dataframe",
                ["col4", "col5"],
                ["col4_encoded", "col5_encoded"],
                ["a", "c", "b"],
                False,
                "shared_one_hot_encoder_expected_2",
            ),
        ],
    )
    def test_spark_shared_one_hot_encoder_transform(
        self,
        input_dataframe,
        input_cols,
        output_cols,
        labels_array,
        drop_unseen,
        expected_dataframe,
        request,
    ):
        # given
        example_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        oh_encoder_model = SharedOneHotEncodeTransformer(
            inputCols=input_cols,
            outputCols=output_cols,
            labelsArray=labels_array,
            dropUnseen=drop_unseen,
        )
        actual = oh_encoder_model.transform(example_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_shared_one_hot_encoder_model_defaults(self):
        # when
        one_hot_encoder_model = SharedOneHotEncodeTransformer(
            labelsArray=["a", "b", "c"],
        )
        # then
        assert one_hot_encoder_model.getLayerName() == one_hot_encoder_model.uid

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, labels_array, drop_unseen",
        [
            (
                [tf.constant(["a", "b", "c", "a", "b", "c", "a", "b"])],
                None,
                None,
                ["a", "b", "c"],
                True,
            ),
            (
                [tf.constant([1, 2, 3, 4, 5, 5, 1, 2], dtype="int32")],
                "string",
                "string",
                ["1", "3"],
                False,
            ),
            (
                [tf.constant(["a", "b", "c", "a", "b", "c", "a", "b", "z"])],
                None,
                "string",
                ["z", "a"],
                False,
            ),
        ],
    )
    def test_one_hot_encoder_spark_tf_parity(
        self,
        spark_session,
        input_tensors,
        input_dtype,
        output_dtype,
        labels_array,
        drop_unseen,
    ):
        # given
        transformer = SharedOneHotEncodeTransformer(
            inputCols=["input"],
            outputCols=["output"],
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            labelsArray=labels_array,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                (v.decode("utf-8") if isinstance(v, bytes) else v,)
                for v in input_tensors[0].numpy().tolist()
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
            for v in transformer.get_tf_layer()[0](input_tensors[0]).numpy().tolist()
        ]
        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
