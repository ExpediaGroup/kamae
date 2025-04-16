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

from kamae.spark.transformers import LogicalAndTransformer

from ..test_helpers import tensor_to_python_type


class TestLogicalAnd:
    @pytest.fixture(scope="class")
    def example_dataframe_with_nested_array_bools(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [[True, True, True, True, True]],
                        [[True, False, True, True, False]],
                    ],
                    [
                        [[True, True, False, False, True]],
                        [[False, False, True, True, True]],
                    ],
                ),
                (
                    [[[False, True, False, True, True]]],
                    [[[True, True, True, True, True]]],
                ),
            ],
            ["col1", "col2"],
        )

    @pytest.fixture(scope="class")
    def example_dataframe_with_bools(self, spark_session):
        return spark_session.createDataFrame(
            [
                (True, False, True, True, False),
                (True, True, False, False, True),
                (False, True, False, True, True),
            ],
            ["col1", "col2", "col3", "col4", "col5"],
        )

    @pytest.fixture(scope="class")
    def logical_and_transform_col1_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (True, False, True, True, False, False),
                (True, True, False, False, True, True),
                (False, True, False, True, True, False),
            ],
            ["col1", "col2", "col3", "col4", "col5", "logical_and_col1_col2"],
        )

    @pytest.fixture(scope="class")
    def logical_and_transform_col1_col2_col3_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (True, False, True, True, False, False),
                (True, True, False, False, True, False),
                (False, True, False, True, True, False),
            ],
            ["col1", "col2", "col3", "col4", "col5", "logical_and_col1_col2_col3"],
        )

    @pytest.fixture(scope="class")
    def logical_and_transform_col4_col5_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (True, False, True, True, False, False),
                (True, True, False, False, True, False),
                (False, True, False, True, True, True),
            ],
            ["col1", "col2", "col3", "col4", "col5", "logical_and_col4_col5"],
        )

    @pytest.fixture(scope="class")
    def logical_and_array_transform_col1_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [[True, True, True, True, True]],
                        [[True, False, True, True, False]],
                    ],
                    [
                        [[True, True, False, False, True]],
                        [[False, False, True, True, True]],
                    ],
                    [
                        [[True, True, False, False, True]],
                        [[False, False, True, True, False]],
                    ],
                ),
                (
                    [[[False, True, False, True, True]]],
                    [[[True, True, True, True, True]]],
                    [[[False, True, False, True, True]]],
                ),
            ],
            ["col1", "col2", "logical_and_col1_col2"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, expected_dataframe",
        [
            (
                "example_dataframe_with_bools",
                ["col1", "col2"],
                "logical_and_col1_col2",
                "logical_and_transform_col1_col2_expected",
            ),
            (
                "example_dataframe_with_bools",
                ["col1", "col2", "col3"],
                "logical_and_col1_col2_col3",
                "logical_and_transform_col1_col2_col3_expected",
            ),
            (
                "example_dataframe_with_bools",
                ["col4", "col5"],
                "logical_and_col4_col5",
                "logical_and_transform_col4_col5_expected",
            ),
            (
                "example_dataframe_with_nested_array_bools",
                ["col1", "col2"],
                "logical_and_col1_col2",
                "logical_and_array_transform_col1_col2_expected",
            ),
        ],
    )
    def test_spark_logical_and_transform(
        self,
        input_dataframe,
        input_cols,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = LogicalAndTransformer(
            inputCols=input_cols,
            outputCol=output_col,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are and equal"

    def test_logical_and_transform_defaults(self):
        # when
        logical_and_transform = LogicalAndTransformer()
        # then
        assert logical_and_transform.getLayerName() == logical_and_transform.uid
        assert (
            logical_and_transform.getOutputCol()
            == f"{logical_and_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype",
        [
            (
                [
                    tf.constant(["True", "False", "True"]),
                    tf.constant([1, 1, 1], dtype="int32"),
                    tf.constant([False, False, True]),
                ],
                "boolean",
                "string",
            ),
            (
                [
                    tf.constant([1, 1, 1, 0, 0, 0], dtype="int32"),
                    tf.constant(["True", "False", "True", "False", "True", "False"]),
                    tf.constant([1, 1, 1, 0, 0, 0], dtype="int32"),
                    tf.constant([True, False, True, False, True, False]),
                ],
                "boolean",
                None,
            ),
        ],
    )
    def test_logical_and_transform_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = LogicalAndTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )

        spark_df = spark_session.createDataFrame(
            [
                tuple([tensor_to_python_type(ti) for ti in t])
                for t in zip(*input_tensors)
            ],
            col_names,
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensors).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
