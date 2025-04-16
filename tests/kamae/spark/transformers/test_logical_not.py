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

from kamae.spark.transformers import LogicalNotTransformer


class TestLogicalNot:
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
    def logical_not_transform_col1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (True, False, True, True, False, False),
                (True, True, False, False, True, False),
                (False, True, False, True, True, True),
            ],
            ["col1", "col2", "col3", "col4", "col5", "logical_not_col1"],
        )

    @pytest.fixture(scope="class")
    def logical_not_transform_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (True, False, True, True, False, True),
                (True, True, False, False, True, False),
                (False, True, False, True, True, False),
            ],
            ["col1", "col2", "col3", "col4", "col5", "logical_not_col2"],
        )

    @pytest.fixture(scope="class")
    def logical_not_transform_array_expected(self, spark_session):
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
                        [[False, False, False, False, False]],
                        [[False, True, False, False, True]],
                    ],
                ),
                (
                    [[[False, True, False, True, True]]],
                    [[[True, True, True, True, True]]],
                    [[[True, False, True, False, False]]],
                ),
            ],
            ["col1", "col2", "logical_not_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, expected_dataframe",
        [
            (
                "example_dataframe_with_bools",
                "col1",
                "logical_not_col1",
                "logical_not_transform_col1_expected",
            ),
            (
                "example_dataframe_with_bools",
                "col2",
                "logical_not_col2",
                "logical_not_transform_col2_expected",
            ),
            (
                "example_dataframe_with_nested_array_bools",
                "col1",
                "logical_not_col1",
                "logical_not_transform_array_expected",
            ),
        ],
    )
    def test_spark_logical_not_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = LogicalNotTransformer(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_logical_not_transform_defaults(self):
        # when
        logical_not_transform = LogicalNotTransformer()
        # then
        assert logical_not_transform.getLayerName() == logical_not_transform.uid
        assert (
            logical_not_transform.getOutputCol()
            == f"{logical_not_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype",
        [
            (tf.constant([True, False, True]), None, "string"),
            (tf.constant([1, 1, 1, 1, 0, 1]), "boolean", "int"),
            (tf.constant(["False", "False", "True"]), "boolean", "string"),
            (tf.constant([False, False, False, False, False, False]), None, None),
        ],
    )
    def test_logical_not_transform_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype
    ):
        # given
        transformer = LogicalNotTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        if input_tensor.dtype.name == "string":
            spark_df = spark_session.createDataFrame(
                [(v.decode("utf-8"),) for v in input_tensor.numpy().tolist()], ["input"]
            )
        else:
            spark_df = spark_session.createDataFrame(
                [(v,) for v in input_tensor.numpy().tolist()], ["input"]
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
