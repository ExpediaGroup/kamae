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

from kamae.spark.transformers import AbsoluteValueTransformer


class TestAbsoluteValue:
    @pytest.fixture(scope="class")
    def example_dataframe_with_negatives(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, -2, 3),
                (-4, 2, -6),
                (-7, 8, 3),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def abs_transform_nested_arrays_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [1.0, -2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, -3.0],
                        [4.0, 2.0, -6.0],
                    ],
                    [
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                    ],
                    [
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [4.0, 2.0, 6.0],
                    ],
                ),
                (
                    [
                        [4.0, -2.0, 6.0],
                        [4.0, -2.0, 6.0],
                        [4.0, 2.0, -6.0],
                        [7.0, 8.0, 3.0],
                    ],
                    [
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                    ],
                    [
                        [4.0, 2.0, 6.0],
                        [4.0, 2.0, 6.0],
                        [4.0, 2.0, 6.0],
                        [7.0, 8.0, 3.0],
                    ],
                ),
                (
                    [
                        [7.0, 8.0, 3.0],
                        [7.0, -8.0, 3.0],
                        [7.0, 8.0, -3.0],
                        [-1.0, 2.0, -3.0],
                    ],
                    [
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                    ],
                    [
                        [7.0, 8.0, 3.0],
                        [7.0, 8.0, 3.0],
                        [7.0, 8.0, 3.0],
                        [1.0, 2.0, 3.0],
                    ],
                ),
            ],
            ["col1", "col2", "abs_col1"],
        )

    @pytest.fixture(scope="class")
    def abs_transform_col1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, -2, 3, 1),
                (-4, 2, -6, 4),
                (-7, 8, 3, 7),
            ],
            ["col1", "col2", "col3", "abs_col1"],
        )

    @pytest.fixture(scope="class")
    def abs_transform_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, -2, 3, 2),
                (-4, 2, -6, 2),
                (-7, 8, 3, 8),
            ],
            ["col1", "col2", "col3", "abs_col2"],
        )

    @pytest.fixture(scope="class")
    def abs_transform_col3_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, -2, 3, 3),
                (-4, 2, -6, 6),
                (-7, 8, 3, 3),
            ],
            ["col1", "col2", "col3", "abs_col3"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, expected_dataframe",
        [
            (
                "example_dataframe_with_negatives",
                "col1",
                "abs_col1",
                "abs_transform_col1_expected",
            ),
            (
                "example_dataframe_with_negatives",
                "col2",
                "abs_col2",
                "abs_transform_col2_expected",
            ),
            (
                "example_dataframe_with_negatives",
                "col3",
                "abs_col3",
                "abs_transform_col3_expected",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "abs_col1",
                "abs_transform_nested_arrays_expected",
            ),
        ],
    )
    def test_spark_absolute_transform(
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
        transformer = AbsoluteValueTransformer(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_abs_value_transform_defaults(self):
        # when
        abs_transformer = AbsoluteValueTransformer()
        # then
        assert abs_transformer.getLayerName() == abs_transformer.uid
        assert abs_transformer.getOutputCol() == f"{abs_transformer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype",
        [
            (tf.constant([1.0, -5.0, -6.0, 7.0, -8.0, 9.0]), "float", "float"),
            (tf.constant([-4.0, -5.0, -3.0, -47.0, -8.2, -11.0]), "int", None),
            (tf.constant(["1", "-2", "-3"]), "bigint", "string"),
            (tf.constant([-56.3, 55.3]), "double", "float"),
            (tf.constant([-76.4, -55.3]), None, None),
        ],
    )
    def test_absolute_value_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype
    ):
        # given
        transformer = AbsoluteValueTransformer(
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
