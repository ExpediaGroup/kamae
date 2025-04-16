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
from pyspark.sql.types import DoubleType

from kamae.spark.transformers import SubtractTransformer

from ..test_helpers import tensor_to_python_type


class TestSubtract:
    @pytest.fixture(scope="class")
    def subtract_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], -4),
                (4, 2, 6, "b", "c", [4, 2, 6], -1),
                (7, 8, 3, "a", "a", [7, 8, 3], 2),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "subtract_col1_w_constant",
            ],
        )

    @pytest.fixture(scope="class")
    def subtract_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], -4),
                (4, 2, 6, "b", "c", [4, 2, 6], -4),
                (7, 8, 3, "a", "a", [7, 8, 3], -4),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "subtract_col1_col2_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def subtract_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], -2),
                (4, 2, 6, "b", "c", [4, 2, 6], -2),
                (7, 8, 3, "a", "a", [7, 8, 3], 4),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "subtract_col1_col3",
            ],
        )

    @pytest.fixture(scope="class")
    def subtract_transform_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 14),
                (4, 2, 6, "b", "c", [4, 2, 6], 14),
                (7, 8, 3, "a", "a", [7, 8, 3], 20),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "subtract_col2_w_constant",
            ],
        )

    @pytest.fixture(scope="class")
    def subtract_transform_array_col1(self, spark_session):
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
                        [4.0, 5.0, -1.2],
                        [41.0, -89.45, 56.5],
                        [14.0, -6.0, 9.5],
                        [43.45, -2.0, 4.5],
                    ],
                    [
                        [-9.0, -12.0, -7.0],
                        [-9.0, -8.0, -7.0],
                        [-9.0, -8.0, -13.0],
                        [-6.0, -8.0, -16.0],
                    ],
                )
            ],
            ["col1", "col2", "subtract_col1_w_constant"],
        )

    @pytest.fixture(scope="class")
    def subtract_transform_array_col1_col2(self, spark_session):
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
                        [4.0, 5.0, -1.2],
                        [41.0, -89.45, 56.5],
                        [14.0, -6.0, 9.5],
                        [43.45, -2.0, 4.5],
                    ],
                    [
                        [-3.0, -7.0, 4.2],
                        [-40.0, 91.45, -53.5],
                        [-13.0, 8.0, -12.5],
                        [-39.45, 4.0, -10.5],
                    ],
                )
            ],
            ["col1", "col2", "subtract_col1_col2"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, input_cols, float_constant, output_col, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1",
                None,
                5,
                "subtract_col1_w_constant",
                "subtract_transform_expected_1",
            ),
            (
                "example_dataframe",
                None,
                ["col1", "col2", "col3"],
                None,
                "subtract_col1_col2_col3",
                "subtract_transform_expected_2",
            ),
            (
                "example_dataframe",
                None,
                ["col1", "col3"],
                None,
                "subtract_col1_col3",
                "subtract_transform_expected_3",
            ),
            (
                "example_dataframe",
                "col2",
                None,
                -12,
                "subtract_col2_w_constant",
                "subtract_transform_expected_4",
            ),
            (
                "example_dataframe_w_multiple_numeric_nested_arrays",
                "col1",
                None,
                10,
                "subtract_col1_w_constant",
                "subtract_transform_array_col1",
            ),
            (
                "example_dataframe_w_multiple_numeric_nested_arrays",
                None,
                ["col1", "col2"],
                None,
                "subtract_col1_col2",
                "subtract_transform_array_col1_col2",
            ),
        ],
    )
    def test_spark_subtract_transform(
        self,
        input_dataframe,
        input_col,
        input_cols,
        float_constant,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = (
            SubtractTransformer(
                inputCol=input_col,
                outputCol=output_col,
                mathFloatConstant=float_constant,
            )
            if input_col is not None
            else SubtractTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_subtract_transform_defaults(self):
        # when
        subtract_transformer = SubtractTransformer()
        # then
        assert subtract_transformer.getLayerName() == subtract_transformer.uid
        assert (
            subtract_transformer.getOutputCol() == f"{subtract_transformer.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, float_constant",
        [
            (tf.constant([1.0, -5.0, -6.0, 7.0, -8.0, 9.0]), None, "string", 5.0),
            (
                tf.constant(["-4.0", "-5.0", "-3.0", "-47.0", "-8.2", "-11.0"]),
                "double",
                "float",
                -14.5,
            ),
            (tf.constant([1.0, -2.0, -3.0]), "smallint", "bigint", -3.0),
            (tf.constant([1.0, -2.0, -3.0]), None, None, -3.0),
            (tf.constant([-56.4, 55.4]), None, "double", 50.0),
            (tf.constant([-76.4, -55.4]), None, None, -25.0),
        ],
    )
    def test_subtract_transform_single_input_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, float_constant
    ):
        # given
        transformer = SubtractTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            mathFloatConstant=float_constant,
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
        if isinstance(spark_values[0], str):
            np.testing.assert_equal(
                spark_values,
                tensorflow_values,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
        else:
            np.testing.assert_almost_equal(
                spark_values,
                tensorflow_values,
                decimal=6,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype",
        [
            (
                [
                    tf.constant([1.0, -5.0, -6.0, 7.0, -8.0, 9.0], dtype=tf.float32),
                    tf.constant(
                        [10.0, -56.0, -6.0, 14.0, -3.2, 9.56], dtype=tf.float32
                    ),
                    tf.constant([1.0, -5.0, -3.0, 2.0, -4.25, -1.5], dtype=tf.float32),
                ],
                "double",
                "float",
            ),
            (
                [
                    tf.constant(
                        [-4.0, -5.0, -3.0, -47.0, -8.0, -11.0], dtype=tf.float32
                    ),
                    tf.constant(
                        [-2.0, -17.0, -3.45, -4.2, -0.1, -1.0], dtype=tf.float32
                    ),
                ],
                "float",
                None,
            ),
            (
                [
                    tf.constant([1.0, -2.0, -3.0], dtype=tf.float32),
                    tf.constant([1.45, -20.0, -3.23], dtype=tf.float32),
                ],
                None,
                None,
            ),
            (
                [
                    tf.constant([1.0, -2.0, -3.0], dtype=tf.float32),
                    tf.constant([1.0, -2.0, -3.0], dtype=tf.float32),
                ],
                "bigint",
                None,
            ),
        ],
    )
    def test_subtract_transform_multiple_input_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = SubtractTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
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
        if isinstance(spark_values[0], str):
            np.testing.assert_equal(
                spark_values,
                tensorflow_values,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
        else:
            np.testing.assert_almost_equal(
                spark_values,
                tensorflow_values,
                decimal=6,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
