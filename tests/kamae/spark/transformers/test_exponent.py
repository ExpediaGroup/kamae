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

from kamae.spark.transformers import ExponentTransformer


class TestExponent:
    @pytest.fixture(scope="class")
    def example_dataframe_nested_data(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]]),
                (4.0, 2.0, [[4.0, 2.0, 5.0], [4.0, 2.0, 5.0]]),
                (7.0, 2.0, [[7.0, 8.0, 2.0], [7.0, 8.0, 2.0]]),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def exponent_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]], 1.0),
                (4.0, 2.0, [[4.0, 2.0, 5.0], [4.0, 2.0, 5.0]], 1024.0),
                (7.0, 2.0, [[7.0, 8.0, 2.0], [7.0, 8.0, 2.0]], 16807.0),
            ],
            ["col1", "col2", "col3", "exponent_col1_w_constant"],
        )

    @pytest.fixture(scope="class")
    def exponent_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]],
                    [[2.0, 4.0, 4.0], [2.0, 4.0, 4.0]],
                ),
                (
                    4.0,
                    2.0,
                    [[4.0, 2.0, 5.0], [4.0, 2.0, 5.0]],
                    [[16.0, 4.0, 32.0], [16.0, 4.0, 32.0]],
                ),
                (
                    7.0,
                    2.0,
                    [[7.0, 8.0, 2.0], [7.0, 8.0, 2.0]],
                    [[128.0, 256.0, 4.0], [128.0, 256.0, 4.0]],
                ),
            ],
            ["col1", "col2", "col3", "exponent_col2_col3"],
        )

    @pytest.fixture(scope="class")
    def exponent_transform_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]], 0.5),
                (4.0, 2.0, [[4.0, 2.0, 5.0], [4.0, 2.0, 5.0]], 0.5),
                (7.0, 2.0, [[7.0, 8.0, 2.0], [7.0, 8.0, 2.0]], 0.5),
            ],
            ["col1", "col2", "col3", "exponent_col2_w_constant"],
        )

    @pytest.mark.parametrize(
        "input_col, input_cols, exponent, output_col, expected_dataframe",
        [
            (
                "col1",
                None,
                5,
                "exponent_col1_w_constant",
                "exponent_transform_expected_1",
            ),
            (
                None,
                ["col2", "col3"],
                None,
                "exponent_col2_col3",
                "exponent_transform_expected_2",
            ),
            (
                "col2",
                None,
                -1,
                "exponent_col2_w_constant",
                "exponent_transform_expected_4",
            ),
        ],
    )
    def test_spark_exponent_transform(
        self,
        example_dataframe_nested_data,
        input_col,
        input_cols,
        exponent,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = (
            ExponentTransformer(
                inputCol=input_col,
                outputCol=output_col,
                exponent=exponent,
            )
            if input_col is not None
            else ExponentTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(example_dataframe_nested_data)
        actual.show(20, False)
        expected.show(20, False)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_exponent_transform_defaults(self):
        # when
        exponent_transformer = ExponentTransformer()
        # then
        assert exponent_transformer.getLayerName() == exponent_transformer.uid
        assert (
            exponent_transformer.getOutputCol() == f"{exponent_transformer.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, float_constant",
        [
            (
                tf.constant(["1.0", "-5.0", "-6.0", "7.0", "-8.0", "9.0"]),
                "float",
                "string",
                2.0,
            ),
            (
                tf.constant([-4.0, -5.0, -3.0, -4.0, -8.2, -11.0]),
                "double",
                None,
                -1.6,
            ),
            (tf.constant([1, -2, -3], dtype="int32"), "float", "string", 2.0),
            (tf.constant([1.0, -2.0, -3.0]), None, "bigint", -3),
            (tf.constant([-1.0, 2.0]), None, None, 5.0),
            (tf.constant([-6.4, -5.4]), None, "double", -2.0),
        ],
    )
    def test_exponent_transform_single_input_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, float_constant
    ):
        # given
        transformer = ExponentTransformer(
            inputCol="input",
            outputCol="output",
            exponent=float_constant,
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
                decimal=5,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype",
        [
            (
                [
                    tf.constant([1.0, -5.0, -6.0, 7.0, -8.0, 9.0], dtype=tf.float32),
                    tf.constant([10.0, -5.0, 0.0, 1.0, -3.2, 5.0], dtype=tf.float32),
                ],
                "float",
                "double",
            ),
            (
                [
                    tf.constant([-4.0, -5.0, -3.0, -4.0, -8.0, -1.0], dtype=tf.float32),
                    tf.constant(
                        [-2.0, -1.0, -3.45, -4.2, -0.1, -1.0], dtype=tf.float32
                    ),
                ],
                "double",
                None,
            ),
            (
                [
                    tf.constant([1.0, -2.0, -3.0], dtype=tf.float32),
                    tf.constant([1.45, -2.0, -3.23], dtype=tf.float32),
                ],
                None,
                None,
            ),
            (
                [
                    tf.constant([1.0, -2.0, -3.0], dtype=tf.float32),
                    tf.constant([1.0, -2.0, -3.0], dtype=tf.float32),
                ],
                "float",
                None,
            ),
        ],
    )
    def test_exponent_transform_multiple_input_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = ExponentTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [tuple([float(ti.numpy()) for ti in t]) for t in zip(*input_tensors)],
            col_names,
            DoubleType(),
        )

        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = transformer.get_tf_layer()(input_tensors).numpy().tolist()

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
