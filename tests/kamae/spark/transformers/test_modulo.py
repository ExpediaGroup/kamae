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
from pyspark.sql.types import DoubleType, IntegerType

from kamae.spark.transformers import ModuloTransformer

from ..test_helpers import tensor_to_python_type


class TestModulo:
    @pytest.fixture(scope="class")
    def example_dataframe_with_nice_modulos(self, spark_session):
        return spark_session.createDataFrame(
            [
                (10, 5),
                (31, 6),
                (56, 9),
            ],
            ["col1", "col2"],
        )

    @pytest.fixture(scope="class")
    def modulo_transform_expected_col1_mod_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (10, 5, 2),
                (31, 6, 3),
                (56, 9, 0),
            ],
            ["col1", "col2", "modulo_col1_w_constant"],
        )

    @pytest.fixture(scope="class")
    def modulo_transform_expected_col2_mod_7(self, spark_session):
        return spark_session.createDataFrame(
            [
                (10, 5, 5),
                (31, 6, 6),
                (56, 9, 2),
            ],
            ["col1", "col2", "modulo_col2_w_constant"],
        )

    @pytest.fixture(scope="class")
    def modulo_transform_expected_col1_col2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (10, 5, 0),
                (31, 6, 1),
                (56, 9, 2),
            ],
            ["col1", "col2", "modulo_col1_col2"],
        )

    @pytest.fixture(scope="class")
    def mod_transform_array_col1(self, spark_session):
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
                        [1.0, 1.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [1.0, 2.0, 0.0],
                    ],
                )
            ],
            ["col1", "col2", "mod_col1_w_constant"],
        )

    @pytest.fixture(scope="class")
    def mod_transform_array_col1_col2(self, spark_session):
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
                        [1.0, 3.0, 0.6000000000000001],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 6.5],
                        [4.0, 0.0, 3.0],
                    ],
                )
            ],
            ["col1", "col2", "mod_col1_col2"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, input_cols, divisor, output_col, expected_dataframe",
        [
            (
                "example_dataframe_with_nice_modulos",
                "col1",
                None,
                4,
                "modulo_col1_w_constant",
                "modulo_transform_expected_col1_mod_4",
            ),
            (
                "example_dataframe_with_nice_modulos",
                "col2",
                None,
                7,
                "modulo_col2_w_constant",
                "modulo_transform_expected_col2_mod_7",
            ),
            (
                "example_dataframe_with_nice_modulos",
                None,
                ["col1", "col2"],
                None,
                "modulo_col1_col2",
                "modulo_transform_expected_col1_col2",
            ),
            (
                "example_dataframe_w_multiple_numeric_nested_arrays",
                "col1",
                None,
                3,
                "mod_col1_w_constant",
                "mod_transform_array_col1",
            ),
            (
                "example_dataframe_w_multiple_numeric_nested_arrays",
                None,
                ["col1", "col2"],
                None,
                "mod_col1_col2",
                "mod_transform_array_col1_col2",
            ),
        ],
    )
    def test_spark_modulo_transform(
        self,
        input_dataframe,
        input_col,
        input_cols,
        divisor,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = (
            ModuloTransformer(
                inputCol=input_col,
                outputCol=output_col,
                divisor=divisor,
            )
            if input_col is not None
            else ModuloTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_modulo_transform_defaults(self):
        # when
        modulo_transformer = ModuloTransformer()
        # then
        assert modulo_transformer.getLayerName() == modulo_transformer.uid
        assert modulo_transformer.getOutputCol() == f"{modulo_transformer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, divisor",
        [
            (tf.constant(["1", "-5", "-6", "7", "-8", "9"]), "bigint", None, 5),
            (tf.constant([-4, -5, -3, -47, -8, -11]), "double", "int", 12),
            (tf.constant([1, -2, -3]), "bigint", None, 3),
            (tf.constant([1, -2, -3]), None, None, 4),
            (tf.constant([-56, 55]), "int", "double", 5),
            (tf.constant([-76, -55]), None, None, 21),
        ],
    )
    def test_modulo_transform_single_input_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, divisor
    ):
        # given
        transformer = ModuloTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            divisor=divisor,
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
                    tf.constant(["10", "-5", "-6", "7", "-8", "9"]),
                    tf.constant([10, 2, 1, 14, 3, 9], dtype=tf.int32),
                ],
                "double",
                None,
            ),
            (
                [
                    tf.constant(["1", "-5", "-6", "7", "-8", "9"]),
                    tf.constant([1, 2, 1, 4, 3, 9], dtype=tf.int32),
                ],
                "bigint",
                "int",
            ),
            (
                [
                    tf.constant([1.0, -5.0, -6.0, 7.0, -8.0, 9.0]),
                    tf.constant([1, 2.1, 1.5, 4.3, 3.2, 9.0]),
                ],
                None,
                None,
            ),
        ],
    )
    def test_modulo_transform_multiple_input_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = ModuloTransformer(
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
