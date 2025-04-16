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
from pyspark.sql.types import ArrayType, DoubleType, FloatType, StringType

from kamae.spark.transformers import LambdaFunctionTransformer


class TestLambdaFunction:
    @pytest.fixture(scope="class")
    def single_input_single_output_int_tf_function(self):
        def my_tf_fn(x):
            return tf.square(x)

        return my_tf_fn

    @pytest.fixture(scope="class")
    def single_input_single_output_float_tf_function(self):
        def my_tf_fn(x):
            return tf.square(x) - tf.math.log(x)

        return my_tf_fn

    @pytest.fixture(scope="class")
    def multiple_input_single_output_float_tf_function(self):
        def my_multi_tf_fn(x):
            x0 = x[0]
            x1 = x[1]

            return tf.square(x0) * tf.math.log(x1)

        return my_multi_tf_fn

    @pytest.fixture(scope="class")
    def single_input_multi_output_float_tf_function(self):
        def my_tf_fn(x):
            return [tf.math.add(x, 1.0), tf.math.subtract(x, 1.0)]

        return my_tf_fn

    @pytest.fixture(scope="class")
    def multiple_input_multi_output_float_tf_function(self):
        def my_multi_tf_fn(x):
            x0 = x[0]
            x1 = x[1]

            return [
                tf.square(x0),
                tf.concat([tf.square(x0), tf.math.log(x1)], axis=-1),
                tf.math.add(x0, x1),
            ]

        return my_multi_tf_fn

    @pytest.fixture(scope="class")
    def single_input_single_output_string_tf_function(self):
        def my_tf_fn(x):
            return tf.strings.regex_replace(x, "a", "b")

        return my_tf_fn

    @pytest.fixture(scope="class")
    def multiple_input_single_output_string_tf_function(self):
        def my_multi_tf_fn(x):
            x0 = x[0]
            x1 = x[1]

            return tf.strings.join([x0, x1], separator=" ")

        return my_multi_tf_fn

    @pytest.fixture(scope="class")
    def single_input_single_output_float_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], 1.0),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], 14.6137056350708),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], 47.0540885925293),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col1_tf_output",
            ],
        )

    @pytest.fixture(scope="class")
    def single_input_single_output_float_singleton_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0], [1.0]),
                (
                    4.0,
                    2.0,
                    6.0,
                    "b",
                    "c",
                    [4.0, 2.0],
                    [14.6137056350708, 3.3068528175354004],
                ),
                (
                    7.0,
                    8.0,
                    3.0,
                    "a",
                    "a",
                    [7.0, 8.0, 3.0],
                    [47.0540885925293, 61.92055892944336, 7.901387691497803],
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col1_col2_col3_tf_output",
            ],
        )

    @pytest.fixture(scope="class")
    def multiple_input_single_output_float_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], 0.6931471824645996),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], 11.090354919433594),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], 101.89263916015625),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col1_col2_tf_output",
            ],
        )

    @pytest.fixture(scope="class")
    def single_input_multiple_output_float_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], 2.0, 0.0),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], 5.0, 3.0),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], 8.0, 6.0),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col1_add_1_output",
                "col1_subtract_1_output",
            ],
        )

    @pytest.fixture(scope="class")
    def multiple_input_multiple_output_float_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "c",
                    [1.0, 2.0, 3.0],
                    1.0,
                    [1.0, 0.6931471824645996],
                    3.0,
                ),
                (
                    4.0,
                    2.0,
                    6.0,
                    "b",
                    "c",
                    [4.0, 2.0, 6.0],
                    16.0,
                    [16.0, 0.6931471824645996],
                    6.0,
                ),
                (
                    7.0,
                    8.0,
                    3.0,
                    "a",
                    "a",
                    [7.0, 8.0, 3.0],
                    49.0,
                    [49.0, 2.079441547393799],
                    15.0,
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col1_tf_square_output",
                "col1_square_col2_log_concat_tf_output",
                "col1_col2_add_tf_output",
            ],
        )

    @pytest.fixture(scope="class")
    def single_input_single_output_float_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1.0,
                    2.0,
                    3.0,
                    "a",
                    "c",
                    [1.0, 2.0, 3.0],
                    [1.0, 3.3068528175354004, 7.901387691497803],
                ),
                (
                    4.0,
                    2.0,
                    6.0,
                    "b",
                    "c",
                    [4.0, 2.0, 6.0],
                    [14.6137056350708, 3.3068528175354004, 34.2082405090332],
                ),
                (
                    7.0,
                    8.0,
                    3.0,
                    "a",
                    "a",
                    [7.0, 8.0, 3.0],
                    [47.0540885925293, 61.92055892944336, 7.901387691497803],
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col1_col2_col3_tf_output",
            ],
        )

    @pytest.fixture(scope="class")
    def single_input_single_output_string_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], "b"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], "b"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], "b"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_tf_output",
            ],
        )

    @pytest.fixture(scope="class")
    def multiple_input_single_output_string_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], "a c"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], "b c"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], "a a"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_col5_tf_output",
            ],
        )

    @pytest.fixture(scope="class")
    def single_input_single_output_string_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]],
                    [["b", "c", "c"], ["b", "c", "c"], ["b", "b", "b"]],
                ),
                (
                    4,
                    2,
                    6,
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                    [["b", "d", "c"], ["b", "t", "s"], ["x", "o", "p"]],
                ),
                (
                    7,
                    8,
                    3,
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                    [["l", "c", "c"], ["b", "h", "c"], ["b", "w", "b"]],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col4_tf_output"],
        )

    @pytest.mark.parametrize(
        "input_col, input_cols, function, function_return_type, output_col, output_cols, input_dataframe, expected_dataframe",
        [
            (
                "col1",
                None,
                "single_input_single_output_float_tf_function",
                [DoubleType()],
                "col1_tf_output",
                None,
                "example_dataframe",
                "single_input_single_output_float_expected",
            ),
            (
                "col1_col2_col3",
                None,
                "single_input_single_output_float_tf_function",
                [ArrayType(DoubleType())],
                "col1_col2_col3_tf_output",
                None,
                "example_dataframe_with_singleton_array",
                "single_input_single_output_float_singleton_array_expected",
            ),
            (
                None,
                ["col1", "col2"],
                "multiple_input_single_output_float_tf_function",
                [DoubleType()],
                "col1_col2_tf_output",
                None,
                "example_dataframe",
                "multiple_input_single_output_float_expected",
            ),
            (
                "col1_col2_col3",
                None,
                "single_input_single_output_float_tf_function",
                [ArrayType(DoubleType())],
                "col1_col2_col3_tf_output",
                None,
                "example_dataframe",
                "single_input_single_output_float_array_expected",
            ),
            (
                "col4",
                None,
                "single_input_single_output_string_tf_function",
                [StringType()],
                "col4_tf_output",
                None,
                "example_dataframe",
                "single_input_single_output_string_expected",
            ),
            (
                None,
                ["col4", "col5"],
                "multiple_input_single_output_string_tf_function",
                [StringType()],
                "col4_col5_tf_output",
                None,
                "example_dataframe",
                "multiple_input_single_output_string_expected",
            ),
            (
                "col4",
                None,
                "single_input_single_output_string_tf_function",
                [ArrayType(ArrayType(StringType()))],
                "col4_tf_output",
                None,
                "example_index_input_with_string_arrays",
                "single_input_single_output_string_array_expected",
            ),
            (
                "col1",
                None,
                "single_input_multi_output_float_tf_function",
                [DoubleType(), DoubleType()],
                None,
                ["col1_add_1_output", "col1_subtract_1_output"],
                "example_dataframe",
                "single_input_multiple_output_float_expected",
            ),
            (
                None,
                ["col1", "col2"],
                "multiple_input_multi_output_float_tf_function",
                [DoubleType(), ArrayType(DoubleType()), DoubleType()],
                None,
                [
                    "col1_tf_square_output",
                    "col1_square_col2_log_concat_tf_output",
                    "col1_col2_add_tf_output",
                ],
                "example_dataframe",
                "multiple_input_multiple_output_float_expected",
            ),
        ],
    )
    def test_spark_lambda_function_transform(
        self,
        input_dataframe,
        input_col,
        input_cols,
        function,
        function_return_type,
        output_col,
        output_cols,
        expected_dataframe,
        request,
    ):
        # given
        lambda_fn = request.getfixturevalue(function)
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = LambdaFunctionTransformer(
            function=lambda_fn,
            functionReturnTypes=function_return_type,
        )
        if input_col is not None:
            transformer.setInputCol(input_col)
        else:
            transformer.setInputCols(input_cols)
        if output_col is not None:
            transformer.setOutputCol(output_col)
        else:
            transformer.setOutputCols(output_cols)

        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_lambda_function_transform_defaults(self):
        # when
        lambda_function_transformer = LambdaFunctionTransformer()
        # then
        assert (
            lambda_function_transformer.getLayerName()
            == lambda_function_transformer.uid
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, function, function_return_types",
        [
            (
                tf.constant(["1.0", "5.0", "6.0", "7.0", "8.0", "9.0"]),
                "float",
                "string",
                "single_input_single_output_int_tf_function",
                [DoubleType()],
            ),
            (
                tf.constant([4.0, 5.0, 3.0, 47.0, 8.2, 11.0]),
                "double",
                None,
                "single_input_single_output_float_tf_function",
                [FloatType()],
            ),
            (
                tf.constant([-56.4, 55.4]),
                None,
                None,
                "single_input_single_output_float_tf_function",
                [FloatType()],
            ),
            (
                tf.constant(["a", "hello", "a", "b"]),
                None,
                None,
                "single_input_single_output_string_tf_function",
                [StringType()],
            ),
        ],
    )
    def test_lambda_function_transform_single_input_single_output_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        function,
        function_return_types,
        request,
    ):
        # given
        function = request.getfixturevalue(function)
        transformer = LambdaFunctionTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            function=function,
            functionReturnTypes=function_return_types,
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
                decimal=4,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, function, function_return_types",
        [
            (
                [
                    tf.constant([1.0, -5.0, -6.0, 7.0, -8.0, 9.0], dtype=tf.float32),
                    tf.constant([10.0, 56.0, 6.0, 14.0, 3.2, 9.56], dtype=tf.float32),
                ],
                "float",
                "double",
                "multiple_input_single_output_float_tf_function",
                [DoubleType()],
            ),
            (
                [
                    tf.constant(
                        [-4.0, -5.0, -3.0, -47.0, -8.0, -11.0], dtype=tf.float32
                    ),
                    tf.constant([2.0, 17.0, 3.45, 4.2, 0.1, 1.0], dtype=tf.float32),
                ],
                "double",
                None,
                "multiple_input_single_output_float_tf_function",
                [FloatType()],
            ),
            (
                [
                    tf.constant(["a", "b", "c"], dtype=tf.string),
                    tf.constant(["f", "e", "d"], dtype=tf.string),
                ],
                None,
                None,
                "multiple_input_single_output_string_tf_function",
                [StringType()],
            ),
        ],
    )
    def test_lambda_function_transform_multiple_input_single_output_spark_tf_parity(
        self,
        spark_session,
        input_tensors,
        input_dtype,
        output_dtype,
        function,
        function_return_types,
        request,
    ):
        function = request.getfixturevalue(function)
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = LambdaFunctionTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            function=function,
            functionReturnTypes=function_return_types,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                tuple(
                    [
                        ti.numpy().decode("utf-8")
                        if isinstance(function_return_types[0], StringType)
                        else float(ti.numpy())
                        for ti in t
                    ]
                )
                for t in zip(*input_tensors)
            ],
            col_names,
            function_return_types[0],
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
                [t.decode("utf-8") for t in tensorflow_values],
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
        else:
            np.testing.assert_almost_equal(
                spark_values,
                tensorflow_values,
                decimal=4,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, function, function_return_types",
        [
            (
                tf.constant([1.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=tf.float32),
                "float",
                "int",
                "single_input_multi_output_float_tf_function",
                [DoubleType(), DoubleType()],
            ),
        ],
    )
    def test_lambda_function_transform_single_input_multiple_output_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        function,
        function_return_types,
        request,
    ):
        function = request.getfixturevalue(function)
        output_col_names = [f"output{i}" for i in range(len(function_return_types))]
        # given
        transformer = LambdaFunctionTransformer(
            inputCol="input",
            outputCols=output_col_names,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            function=function,
            functionReturnTypes=function_return_types,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [(v,) for v in input_tensor.numpy().tolist()], ["input"], [DoubleType()]
        )

        spark_values = [
            transformer.transform(spark_df).select(c).rdd.map(lambda r: r[0]).collect()
            for c in output_col_names
        ]
        tensorflow_values = [
            v.numpy().tolist() for v in transformer.get_tf_layer()(input_tensor)
        ]

        # then
        for spark_output, tensorflow_output in zip(spark_values, tensorflow_values):
            print(spark_output)
            print(tensorflow_output)
            np.testing.assert_almost_equal(
                np.array(spark_output).flatten(),
                np.array(tensorflow_output).flatten(),
                decimal=4,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, function, function_return_types",
        [
            (
                [
                    tf.constant(
                        [[1.0], [-5.0], [-6.0], [7.0], [-8.0], [9.0]], dtype=tf.float32
                    ),
                    tf.constant(
                        [[10.0], [56.0], [6.0], [14.0], [3.2], [9.56]], dtype=tf.float32
                    ),
                ],
                "float",
                "double",
                "multiple_input_multi_output_float_tf_function",
                [DoubleType(), ArrayType(DoubleType()), DoubleType()],
            )
        ],
    )
    def test_lambda_function_transform_multiple_input_multiple_output_spark_tf_parity(
        self,
        spark_session,
        input_tensors,
        input_dtype,
        output_dtype,
        function,
        function_return_types,
        request,
    ):
        function = request.getfixturevalue(function)
        input_col_names = [f"input{i}" for i in range(len(input_tensors))]
        output_col_names = [f"output{i}" for i in range(len(function_return_types))]
        # given
        transformer = LambdaFunctionTransformer(
            inputCols=input_col_names,
            outputCols=output_col_names,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            function=function,
            functionReturnTypes=function_return_types,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [tuple([float(ti[0].numpy()) for ti in t]) for t in zip(*input_tensors)],
            input_col_names,
            [DoubleType() for _ in range(len(input_tensors))],
        )

        spark_values = [
            transformer.transform(spark_df).select(c).rdd.map(lambda r: r[0]).collect()
            for c in output_col_names
        ]
        tensorflow_values = [
            v.numpy().tolist() for v in transformer.get_tf_layer()(input_tensors)
        ]

        # then
        for spark_output, tensorflow_output in zip(spark_values, tensorflow_values):
            np.testing.assert_almost_equal(
                np.array(spark_output).flatten(),
                np.array(tensorflow_output).flatten(),
                decimal=4,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
