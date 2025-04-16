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

from kamae.spark.transformers import ArrayCropTransformer


class TestArrayCrop:
    @pytest.fixture(scope="class")
    def array_crop_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    ["a", "a", "a", "b", "c"],
                    ["a", "a", "a"],
                ),
                (
                    4,
                    ["x", "z", "y"],
                    ["x", "z", "y"],
                ),
                (
                    7,
                    ["a", "b"],
                    ["a", "b", "-1"],
                ),
                (
                    1,
                    ["a", "x", "a", "b"],
                    ["a", "x", "a"],
                ),
                (
                    7,
                    [],
                    ["-1", "-1", "-1"],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def array_crop_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    [1, 2, 3, 4, 5],
                    [1, 2, 3],
                ),
                (
                    4,
                    [6, 7, 8],
                    [6, 7, 8],
                ),
                (
                    7,
                    [1, 2],
                    [1, 2, -1],
                ),
                (
                    7,
                    [],
                    [-1, -1, -1],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def array_crop_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 2.0, 3.0],
                ),
                (
                    4,
                    [6.0, 7.0, 8.0],
                    [6.0, 7.0, 8.0],
                ),
                (
                    7,
                    [1.0, 2.0],
                    [1.0, 2.0, -1.0],
                ),
                (
                    7,
                    [],
                    [-1.0, -1.0, -1.0],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def array_crop_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0],
                ),
                (
                    4,
                    [6.0, 7.0, 8.0],
                    [6.0],
                ),
                (
                    7,
                    [1.0, 2.0],
                    [1.0],
                ),
                (
                    7,
                    [],
                    [-1.0],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def array_crop_expected_5(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, [["-1", "a", "b", "-1"]], [["-1", "a", "b"]]),
                (4, [["a", "a", "b", "c"]], [["a", "a", "b"]]),
                (7, [["b", "b", "b", "a"]], [["b", "b", "b"]]),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def array_crop_expected_6(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    [["-1", "a", "b", "-1"]],
                    [["-1", "a", "b", "-1", "pad"]],
                ),
                (
                    4,
                    [["a", "a", "b", "c"]],
                    [["a", "a", "b", "c", "pad"]],
                ),
                (
                    7,
                    [["b", "b", "b", "a"]],
                    [["b", "b", "b", "a", "pad"]],
                ),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.fixture(scope="class")
    def array_crop_expected_7(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    [
                        1687087026136,
                        1687087026136,
                        1687087026136,
                        1687087026136,
                        1687087026136,
                    ],
                    [1687087026136, 1687087026136, 1687087026136],
                ),
                (
                    4,
                    [1687087026136, 1687087026136, 1687087026136],
                    [1687087026136, 1687087026136, 1687087026136],
                ),
                (7, [1687087026136, 1687087026136], [1687087026136, 1687087026136, -1]),
                (7, [], [-1, -1, -1]),
            ],
            ["col1", "col2", "col2_diff"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, input_dtype, output_dtype, array_length, pad_value, expected_dataframe",
        [
            (
                "example_dataframe_with_ragged_string_array",
                "col2",
                "col2_diff",
                "string",
                "string",
                3,
                "-1",
                "array_crop_expected_1",
            ),
            (
                "example_dataframe_with_ragged_int_array",
                "col2",
                "col2_diff",
                "int",
                "int",
                3,
                -1,
                "array_crop_expected_2",
            ),
            (
                "example_dataframe_with_ragged_float_array",
                "col2",
                "col2_diff",
                "float",
                "float",
                3,
                -1.0,
                "array_crop_expected_3",
            ),
            (
                "example_dataframe_with_ragged_float_array",
                "col2",
                "col2_diff",
                "float",
                "float",
                1,
                -1.0,
                "array_crop_expected_4",
            ),
            (
                "example_dataframe_with_nested_string_array",
                "col2",
                "col2_diff",
                "string",
                "string",
                3,
                "-1",
                "array_crop_expected_5",
            ),
            (
                "example_dataframe_with_nested_string_array",
                "col2",
                "col2_diff",
                "string",
                "string",
                5,
                "pad",
                "array_crop_expected_6",
            ),
            (
                "example_dataframe_with_long_array",
                "col2",
                "col2_diff",
                "bigint",
                "bigint",
                3,
                -1,
                "array_crop_expected_7",
            ),
        ],
    )
    def test_spark_array_crop_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        input_dtype,
        output_dtype,
        array_length,
        pad_value,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        array_crop_transformer = ArrayCropTransformer(
            inputCol=input_col,
            outputCol=output_col,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            arrayLength=array_length,
            padValue=pad_value,
        )
        actual = array_crop_transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)

        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, input_dtype, output_dtype, array_length, pad_value",
        [
            (
                "example_dataframe_with_ragged_string_array",
                "col2",
                "col2_diff",
                "string",
                "string",
                3,
                -1,
            ),
            (
                "example_dataframe_with_ragged_int_array",
                "col2",
                "col2_diff",
                "int",
                "int",
                3,
                "-1",
            ),
        ],
    )
    def test_spark_array_crop_transform_raises_error_with_mismatching_input_types(
        self,
        input_dataframe,
        input_col,
        output_col,
        input_dtype,
        output_dtype,
        array_length,
        pad_value,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        array_crop_transformer = ArrayCropTransformer(
            inputCol=input_col,
            outputCol=output_col,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            arrayLength=array_length,
            padValue=pad_value,
        )
        # then
        with pytest.raises(ValueError):
            _ = array_crop_transformer.transform(input_dataframe)

    @pytest.mark.parametrize(
        "array_length, pad_value",
        [
            (
                -1,
                -1,
            ),
            (
                0,
                "-1",
            ),
        ],
    )
    def test_spark_array_crop_transform_raises_error_with_invalid_array_length(
        self,
        array_length,
        pad_value,
    ):
        # then
        with pytest.raises(ValueError):
            _ = ArrayCropTransformer(
                inputCol="foo",
                outputCol="bar",
                inputDtype="int",
                outputDtype="int",
                arrayLength=array_length,
                padValue=pad_value,
            )

    @pytest.mark.parametrize(
        "input_tensor, dtype, array_length, pad_value",
        [
            (
                tf.constant([["a", "a", "b", "c"]]),
                "string",
                3,
                "-1",
            ),
            (
                tf.constant([["a", "a", "b", "c"]]),
                "string",
                5,
                "-1",
            ),
            (
                tf.constant([[1, 2, 3, 4]]),
                "int",
                3,
                -1,
            ),
            (
                tf.constant([[1, 2, 3, 4]]),
                "int",
                5,
                -1,
            ),
            (
                tf.constant([["a", "a", "b", "c"], ["a", "a", "x", "y"]]),
                "string",
                3,
                "-1",
            ),
            (
                tf.constant([["a", "a", "b", "c"], ["a", "a", "x", "y"]]),
                "string",
                5,
                "-1",
            ),
            (
                tf.constant([[1, 2, 3, 4], [9, 8, 1, 0]]),
                "int",
                3,
                -1,
            ),
            (
                tf.constant([[1, 2, 3, 4], [9, 8, 1, 0]]),
                "int",
                5,
                -1,
            ),
            (
                tf.constant([[1.0, 2.0, 3.0, 4.0], [9.0, 8.0, 1.0, 0.0]]),
                "int",
                5,
                -1,
            ),
            (
                tf.constant(
                    [
                        [1687087026136, 1687087026136, 1687087026136],
                        [1687087026136, 1687087026136, 1687087026136],
                    ]
                ),
                "bigint",
                5,
                -1,
            ),
        ],
    )
    def test_array_crop_spark_tf_parity(
        self, spark_session, input_tensor, dtype, array_length, pad_value
    ):
        # given
        transformer = ArrayCropTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=dtype,
            outputDtype=dtype,
            arrayLength=array_length,
            padValue=pad_value,
        )

        array_decoder = np.vectorize(lambda x: x.decode("utf-8"))

        if input_tensor.dtype.name == "string":
            spark_df = spark_session.createDataFrame(
                [(array_decoder(v).tolist(),) for v in input_tensor.numpy().tolist()],
                ["input"],
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
            array_decoder(v) if isinstance(v[0], bytes) else v
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
