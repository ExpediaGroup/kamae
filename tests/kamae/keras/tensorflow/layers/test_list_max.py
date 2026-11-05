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

from kamae.keras.tensorflow.layers import ListMaxLayer


class TestListMax:
    @pytest.mark.parametrize(
        "inputs, min_filter_value, top_n, with_segment, sort_order, input_dtype, output_dtype, expected_output",
        [
            # Base case
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1.0],
                                [1.0],
                                [9.0],
                                [4.0],
                                [6.0],
                                [2.0],
                                [0.0],
                                [0.0],
                            ],
                            [
                                [5.0],
                                [1.0],
                                [9.0],
                                [4.0],
                                [6.0],
                                [8.0],
                                [0.0],
                                [0.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                None,
                None,
                False,
                "asc",
                "float64",
                "float32",
                # values
                tf.constant(
                    [
                        [
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                        ],
                        [
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With min_filter_value
            (
                [
                    tf.constant(
                        [
                            [
                                [1.0],
                                [1.0],
                                [9.0],
                                [4.0],
                                [6.0],
                                [2.0],
                                [0.0],
                                [0.0],
                            ],
                            [
                                [5.0],
                                [1.0],
                                [9.0],
                                [4.0],
                                [6.0],
                                [8.0],
                                [0.0],
                                [0.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                1,
                None,
                False,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                        ],
                        [
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With top_n
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1.0],
                                [1.0],
                                [9.0],
                                [4.0],
                                [6.0],
                                [2.0],
                                [0.0],
                                [0.0],
                            ],
                            [
                                [5.0],
                                [1.0],
                                [9.0],
                                [4.0],
                                [6.0],
                                [8.0],
                                [0.0],
                                [0.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                    # sort
                    tf.constant(
                        [
                            [
                                [1.0],
                                [2.0],
                                [3.0],
                                [4.0],
                                [5.0],
                                [6.0],
                                [7.0],
                                [8.0],
                            ],
                            [
                                [8.0],
                                [7.0],
                                [6.0],
                                [5.0],
                                [4.0],
                                [3.0],
                                [2.0],
                                [1.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                None,
                5,
                False,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                        ],
                        [
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With top_n and filter
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1.0],
                                [1.0],
                                [9.0],
                                [4.0],
                                [6.0],
                                [2.0],
                                [0.0],
                                [0.0],
                            ],
                            [
                                [5.0],
                                [1.0],
                                [9.0],
                                [4.0],
                                [6.0],
                                [8.0],
                                [0.0],
                                [0.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                    # sort
                    tf.constant(
                        [
                            [
                                [1.0],
                                [2.0],
                                [3.0],
                                [4.0],
                                [5.0],
                                [6.0],
                                [7.0],
                                [8.0],
                            ],
                            [
                                [8.0],
                                [7.0],
                                [6.0],
                                [5.0],
                                [4.0],
                                [3.0],
                                [2.0],
                                [1.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                1,
                5,
                False,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                            [9.0],
                        ],
                        [
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                            [8.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With top_n > list size
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1.0],
                                [1.0],
                                [9.0],
                            ],
                            [
                                [5.0],
                                [1.0],
                                [9.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                    # sort
                    tf.constant(
                        [
                            [
                                [1.0],
                                [2.0],
                                [3.0],
                            ],
                            [
                                [8.0],
                                [7.0],
                                [6.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                1,
                5,
                False,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [9.0],
                            [9.0],
                            [9.0],
                        ],
                        [
                            [9.0],
                            [9.0],
                            [9.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With segmentation
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1.0],
                                [1.0],
                                [9.0],
                            ],
                            [
                                [5.0],
                                [1.0],
                                [9.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                    # segment
                    tf.constant(
                        [
                            [
                                [1.0],
                                [2.0],
                                [2.0],
                            ],
                            [
                                [1.0],
                                [2.0],
                                [2.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                None,
                None,
                True,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [1.0],
                            [9.0],
                            [9.0],
                        ],
                        [
                            [5.0],
                            [9.0],
                            [9.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With segmentation and multiple features
            (
                [
                    # values
                    tf.constant(
                        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]], dtype=tf.float32
                    ),
                    # segment
                    tf.constant(
                        [[[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]], dtype=tf.float32
                    ),
                ],
                None,
                None,
                True,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [[[1.0, 10.0], [3.0, 30.0], [3.0, 30.0]]], dtype=tf.float32
                ),
            ),
            # With segmentation ID as string
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1.0],
                                [1.0],
                                [9.0],
                            ],
                            [
                                [5.0],
                                [1.0],
                                [9.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                    # segment
                    tf.constant(
                        [
                            [
                                ["1.0"],
                                ["2.0"],
                                ["2.0"],
                            ],
                            [
                                ["1.0"],
                                ["2.0"],
                                ["2.0"],
                            ],
                        ],
                        dtype=tf.string,
                    ),
                ],
                None,
                None,
                True,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [1.0],
                            [9.0],
                            [9.0],
                        ],
                        [
                            [5.0],
                            [9.0],
                            [9.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With segmentation and min_filter_val
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1.0],
                                [1.0],
                                [9.0],
                            ],
                            [
                                [5.0],
                                [1.0],
                                [9.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                    # segment
                    tf.constant(
                        [
                            [
                                [1.0],
                                [2.0],
                                [2.0],
                            ],
                            [
                                [1.0],
                                [2.0],
                                [2.0],
                            ],
                        ],
                        dtype=tf.float32,
                    ),
                ],
                2.0,
                None,
                True,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [0.0],
                            [9.0],
                            [9.0],
                        ],
                        [
                            [5.0],
                            [9.0],
                            [9.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # Integer values with segmentation
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1],
                                [5],
                                [3],
                            ],
                            [
                                [7],
                                [2],
                                [9],
                            ],
                        ],
                        dtype=tf.int64,
                    ),
                    # segment
                    tf.constant(
                        [
                            [
                                [1],
                                [1],
                                [2],
                            ],
                            [
                                [1],
                                [1],
                                [2],
                            ],
                        ],
                        dtype=tf.int64,
                    ),
                ],
                None,
                None,
                True,
                "asc",
                "int64",
                "int64",
                tf.constant(
                    [
                        [
                            [5],
                            [5],
                            [3],
                        ],
                        [
                            [7],
                            [7],
                            [9],
                        ],
                    ],
                    dtype=tf.int64,
                ),
            ),
            # Integer values with segmentation and min_filter_value, where the
            # filter empties a segment entirely and nan_fill_value is applied.
            (
                [
                    # values
                    tf.constant(
                        [
                            [
                                [1],
                                [1],
                                [9],
                            ],
                            [
                                [5],
                                [1],
                                [9],
                            ],
                        ],
                        dtype=tf.int64,
                    ),
                    # segment
                    tf.constant(
                        [
                            [
                                [1],
                                [2],
                                [2],
                            ],
                            [
                                [1],
                                [2],
                                [2],
                            ],
                        ],
                        dtype=tf.int64,
                    ),
                ],
                2,
                None,
                True,
                "asc",
                "int64",
                "int64",
                tf.constant(
                    [
                        [
                            [0],
                            [9],
                            [9],
                        ],
                        [
                            [5],
                            [9],
                            [9],
                        ],
                    ],
                    dtype=tf.int64,
                ),
            ),
            # Remaining integer widths, exercising segmentation together with
            # min_filter_value emptying a segment, so that nan_fill_value is
            # applied for every integer dtype the layer accepts.
            *[
                (
                    [
                        # values
                        tf.constant(
                            [
                                [
                                    [1],
                                    [1],
                                    [9],
                                ],
                                [
                                    [5],
                                    [1],
                                    [9],
                                ],
                            ],
                            dtype=int_dtype,
                        ),
                        # segment
                        tf.constant(
                            [
                                [
                                    [1],
                                    [2],
                                    [2],
                                ],
                                [
                                    [1],
                                    [2],
                                    [2],
                                ],
                            ],
                            dtype=int_dtype,
                        ),
                    ],
                    2,
                    None,
                    True,
                    "asc",
                    int_dtype.name,
                    int_dtype.name,
                    tf.constant(
                        [
                            [
                                [0],
                                [9],
                                [9],
                            ],
                            [
                                [5],
                                [9],
                                [9],
                            ],
                        ],
                        dtype=int_dtype,
                    ),
                )
                for int_dtype in [tf.int8, tf.int16, tf.int32]
            ],
        ],
    )
    def test_listwise_max(
        self,
        inputs,
        min_filter_value,
        top_n,
        with_segment,
        sort_order,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        name = "listwise_max_test"
        layer = ListMaxLayer(
            name=name,
            min_filter_value=min_filter_value,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            sort_order=sort_order,
            top_n=top_n,
            with_segment=with_segment,
        )
        inputs = inputs if len(inputs) > 1 else inputs[0]
        output_tensor = layer(inputs)
        # then
        assert layer.name == name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(output_tensor, expected_output)

    @pytest.mark.parametrize(
        "dtype, nan_fill_value, expected_fill",
        [
            (tf.float64, 0.1, 0.1),
            (tf.float64, 123.456, 123.456),
            (tf.float32, 0.1, np.float32(0.1)),
            (tf.int64, 7.0, 7),
            (tf.int32, 7.0, 7),
        ],
    )
    def test_listwise_max_nan_fill_value_dtype(
        self, dtype, nan_fill_value, expected_fill
    ):
        """The fill value applied to a segment emptied by min_filter_value must
        keep full precision on float dtypes and be usable on integer dtypes."""
        # given, a segment whose values are all removed by the filter
        values = tf.constant([[[1], [1], [9]]], dtype=dtype)
        segments = tf.constant([[[1], [2], [2]]], dtype=dtype)
        layer = ListMaxLayer(
            name="listwise_max_nan_fill_test",
            min_filter_value=5,
            with_segment=True,
            nan_fill_value=nan_fill_value,
            input_dtype=dtype.name,
            output_dtype=dtype.name,
        )
        # when
        output_tensor = layer([values, segments])
        # then
        assert (
            output_tensor.numpy().flatten()[0] == expected_fill
        ), "Emptied segment was not filled with the exact nan_fill_value"

    @pytest.mark.parametrize("with_segment", [False, True])
    @pytest.mark.parametrize("dtype", [tf.int8, tf.int16, tf.int32, tf.int64])
    def test_listwise_max_dtype_minimum_is_a_valid_value(self, dtype, with_segment):
        """The dtype minimum is ordinary data on the narrow integer types, -128 for
        int8, so a list whose genuine maximum is that value must be returned as-is
        rather than mistaken for a list emptied by the filter."""
        # given, a list whose real maximum is the dtype minimum, kept by the filter
        dtype_min = dtype.min
        values = tf.constant([[[dtype_min], [dtype_min], [dtype_min]]], dtype=dtype)
        layer = ListMaxLayer(
            name="listwise_max_dtype_min_test",
            min_filter_value=dtype_min,
            nan_fill_value=0.0,
            with_segment=with_segment,
            input_dtype=dtype.name,
            output_dtype=dtype.name,
        )
        inputs = values
        if with_segment:
            inputs = [values, tf.constant([[[1], [1], [2]]], dtype=dtype)]
        # when
        output_tensor = layer(inputs)
        # then
        tf.debugging.assert_equal(
            output_tensor,
            tf.constant([[[dtype_min], [dtype_min], [dtype_min]]], dtype=dtype),
        )

    @pytest.mark.parametrize(
        "min_filter_value, expected",
        [
            (-200.0, 3),
            (-129.0, 3),
            (-128.0, 3),
            (127.0, 0),
            (128.0, 0),
            (200.0, 0),
        ],
    )
    def test_listwise_max_min_filter_value_outside_integer_dtype_range(
        self, min_filter_value, expected
    ):
        """Narrowing a threshold that falls outside the value dtype wraps around, so
        the out-of-range cases are decided directly: below the dtype minimum keeps
        every value, above the dtype maximum keeps none of them."""
        # given, int8 values of 1, 2 and 3 against thresholds beyond the int8 bounds
        values = tf.constant([[[1], [2], [3]]], dtype=tf.int8)
        layer = ListMaxLayer(
            name="listwise_max_out_of_range_filter_test",
            min_filter_value=min_filter_value,
            nan_fill_value=0.0,
            input_dtype="int8",
            output_dtype="int8",
        )
        # when
        output_tensor = layer(values)
        # then
        tf.debugging.assert_equal(
            output_tensor, tf.constant([[[expected]] * 3], dtype=tf.int8)
        )

    @pytest.mark.parametrize("dtype", [tf.int8, tf.int16, tf.int32, tf.int64])
    def test_listwise_max_float_min_filter_value_on_integer_input(self, dtype):
        """min_filter_value is declared as a float, so passing one against an
        integer input column must work rather than fail the dtype comparison."""
        # given
        values = tf.constant([[[1], [2], [3]]], dtype=dtype)
        layer = ListMaxLayer(
            name="listwise_max_float_filter_test",
            min_filter_value=0.0,
            nan_fill_value=-1.0,
            input_dtype=dtype.name,
            output_dtype=dtype.name,
        )
        # when
        output_tensor = layer(values)
        # then
        tf.debugging.assert_equal(
            output_tensor, tf.constant([[[3], [3], [3]]], dtype=dtype)
        )
