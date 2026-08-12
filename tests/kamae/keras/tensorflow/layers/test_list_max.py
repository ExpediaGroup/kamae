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
