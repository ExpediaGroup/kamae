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

import pytest
import tensorflow as tf

from kamae.keras.tensorflow.layers import ListSumLayer


class TestListSum:
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
                tf.constant(
                    [
                        [
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                        ],
                        [
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With min_filter_value. The excluded values are zeros, which contribute
            # nothing to a sum, so the result matches the base case.
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
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                            [23.0],
                        ],
                        [
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
                            [33.0],
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
                # Top 5 ascending picks values [1, 1, 9, 4, 6] and [0, 0, 8, 6, 4].
                tf.constant(
                    [
                        [
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                        ],
                        [
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
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
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                            [21.0],
                        ],
                        [
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
                            [18.0],
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
                            [11.0],
                            [11.0],
                            [11.0],
                        ],
                        [
                            [15.0],
                            [15.0],
                            [15.0],
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
                            [10.0],
                            [10.0],
                        ],
                        [
                            [5.0],
                            [10.0],
                            [10.0],
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
                    [[[1.0, 10.0], [5.0, 50.0], [5.0, 50.0]]], dtype=tf.float32
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
                            [10.0],
                            [10.0],
                        ],
                        [
                            [5.0],
                            [10.0],
                            [10.0],
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
        ],
    )
    def test_listwise_sum(
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
        name = "listwise_sum_test"
        layer = ListSumLayer(
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

    def test_listwise_sum_raises_without_top_n_when_sorting(self):
        # given
        layer = ListSumLayer(name="listwise_sum_no_top_n", with_segment=False)
        inputs = [
            tf.constant([[[1.0], [2.0], [3.0]]]),
            tf.constant([[[1.0], [2.0], [3.0]]]),
        ]
        # when / then
        with pytest.raises(ValueError, match="topN must be specified"):
            layer(inputs)

    def test_listwise_sum_raises_with_segment_and_single_input(self):
        # given
        layer = ListSumLayer(name="listwise_sum_single_input", with_segment=True)
        # when / then
        with pytest.raises(ValueError, match="expected two inputs"):
            layer(tf.constant([[[1.0], [2.0], [3.0]]]))
