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

from kamae.tensorflow.layers import ListMeanLayer


class TestListMean:
    @pytest.mark.parametrize(
        "inputs, min_filter_value, top_n, sort_order, input_dtype, output_dtype, expected_output",
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
                "asc",
                "float64",
                "float32",
                # values
                tf.constant(
                    [
                        [
                            [2.875],
                            [2.875],
                            [2.875],
                            [2.875],
                            [2.875],
                            [2.875],
                            [2.875],
                            [2.875],
                        ],
                        [
                            [4.125],
                            [4.125],
                            [4.125],
                            [4.125],
                            [4.125],
                            [4.125],
                            [4.125],
                            [4.125],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # With min_filter_value
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
                1,
                None,
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [3.8333333],
                            [3.8333333],
                            [3.8333333],
                            [3.8333333],
                            [3.8333333],
                            [3.8333333],
                            [3.8333333],
                            [3.8333333],
                        ],
                        [
                            [5.5],
                            [5.5],
                            [5.5],
                            [5.5],
                            [5.5],
                            [5.5],
                            [5.5],
                            [5.5],
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
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                        ],
                        [
                            [3.6],
                            [3.6],
                            [3.6],
                            [3.6],
                            [3.6],
                            [3.6],
                            [3.6],
                            [3.6],
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
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                            [4.2],
                        ],
                        [
                            [6.0],
                            [6.0],
                            [6.0],
                            [6.0],
                            [6.0],
                            [6.0],
                            [6.0],
                            [6.0],
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
                "asc",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [3.6666667],
                            [3.6666667],
                            [3.6666667],
                        ],
                        [
                            [5.0],
                            [5.0],
                            [5.0],
                        ],
                    ],
                    dtype=tf.float32,
                ),
            ),
        ],
    )
    def test_listwise_mean(
        self,
        inputs,
        min_filter_value,
        top_n,
        sort_order,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        name = "listwise_mean_test"
        layer = ListMeanLayer(
            name=name,
            min_filter_value=min_filter_value,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            sort_order=sort_order,
            top_n=top_n,
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
