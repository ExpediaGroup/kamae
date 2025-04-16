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

from kamae.tensorflow.layers import ArraySubtractMinimumLayer


class TestArraySubtractMinimum:
    @pytest.mark.parametrize(
        "input_tensor, pad_value, axis, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[11.0, 5.0, 2.0, -1.0]], dtype="float64"),
                -1.5,
                -1,
                "input_1",
                "int64",
                "int32",
                tf.constant([[12, 6, 3, 0]], dtype="int32"),
            ),
            (
                tf.constant([[11002.0, 5.0, 2.0, -1.0], [11.0, 5.0, -1.0, -1.0]]),
                -1.0,
                -1,
                "input_2",
                None,
                None,
                tf.constant([[11000.0, 3.0, 0.0, -1.0], [6.0, 0.0, -1.0, -1.0]]),
            ),
            (
                tf.constant([[11002.0, 5.0, 2.0, -1.0], [11.0, 5.0, -1.0, -1.0]]),
                None,
                -1,
                "input_3",
                "float32",
                "string",
                tf.constant(
                    [["11003.0", "6.0", "3.0", "0.0"], ["12.0", "6.0", "0.0", "0.0"]]
                ),
            ),
            (
                tf.constant([[11002, 5, 2, 0], [11, 5, 0, 0]], dtype="int64"),
                0,
                -1,
                "input_4",
                "int64",
                "int32",
                tf.constant([[11000, 3, 0, 0], [6, 0, 0, 0]], dtype="int32"),
            ),
            (
                tf.constant(
                    [
                        [[11002.0, 5.0, 2.0, 0.0], [112.0, 50.0, 21.0, 0.0]],
                        [[11.0, 5.0, 0.0, 0.0], [67.0, 56.0, 276.0, 0.0]],
                    ]
                ),
                0,
                0,
                "input_5",
                None,
                None,
                tf.constant(
                    [
                        [[10991.0, 0.0, 0.0, 0.0], [45.0, 0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 6.0, 255.0, 0.0]],
                    ]
                ),
            ),
            (
                tf.constant(
                    [
                        [
                            [19, 18, 13, 11, 10, -1, -1, -1],
                            [12, 2, 1, -1, -1, -1, -1, -1],
                        ],
                        [
                            [26, 13, 12, 11, 10, -1, -1, -1],
                            [8, 2, 1, -1, -1, -1, -1, -1],
                        ],
                    ]
                ),
                -1,
                -1,
                "input_6",
                "int32",
                "int64",
                tf.constant(
                    [
                        [
                            [9, 8, 3, 1, 0, -1, -1, -1],
                            [11, 1, 0, -1, -1, -1, -1, -1],
                        ],
                        [
                            [16, 3, 2, 1, 0, -1, -1, -1],
                            [7, 1, 0, -1, -1, -1, -1, -1],
                        ],
                    ],
                    dtype="int64",
                ),
            ),
            (
                tf.constant(
                    [
                        [
                            [19, 18, 13, 11, 10, -1, -1, -1],
                            [12, 2, 1, -1, -1, -1, -1, -1],
                            [8, 2, 1, -1, -1, -1, -1, -1],
                        ],
                        [
                            [26, 13, 12, 11, 10, -1, -1, -1],
                            [8, 2, 1, -1, -1, -1, -1, -1],
                            [7, 1, 0, -1, -1, -1, -1, -1],
                        ],
                    ]
                ),
                -1,
                1,
                "input_7",
                None,
                None,
                tf.constant(
                    [
                        [
                            [11, 16, 12, 0, 0, -1, -1, -1],
                            [4, 0, 0, -1, -1, -1, -1, -1],
                            [0, 0, 0, -1, -1, -1, -1, -1],
                        ],
                        [
                            [19, 12, 12, 0, 0, -1, -1, -1],
                            [1, 1, 1, -1, -1, -1, -1, -1],
                            [0, 0, 0, -1, -1, -1, -1, -1],
                        ],
                    ]
                ),
            ),
        ],
    )
    def test_array_subtract_minimum(
        self,
        input_tensor,
        pad_value,
        axis,
        input_name,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = ArraySubtractMinimumLayer(
            pad_value=pad_value,
            name=input_name,
            axis=axis,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(input_tensor)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(expected_output, output_tensor)

    @pytest.mark.parametrize(
        "inputs, input_name, input_dtype, output_dtype",
        [
            (
                tf.constant(["1.0", "2.0", "3.0"], dtype="string"),
                "input_1",
                None,
                "float64",
            )
        ],
    )
    def test_array_subtract_minimum_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = ArraySubtractMinimumLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
