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

from kamae.tensorflow.layers import LogicalAndLayer


class TestLogicalAnd:
    @pytest.mark.parametrize(
        "input_tensors, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                [
                    tf.constant(["True", "True", "True"]),
                    tf.constant(["False", "True", "True"]),
                    tf.constant(["False", "False", "True"]),
                ],
                "input_1",
                "bool",
                "string",
                tf.constant(["false", "false", "true"]),
            ),
            (
                [
                    tf.constant(
                        [
                            [1, 1, 1],
                            [0, 1, 1],
                            [1, 0, 1],
                        ]
                    ),
                    tf.constant(
                        [
                            [False, True, False],
                            [True, False, True],
                            [False, True, False],
                        ]
                    ),
                    tf.constant(
                        [
                            [True, False, True],
                            [False, True, True],
                            [False, False, True],
                        ]
                    ),
                ],
                "input_2",
                "bool",
                None,
                tf.constant(
                    [
                        [False, False, False],
                        [False, False, True],
                        [False, False, False],
                    ]
                ),
            ),
            (
                [
                    tf.constant(
                        [
                            [[0, 1, 1, 1, 0]],
                            [[1, 0, 0, 0, 1]],
                            [[0, 1, 0, 1, 1]],
                        ]
                    ),
                    tf.constant(
                        [
                            [["True", "False", "True", "True", "True"]],
                            [["False", "True", "True", "True", "False"]],
                            [["True", "False", "True", "True", "True"]],
                        ]
                    ),
                ],
                "input_3",
                "bool",
                "bool",
                tf.constant(
                    [
                        [[False, False, True, True, False]],
                        [[False, False, False, False, False]],
                        [[False, False, False, True, True]],
                    ]
                ),
            ),
        ],
    )
    def test_logical_and(
        self, input_tensors, input_name, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = LogicalAndLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        output_tensor = layer(input_tensors)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(output_tensor, expected_output)

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
    def test_logical_and_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = LogicalAndLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
