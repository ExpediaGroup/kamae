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

from kamae.tensorflow.layers import LogicalNotLayer


class TestLogicalNot:
    @pytest.mark.parametrize(
        "input_tensor, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["True", "False", "True"]),
                "input_1",
                "bool",
                "string",
                tf.constant(["false", "true", "false"]),
            ),
            (
                tf.constant(
                    [
                        [1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1],
                    ]
                ),
                "input_2",
                "bool",
                None,
                tf.constant(
                    [
                        [False, True, False],
                        [True, False, True],
                        [False, True, False],
                    ]
                ),
            ),
            (
                tf.constant(
                    [
                        [[False, True, True, True, False]],
                        [[True, False, False, False, True]],
                        [[False, True, False, False, False]],
                    ]
                ),
                "input_3",
                None,
                None,
                tf.constant(
                    [
                        [[True, False, False, False, True]],
                        [[False, True, True, True, False]],
                        [[True, False, True, True, True]],
                    ]
                ),
            ),
        ],
    )
    def test_logical_not(
        self, input_tensor, input_name, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = LogicalNotLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
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
    def test_logical_not_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = LogicalNotLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
