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

from kamae.tensorflow.layers import StringArrayConstantLayer


class TestStringArrayConstant:
    @pytest.mark.parametrize(
        "input_tensor, layer_name, constant_string_array, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[["Sunday Vibes"], ["Saturday Vibes"], ["Friday Vibes"]]]),
                "input_1",
                ["a", "b", "c"],
                None,
                None,
                tf.constant(
                    [
                        [
                            ["a", "b", "c"],
                            ["a", "b", "c"],
                            ["a", "b", "c"],
                        ]
                    ]
                ),
            ),
            (
                tf.constant([[["I'm"], ["very"], ["serious"]]]),
                "input_2",
                ["hello", "world"],
                None,
                None,
                tf.constant(
                    [[["hello", "world"], ["hello", "world"], ["hello", "world"]]]
                ),
            ),
            (
                tf.constant([1], dtype="int32"),
                "input_3",
                ["1", "2", "3", "4"],
                None,
                "int32",
                tf.constant([1, 2, 3, 4], dtype="int32"),
            ),
        ],
    )
    def test_string_array_constant(
        self,
        input_tensor,
        layer_name,
        constant_string_array,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringArrayConstantLayer(
            name=layer_name,
            constant_string_array=constant_string_array,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(input_tensor)
        # then
        assert layer.name == layer_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(expected_output, output_tensor)
