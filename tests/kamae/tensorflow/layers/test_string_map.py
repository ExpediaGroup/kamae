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

from kamae.tensorflow.layers import StringMapLayer


class TestStringMap:
    @pytest.mark.parametrize(
        "input_tensors, input_name, string_match_values, string_replace_values, default_replace_value, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["hello", "there", "friend"]),
                "input_1",
                ["hello", "friend"],
                ["hi", "pal"],
                None,
                None,
                None,
                tf.constant(["hi", "there", "pal"]),
            ),
            # Test default value
            (
                tf.constant(["hello", "there", "friend"]),
                "input_1",
                ["hello", "friend"],
                ["hi", "pal"],
                "default",
                None,
                None,
                tf.constant(["hi", "default", "pal"]),
            ),
            # Test numerics
            (
                tf.constant([0, 1, 3]),
                "input_1",
                ["2", "3"],
                ["4", "4"],
                None,
                "string",
                "int32",
                tf.constant([0, 1, 4]),
            ),
        ],
    )
    def test_string_map(
        self,
        input_tensors,
        input_name,
        string_match_values,
        string_replace_values,
        default_replace_value,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringMapLayer(
            name=input_name,
            string_match_values=string_match_values,
            string_replace_values=string_replace_values,
            default_replace_value=default_replace_value,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # For the sake of this test, convert dict to list
        if isinstance(input_tensors, dict):
            input_tensors = list(input_tensors.values())

        output_tensor = layer(input_tensors)

        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output shape is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        tf.debugging.assert_equal(expected_output, output_tensor)
