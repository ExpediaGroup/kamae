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

from kamae.tensorflow.layers import StringListToStringLayer


class TestStringListToString:
    @pytest.mark.parametrize(
        "input_tensor, input_name, separator, axis, keep_dims, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([["a", "b", "c"], ["d", "e", "f"]]),
                "test_string_list_to_string_1",
                "",
                -1,
                False,
                None,
                None,
                tf.constant(["abc", "def"]),
            ),
            (
                tf.constant([["a", "b", "c"], ["d", "e", "f"]]),
                "test_string_list_to_string_2",
                "_",
                -1,
                True,
                None,
                None,
                tf.constant([["a_b_c"], ["d_e_f"]]),
            ),
            (
                tf.constant(
                    [
                        [["a", "b", "c"], ["d", "e", "f"]],
                        [["g", "h", "i"], ["j", "k", "l"]],
                    ]
                ),
                "test_string_list_to_string_3",
                " ",
                1,
                True,
                None,
                None,
                tf.constant([[["a d", "b e", "c f"]], [["g j", "h k", "i l"]]]),
            ),
            (
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="int32"),
                "test_string_list_to_string_4",
                "",
                -1,
                False,
                "string",
                "int32",
                tf.constant([123, 456], dtype="int32"),
            ),
        ],
    )
    def test_string_list_to_string(
        self,
        input_tensor,
        input_name,
        separator,
        axis,
        keep_dims,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringListToStringLayer(
            name=input_name,
            separator=separator,
            axis=axis,
            keepdims=keep_dims,
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
