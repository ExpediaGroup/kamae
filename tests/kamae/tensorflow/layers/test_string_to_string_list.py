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

from kamae.tensorflow.layers import StringToStringListLayer


class TestStringToStringList:
    @pytest.mark.parametrize(
        "input_tensor, input_name, separator, default_value, list_length, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([["abc"], ["def"]]),
                "test_string_to_string_list_1",
                "",
                "DEFAULT",
                4,
                None,
                None,
                tf.constant([["a", "b", "c", "DEFAULT"], ["d", "e", "f", "DEFAULT"]]),
            ),
            (
                tf.constant([[100], [234]], dtype="int32"),
                "test_string_to_string_list_1point5",
                "",
                "DEFAULT",
                3,
                "string",
                "float32",
                tf.constant([[1.0, 0.0, 0.0], [2.0, 3.0, 4.0]], dtype="float32"),
            ),
            (
                tf.constant([["a_b_c"], ["d_e_f"]]),
                "test_string_to_string_list_2",
                "_",
                "DEFAULT",
                3,
                None,
                None,
                tf.constant([["a", "b", "c"], ["d", "e", "f"]]),
            ),
            (
                tf.constant(
                    [
                        [
                            "-1.0,6.789,3.067,456.078",
                            "0.0,0.0,0.0,0.0",
                            "-56.789,0.0,45.890",
                        ],
                        ["-1.0,6.789,3.067,456.078,87.9078", "0.0,0.0,0.0,0.0", "89.0"],
                    ]
                ),
                "test_string_to_string_list_3",
                ",",
                "DEFAULT",
                4,
                None,
                None,
                tf.constant(
                    [
                        [
                            ["-1.0", "6.789", "3.067", "456.078"],
                            ["0.0", "0.0", "0.0", "0.0"],
                            ["-56.789", "0.0", "45.890", "DEFAULT"],
                        ],
                        [
                            ["-1.0", "6.789", "3.067", "456.078"],
                            ["0.0", "0.0", "0.0", "0.0"],
                            ["89.0", "DEFAULT", "DEFAULT", "DEFAULT"],
                        ],
                    ]
                ),
            ),
            (
                tf.constant(
                    [
                        [
                            "-1.0|6.789|3.067|456.078",
                            "0.0|0.0|0.0|0.0",
                            "-56.789|0.0|45.890",
                        ],
                        ["-1.0|6.789|3.067|456.078|87.9078", "0.0|0.0|0.0||", "89.0"],
                    ]
                ),
                "test_string_to_string_list_3",
                "|",
                "DEFAULT",
                4,
                None,
                None,
                tf.constant(
                    [
                        [
                            ["-1.0", "6.789", "3.067", "456.078"],
                            ["0.0", "0.0", "0.0", "0.0"],
                            ["-56.789", "0.0", "45.890", "DEFAULT"],
                        ],
                        [
                            ["-1.0", "6.789", "3.067", "456.078"],
                            ["0.0", "0.0", "0.0", "DEFAULT"],
                            ["89.0", "DEFAULT", "DEFAULT", "DEFAULT"],
                        ],
                    ]
                ),
            ),
        ],
    )
    def test_string_to_string_list(
        self,
        input_tensor,
        input_name,
        separator,
        default_value,
        list_length,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringToStringListLayer(
            name=input_name,
            separator=separator,
            default_value=default_value,
            list_length=list_length,
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
