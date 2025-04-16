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

from kamae.tensorflow.layers import StringEqualsIfStatementLayer


class TestStringEqualsIfStatement:
    @pytest.mark.parametrize(
        "inputs, input_name, value_to_compare, result_if_true, result_if_false, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1, 2, 3, 4, 5], dtype="int32"),  # input
                "test_string_if_statement",
                "3",  # value_to_compare
                "1",  # result_if_true
                "0",  # result_if_false
                "string",
                "float64",
                tf.constant([0.0, 0.0, 1.0, 0.0, 0.0], dtype="float64"),
            ),
            (
                [
                    tf.constant(["1", "2", "3", "4", "5"]),  # input
                    tf.constant(["-1", "not match", "3", "4", "5"]),  # value_to_compare
                ],
                "test_string_if_statement",
                None,
                "MATCH",  # result_if_true
                "NOT MATCH",  # result_if_false
                None,
                None,
                tf.constant(["NOT MATCH", "NOT MATCH", "MATCH", "MATCH", "MATCH"]),
            ),
            (
                [
                    tf.constant(
                        [
                            ["MATCH", "NOT_MATCH"],
                            ["NOT_MATCH", "MATCH"],
                            ["MATCH", "MATCH"],
                            ["NOT_MATCH", "NOT_MATCH"],
                        ]
                    ),  # input
                    tf.constant(
                        [
                            ["WE_MATCHED_1", "WE_MATCHED_2"],
                            ["WE_MATCHED_3", "WE_MATCHED_4"],
                            ["WE_MATCHED_5", "WE_MATCHED_6"],
                            ["WE_MATCHED_7", "WE_MATCHED_8"],
                        ]
                    ),  # result_if_true
                    tf.constant(
                        [
                            ["WE_DIDNT_MATCH_1", "WE_DIDNT_MATCH_2"],
                            ["WE_DIDNT_MATCH_3", "WE_DIDNT_MATCH_4"],
                            ["WE_DIDNT_MATCH_5", "WE_DIDNT_MATCH_6"],
                            ["WE_DIDNT_MATCH_7", "WE_DIDNT_MATCH_8"],
                        ]
                    ),  # result_if_false
                ],
                "test_string_if_statement",
                "MATCH",  # value_to_compare
                None,
                None,
                None,
                None,
                tf.constant(
                    [
                        ["WE_MATCHED_1", "WE_DIDNT_MATCH_2"],
                        ["WE_DIDNT_MATCH_3", "WE_MATCHED_4"],
                        ["WE_MATCHED_5", "WE_MATCHED_6"],
                        ["WE_DIDNT_MATCH_7", "WE_DIDNT_MATCH_8"],
                    ]
                ),
            ),
        ],
    )
    def test_string_if_statement(
        self,
        inputs,
        input_name,
        value_to_compare,
        result_if_true,
        result_if_false,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringEqualsIfStatementLayer(
            name=input_name,
            value_to_compare=value_to_compare,
            result_if_true=result_if_true,
            result_if_false=result_if_false,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )

        output_tensor = layer(inputs)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as input tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as input tensor shape"
        tf.debugging.assert_equal(output_tensor, expected_output)
