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

from kamae.tensorflow.layers import IfStatementLayer


class TestIfStatement:
    @pytest.mark.parametrize(
        "inputs, input_name, condition_operator, value_to_compare, result_if_true, result_if_false, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1, 2, 3, 4, 5], dtype=tf.int64),  # input
                "test_if_statement",
                "geq",  # condition_operator >=
                3.23,  # value_to_compare
                True,  # result_if_true
                False,  # result_if_false
                None,
                "string",
                tf.constant(["false", "false", "false", "true", "true"]),
            ),
            (
                [
                    tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),  # input
                    tf.constant(
                        [3.0, 3.0, 3.0, 3.0, 3.0], dtype=tf.float32
                    ),  # value_to_compare
                ],
                "test_if_statement",
                "geq",  # condition_operator >=
                None,
                1.0,  # result_if_true
                0.0,  # result_if_false
                None,
                None,
                tf.constant([0.0, 0.0, 1.0, 1.0, 1.0], dtype=tf.float32),
            ),
            (
                [
                    tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),  # input
                    tf.constant(
                        [31, 32, 33, 34, 35], dtype=tf.float32
                    ),  # result_if_true
                    tf.constant(
                        [0.1, 0.2, 0.3, 0.4, 0.5], dtype=tf.float32
                    ),  # result_if_false
                ],
                "test_if_statement",
                "geq",  # condition_operator >=
                3.0,  # value_to_compare
                None,
                None,
                None,
                "float16",
                tf.constant([0.1, 0.2, 33, 34, 35], dtype=tf.float16),
            ),
            (
                [
                    tf.constant([23.45, 2.023, 3.456, 4, 5], dtype=tf.float32),  # input
                    tf.constant(
                        [23.45, 2.023, 3.456, 3.9, 4.9], dtype=tf.float32
                    ),  # value_to_compare
                ],
                "test_if_statement",
                "eq",  # condition_operator >=
                None,
                1.0,  # result_if_true
                0.0,  # result_if_false
                "float16",
                None,
                tf.constant([1.0, 1.0, 1.0, 0.0, 0.0], dtype=tf.float16),
            ),
            (
                [
                    tf.constant([2.45, 2.023, 3.456, 4, 5], dtype=tf.float32),  # input
                    tf.constant(
                        [1.0, 2.0, 3.4, 4.0, -1.9], dtype=tf.float32
                    ),  # result_if_true
                ],
                "test_if_statement",
                "leq",  # condition_operator <=
                3.3,  # value_to_compare
                None,
                0.0,  # result_if_false
                "float64",
                "float32",
                tf.constant([1.0, 2.0, 0.0, 0.0, 0.0], dtype=tf.float32),
            ),
            (
                tf.constant(
                    [[78], [56], [456], [678], [60]], dtype=tf.float32
                ),  # input
                "test_if_statement",
                "neq",  # condition_operator !=
                78.0,  # value_to_compare
                5.0,  # result_if_true
                10.0,  # result_if_false
                None,
                None,
                tf.constant([[10.0], [5.0], [5.0], [5.0], [5.0]], dtype=tf.float32),
            ),
            (
                [
                    tf.constant(
                        [[78], [56], [456], [678], [60]], dtype=tf.float32
                    ),  # input
                    tf.constant(
                        [[23], [100], [250], [1000], [60]], dtype=tf.float32
                    ),  # value_to_compare
                    tf.constant(
                        [[1], [2], [3], [4], [5]], dtype=tf.float32
                    ),  # result_if_true
                ],
                "test_if_statement",
                "lt",  # condition_operator <
                None,
                None,
                -1.0,  # result_if_false
                "float32",
                None,
                tf.constant([[-1.0], [2.0], [-1.0], [4.0], [-1.0]], dtype=tf.float32),
            ),
            (
                [
                    tf.constant(
                        [[78], [56], [456], [678], [60]], dtype=tf.float32
                    ),  # input
                    tf.constant(
                        [[23], [100], [250], [1000], [60]], dtype=tf.float32
                    ),  # value_to_compare
                    tf.constant(
                        [[1], [2], [3], [4], [5]], dtype=tf.float32
                    ),  # result_if_true
                ],
                "test_if_statement",
                "gt",  # condition_operator <
                None,
                None,
                -1.0,  # result_if_false
                None,
                None,
                tf.constant([[1.0], [-1.0], [3.0], [-1.0], [-1.0]], dtype=tf.float32),
            ),
            (
                tf.constant([1, 2, 3, 4, 5], dtype="int32"),  # input
                "test_string_if_statement",
                "eq",
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
                    tf.constant(["-1", "not match", "3", "4", "5"]),
                    # value_to_compare
                ],
                "test_string_if_statement",
                "eq",
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
                "eq",
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
    def test_if_statement(
        self,
        inputs,
        input_name,
        condition_operator,
        value_to_compare,
        result_if_true,
        result_if_false,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = IfStatementLayer(
            name=input_name,
            condition_operator=condition_operator,
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
        if expected_output.dtype.is_floating:
            tf.debugging.assert_near(output_tensor, expected_output)
        else:
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
    def test_if_statement_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # then
        with pytest.raises(TypeError):
            layer = IfStatementLayer(
                name=input_name,
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                condition_operator="geq",
                value_to_compare="hello world",
            )
            layer(inputs)
