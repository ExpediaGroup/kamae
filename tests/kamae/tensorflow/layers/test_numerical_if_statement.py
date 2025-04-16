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

from kamae.tensorflow.layers import NumericalIfStatementLayer


class TestNumericalIfStatement:
    @pytest.mark.parametrize(
        "inputs, input_name, condition_operator, value_to_compare, result_if_true, result_if_false, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),  # input
                "test_numerical_if_statement",
                "geq",  # condition_operator >=
                3.0,  # value_to_compare
                1.0,  # result_if_true
                0.0,  # result_if_false
                "float64",
                "string",
                tf.constant(["0.0", "0.0", "1.0", "1.0", "1.0"]),
            ),
            (
                [
                    tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),  # input
                    tf.constant(
                        [3.0, 3.0, 3.0, 3.0, 3.0], dtype=tf.float32
                    ),  # value_to_compare
                ],
                "test_numerical_if_statement",
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
                "test_numerical_if_statement",
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
                "test_numerical_if_statement",
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
                "test_numerical_if_statement",
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
                "test_numerical_if_statement",
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
                "test_numerical_if_statement",
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
                "test_numerical_if_statement",
                "gt",  # condition_operator <
                None,
                None,
                -1.0,  # result_if_false
                None,
                None,
                tf.constant([[1.0], [-1.0], [3.0], [-1.0], [-1.0]], dtype=tf.float32),
            ),
        ],
    )
    def test_numerical_if_statement(
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
        layer = NumericalIfStatementLayer(
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
    def test_numerical_if_statement_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = NumericalIfStatementLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            condition_operator="geq",
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
