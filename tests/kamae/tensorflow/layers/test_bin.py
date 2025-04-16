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

from kamae.tensorflow.layers import BinLayer


class TestBin:
    @pytest.mark.parametrize(
        "inputs, condition_operators, bin_values, bin_labels, default_label, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, -1.0], dtype=tf.float32),
                ["eq", "eq", "gt"],
                [1.0, 2.5, 4.5],
                ["equal_to_1", "equal_to_2_point_5", "greater_than_4_point_5"],
                "default",
                "input_1",
                "int64",
                "string",
                tf.constant(
                    [
                        "equal_to_1",
                        "default",
                        "default",
                        "default",
                        "greater_than_4_point_5",
                        "default",
                    ]
                ),
            ),
            (
                tf.constant([[[-1, 0, 1], [2, 3, 4], [5, 6, 7]]], dtype=tf.float16),
                ["eq", "lt", "gt", "neq"],
                [0, 2, 5, 5],
                ["equal_to_0", "less_than_2", "greater_than_5", "not_equal_to_5"],
                "default",
                "input_2",
                None,
                "string",
                tf.constant(
                    [
                        [
                            ["less_than_2", "equal_to_0", "less_than_2"],
                            ["not_equal_to_5", "not_equal_to_5", "not_equal_to_5"],
                            ["default", "greater_than_5", "greater_than_5"],
                        ]
                    ]
                ),
            ),
            (
                tf.constant([10.0, 2.0, 31.567, 4.1234], dtype=tf.float64),
                ["lt", "leq", "gt"],
                [1, 2.0, 4.1456],
                [0, 1, 2],
                -1,
                "input_3",
                "float16",
                None,
                tf.constant(
                    [
                        2,
                        1,
                        2,
                        -1,
                    ]
                ),
            ),
        ],
    )
    def test_bin(
        self,
        inputs,
        condition_operators,
        bin_values,
        bin_labels,
        default_label,
        input_name,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = BinLayer(
            name=input_name,
            condition_operators=condition_operators,
            bin_values=bin_values,
            bin_labels=bin_labels,
            default_label=default_label,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(inputs)
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
    def test_bin_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = BinLayer(
            name=input_name,
            condition_operators=["eq", "eq", "gt"],
            bin_values=[1.0, 2.0, 4.0],
            bin_labels=["equal_to_1", "equal_to_2", "greater_than_4"],
            default_label="default",
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
