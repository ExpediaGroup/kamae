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

from kamae.tensorflow.layers import StringIsInListLayer


class TestStringIsInList:
    @pytest.mark.parametrize(
        "input_tensor, input_name, input_dtype, output_dtype, string_constant_list, negation, expected_output",
        [
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_1",
                None,
                "bool",
                ["Mon"],
                False,
                tf.constant([True, False, False], dtype=tf.bool),
            ),
            (
                tf.constant([["Mon"], ["mon"], [""], ["MON"]]),
                "input_3",
                "string",
                "bool",
                ["mon"],
                False,
                tf.constant([[False], [True], [False], [False]], dtype=tf.bool),
            ),
            (
                tf.constant([["Mon"], ["mon"], [""], ["MON"]]),
                "input_3",
                "string",
                "float",
                ["mon"],
                False,
                tf.constant([[0.0], [1.0], [0.0], [0.0]], dtype=tf.float32),
            ),
            (
                tf.constant([["Mon"], ["mon"], [""], ["MON"]]),
                "input_3",
                "string",
                "bool",
                ["mon", ""],
                False,
                tf.constant([[False], [True], [True], [False]], dtype=tf.bool),
            ),
            (
                tf.constant([["Mon"], ["mon"], [""], ["MON"]]),
                "input_3",
                "string",
                "bool",
                ["mon", ""],
                True,
                tf.constant([[True], [False], [False], [True]], dtype=tf.bool),
            ),
            # Casting of inputs
            (
                tf.constant([[1], [2], [3], [4]]),
                "input_3",
                "string",
                "float",
                ["1"],
                False,
                tf.constant([[1.0], [0.0], [0.0], [0.0]], dtype=tf.float32),
            ),
            # Higher dimensional case
            (
                [
                    tf.constant(
                        [
                            [
                                [
                                    "Mon",
                                    "Tue",
                                    "Wed",
                                    "",
                                ],
                                [
                                    "Thu",
                                    "Fri",
                                    "Sat",
                                    "Sun",
                                ],
                            ],
                            [
                                [
                                    "Fri",
                                    "Mon",
                                    "Sun",
                                    "",
                                ],
                                [
                                    "Mon",
                                    "",
                                    "Wed",
                                    "Fri",
                                ],
                            ],
                        ]
                    ),
                ],
                "input_13",
                "string",
                "bool",
                ["Mon", ""],
                False,
                tf.constant(
                    [
                        [
                            [True, False, False, True],
                            [False, False, False, False],
                        ],
                        [
                            [False, True, False, True],
                            [True, True, False, False],
                        ],
                    ]
                ),
            ),
        ],
    )
    def test_string_isin_list(
        self,
        input_tensor,
        input_name,
        input_dtype,
        output_dtype,
        string_constant_list,
        negation,
        expected_output,
    ):
        # when
        layer = StringIsInListLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            string_constant_list=string_constant_list,
            negation=negation,
        )
        # For the sake of this test, convert dict to list
        if isinstance(input_tensor, dict):
            input_tensor = list(input_tensor.values())

        output_tensor = layer(input_tensor)

        if isinstance(input_tensor, list):
            input_shape = input_tensor[0].shape
        else:
            input_shape = input_tensor.shape

        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert output_tensor.shape == input_shape, "Output shape is not set properly"
        tf.debugging.assert_equal(expected_output, output_tensor)
