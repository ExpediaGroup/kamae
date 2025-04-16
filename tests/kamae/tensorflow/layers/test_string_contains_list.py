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

from kamae.tensorflow.layers import StringContainsListLayer


# TODO: Rename and repurpose
class TestStringContainsList:
    @pytest.mark.parametrize(
        "input_tensor, input_name, string_constant_list, negation, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_1",
                ["on"],
                False,
                None,
                None,
                tf.constant([True, False, False], dtype=tf.bool),
            ),
            (
                tf.constant(["Mon", "Tue", "Wed", "aaaa-bbbb-cccc-dddd", "cccc-dddd"]),
                "input_2",
                ["aaaa-bbbb", "cccc-dddd"],
                False,
                None,
                None,
                tf.constant([False, False, False, True, True], dtype=tf.bool),
            ),
            (
                tf.constant([["Mon"], ["Tue"], ["Wed"], ["a-b-c-d\\"]]),
                "input_3",
                ["ue", "d\\"],
                False,
                None,
                None,
                tf.constant([[False], [True], [False], [True]], dtype=tf.bool),
            ),
            (
                [
                    tf.constant([["Mon"], ["Tue"], ["Wed"]]),
                ],
                "input_4",
                ["on", "ed"],
                False,
                None,
                None,
                tf.constant([[True], [False], [True]], dtype=tf.bool),
            ),
            (
                tf.constant([[["Mo*n"]], [["Tu|e"]], [["Wed"]]]),
                "input_5",
                ["*ne", "|e", "ede"],
                False,
                None,
                None,
                tf.constant([[[False]], [[True]], [[False]]], dtype=tf.bool),
            ),
            (
                tf.constant([[["Mon"]], [["Tue"]], [["Wed"]], [["edur"]]]),
                "input_6",
                ["ed", "ur"],
                True,
                None,
                None,
                tf.constant([[[True]], [[True]], [[False]], [[False]]], dtype=tf.bool),
            ),
            (
                tf.constant(["BANG(!)", "WHAT?!", "5 ***** REVIEW", "[]{x}"]),
                "input_7",
                ["(!)", "?", "{"],
                False,
                None,
                None,
                tf.constant([True, True, False, True], dtype=tf.bool),
            ),
            (
                {
                    "first": tf.constant(["Mon", "Tue", "Wed"]),
                },
                "input_8",
                ["ed"],
                True,
                None,
                None,
                tf.constant([True, True, False], dtype=tf.bool),
            ),
            # Higher dimensional case
            (
                [
                    tf.constant(
                        [
                            [
                                [
                                    "BANG(!)",
                                    "WHAT?!",
                                    "5 ***** REVIEW",
                                    "[]{x}",
                                    "",
                                    "",
                                ],
                                [
                                    "BANG(!)",
                                    "WHAT?!",
                                    "5 ***** REVIEW",
                                    "[]{x}",
                                    "",
                                    "",
                                ],
                            ],
                            [
                                [
                                    "BANG(!)",
                                    "WHAT?!",
                                    "5 ***** REVIEW",
                                    "[]{x}",
                                    "",
                                    "",
                                ],
                                [
                                    "BANG(!)",
                                    "WHAT?!",
                                    "5 ***** REVIEW",
                                    "[]{x}",
                                    "",
                                    "",
                                ],
                            ],
                        ]
                    ),
                ],
                "input_9",
                ["AN", "?!", "*****", "x", "[]"],
                False,
                None,
                None,
                tf.constant(
                    [
                        [
                            [True, True, True, True, False, False],
                            [True, True, True, True, False, False],
                        ],
                        [
                            [True, True, True, True, False, False],
                            [True, True, True, True, False, False],
                        ],
                    ]
                ),
            ),
            (
                tf.constant([1, 10, 20, 25], dtype="int32"),
                "input_10",
                ["1", "0"],
                False,
                "string",
                "int32",
                tf.constant([1, 1, 1, 0], dtype="int32"),
            ),
        ],
    )
    def test_string_contains_list(
        self,
        input_tensor,
        input_name,
        string_constant_list,
        negation,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringContainsListLayer(
            name=input_name,
            string_constant_list=string_constant_list,
            negation=negation,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
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
