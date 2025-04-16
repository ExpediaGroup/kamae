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

from kamae.tensorflow.layers import StringContainsLayer


class TestStringContains:
    @pytest.mark.parametrize(
        "input_tensors, input_name, string_constant, negation, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_1",
                "on",
                False,
                None,
                None,
                tf.constant([True, False, False], dtype=tf.bool),
            ),
            (
                [tf.constant(["Mon", "Tue", "Wed"]), tf.constant(["on", "ue", "ed"])],
                "input_2",
                None,
                False,
                None,
                None,
                tf.constant([True, True, True], dtype=tf.bool),
            ),
            (
                tf.constant([["Mon"], ["Tue"], ["Wed"]]),
                "input_3",
                "on",
                False,
                None,
                None,
                tf.constant([[True], [False], [False]], dtype=tf.bool),
            ),
            (
                [
                    tf.constant([["Mon"], ["Tue"], ["Wed"]]),
                    tf.constant([["on"], ["ue"], ["ed"]]),
                ],
                "input_4",
                None,
                False,
                None,
                None,
                tf.constant([[True], [True], [True]], dtype=tf.bool),
            ),
            (
                [
                    tf.constant([[["Mon"]], [["Tue"]], [["Wed"]]]),
                    tf.constant([[["ue"]], [["ed"]], [["W"]]]),
                ],
                "input_5",
                None,
                False,
                None,
                None,
                tf.constant([[[False]], [[False]], [[True]]], dtype=tf.bool),
            ),
            (
                [
                    tf.constant([[["Mon"]], [["Tue"]], [["Wed"]]]),
                    tf.constant([[["ue"]], [["ed"]], [["W"]]]),
                ],
                "input_6",
                None,
                True,
                None,
                None,
                tf.constant([[[True]], [[True]], [[False]]], dtype=tf.bool),
            ),
            (
                tf.constant([[["Mon"]], [[""]], [["Wed"]]]),
                "input_7",
                "",
                True,
                None,
                None,
                tf.constant([[[True]], [[False]], [[True]]], dtype=tf.bool),
            ),
            (
                [
                    tf.constant(
                        ["BANG(!)", "WHAT?!", "5 ***** REVIEW", "[]{x}", "", ""]
                    ),
                    tf.constant(["(!)", "?", "5 *****", "[]{", "^$", ".*"]),
                ],
                "input_8",
                None,
                False,
                None,
                None,
                tf.constant([True, True, True, True, True, True], dtype=tf.bool),
            ),
            (
                {
                    "first": tf.constant(["Mon", "Tue", "Wed"]),
                    "second": tf.constant(["on", "ue", "no"]),
                },
                "input_9",
                None,
                False,
                None,
                None,
                tf.constant([True, True, False], dtype=tf.bool),
            ),
            (
                {
                    "first": tf.constant(["Mon", "Tue", "Wed"]),
                    "second": tf.constant(["u", "u", "u"]),
                },
                "input_10",
                None,
                False,
                None,
                None,
                tf.constant([False, True, False], dtype=tf.bool),
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
                    tf.constant(
                        [
                            [
                                ["A", "A", "W", "]", "", ""],
                                ["A", "A", "W", "]", "", ""],
                            ],
                            [
                                ["A", "A", "W", "]", "", ""],
                                ["A", "A", "W", "]", "", ""],
                            ],
                        ]
                    ),
                ],
                "input_11",
                None,
                False,
                None,
                None,
                tf.constant(
                    [
                        [
                            [True, True, True, True, True, True],
                            [True, True, True, True, True, True],
                        ],
                        [
                            [True, True, True, True, True, True],
                            [True, True, True, True, True, True],
                        ],
                    ]
                ),
            ),
            (
                tf.constant([1, 10, 100], dtype="int32"),
                "input_12",
                "0",
                False,
                "string",
                "int32",
                tf.constant([0, 1, 1], dtype="int32"),
            ),
        ],
    )
    def test_string_contains(
        self,
        input_tensors,
        input_name,
        string_constant,
        negation,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringContainsLayer(
            name=input_name,
            string_constant=string_constant,
            negation=negation,
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
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(expected_output, output_tensor)
