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

from kamae.tensorflow.layers import StringReplaceLayer


class TestStringReplace:
    @pytest.mark.parametrize(
        "input_tensors, input_name, string_match_constant, string_replace_constant, regex, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_1",
                "on",
                "day",
                False,
                None,
                None,
                tf.constant(["Mday", "Tue", "Wed"]),
            ),
            (
                [
                    tf.constant(
                        [
                            [["Mon"], ["Tue"], ["Wed"]],
                        ]
                    ),
                    tf.constant(
                        [
                            [["\\\\z"], ["\\\\z"], ["\\\\z"]],
                        ]
                    ),
                ],
                "input_2",
                "on",
                None,
                False,
                None,
                None,
                tf.constant([[["M\\z"], ["Tue"], ["Wed"]]]),
            ),
            (
                [
                    tf.constant(
                        [
                            [[["Mon"], ["Tue"], ["Wed"]], [["Thu"], ["Fri"], ["Sat"]]],
                        ]
                    ),
                    tf.constant(
                        [
                            [[["M"], ["T"], ["W"]], [["T"], ["F"], ["S"]]],
                        ]
                    ),
                    tf.constant(
                        [
                            [[["A"], ["B"], ["C"]], [["D"], ["E"], ["F"]]],
                        ]
                    ),
                ],
                "input_3",
                None,
                None,
                False,
                None,
                None,
                tf.constant(
                    [
                        [[["Aon"], ["Bue"], ["Ced"]], [["Dhu"], ["Eri"], ["Fat"]]],
                    ]
                ),
            ),
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_4",
                "e",
                "excaliburGuinness",
                False,
                None,
                None,
                tf.constant(["Mon", "TuexcaliburGuinness", "WexcaliburGuinnessd"]),
            ),
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_5",
                "T.",
                "excaliburGuinness",
                True,
                None,
                None,
                tf.constant(["Mon", "excaliburGuinnesse", "Wed"]),
            ),
            (
                [
                    tf.constant(["Mon", "Tue", "Wed"]),
                    tf.constant(["o.", ".*", "W[ed]{2}"]),
                    tf.constant(["a", "b", "c"]),
                ],
                "input_6",
                None,
                None,
                True,
                None,
                None,
                tf.constant(["Ma", "b", "c"]),
            ),
            (
                [
                    tf.constant(["", "", ""]),
                    tf.constant(["", "", ""]),
                    tf.constant(["a", "b", "c"]),
                ],
                "input_7",
                None,
                None,
                False,
                None,
                None,
                tf.constant(["a", "b", "c"]),
            ),
            (
                [
                    tf.constant(["", "", ""]),
                    tf.constant(["$^", ".*", ""]),
                ],
                "input_8",
                None,
                "X",
                True,
                None,
                None,
                tf.constant(["X", "X", "X"]),
            ),
            # Known issue with backslash replacement needing double escaping
            (
                tf.constant(["a", "b", "c"]),
                "input_9",
                "a",
                "\\\\z",
                False,
                None,
                None,
                tf.constant(["\\z", "b", "c"]),
            ),
            (
                tf.constant([100, 200, 300], dtype="int32"),
                "input_10",
                "00",
                "23",
                False,
                "string",
                "float32",
                tf.constant([123.0, 223.0, 323.0], dtype="float32"),
            ),
        ],
    )
    def test_string_replace(
        self,
        input_tensors,
        input_name,
        string_match_constant,
        string_replace_constant,
        regex,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringReplaceLayer(
            name=input_name,
            string_match_constant=string_match_constant,
            string_replace_constant=string_replace_constant,
            regex=regex,
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
