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

from kamae.tensorflow.layers import SubStringDelimAtIndexLayer


class TestSubStringDelimAtIndex:
    @pytest.mark.parametrize(
        "input_tensor, input_name, delimiter, index, default_value, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[["Sunday Vibes"], ["Saturday Vibes"], ["Friday Vibes"]]]),
                "input_1",
                "day",
                1,
                "NOT_FOUND",
                None,
                None,
                tf.constant([[[" Vibes"], [" Vibes"], [" Vibes"]]]),
            ),
            (
                tf.constant(
                    [[["Sunday Vibes"], ["Saturday Vibes"], ["Friday Vibes"], ["Day"]]]
                ),
                "input_2",
                "",
                8,
                "NOT_FOUND",
                None,
                None,
                tf.constant([[["i"], [" "], ["i"], ["NOT_FOUND"]]]),
            ),
            (
                tf.constant(["Monday_Tuesday", "Wednesday_Thursday", "Friday"]),
                "input_3",
                "_",
                1,
                "NOT_FOUND",
                None,
                None,
                tf.constant(["Tuesday", "Thursday", "NOT_FOUND"]),
            ),
            (
                tf.constant(
                    [[["EXPEDIA.COM", "EXPEDIA.CO.UK"], ["EXPEDIA.CA", "EXPEDIA.CH"]]]
                ),
                "input_4",
                ".",
                1,
                "NOT_FOUND",
                None,
                None,
                tf.constant([[["COM", "CO"], ["CA", "CH"]]]),
            ),
            (
                tf.constant(
                    [[["EXPEDIA.COM", "EXPEDIA.CO.UK"], ["EXPEDIA.CA", "EXPEDIA.CH"]]]
                ),
                "input_5",
                "",
                0,
                "NOT_FOUND",
                None,
                None,
                tf.constant([[["E", "E"], ["E", "E"]]]),
            ),
            (
                tf.constant(
                    [
                        [
                            ["EXPEDIA.COM", "EXPEDIA.CO.UK"],
                            ["EXPEDIA.CA", "EXPEDIA.CH"],
                        ],
                        [
                            ["EXPEDIA.HELLO", "EXPEDIA.THIS"],
                            ["EXPEDIA.IS", "EXPEDIA.TEST"],
                        ],
                    ]
                ),
                "input_6",
                ".",
                -1,
                "NOT_FOUND",
                None,
                None,
                tf.constant(
                    [[["COM", "UK"], ["CA", "CH"]], [["HELLO", "THIS"], ["IS", "TEST"]]]
                ),
            ),
            (
                tf.constant(
                    [
                        [
                            ["EXPEDIA.COM", "EXPEDIA.CO.UK"],
                            ["EXPEDIA.CA", "EXPEDIA.CH"],
                        ],
                        [
                            ["EXPEDIA.HELLO", "EXPEDIA.THIS"],
                            ["EXPEDIA.IS", "EXPEDIA.TEST"],
                        ],
                    ]
                ),
                "input_6",
                ".",
                -3,
                "NOT_FOUND",
                None,
                None,
                tf.constant(
                    [
                        [["NOT_FOUND", "EXPEDIA"], ["NOT_FOUND", "NOT_FOUND"]],
                        [["NOT_FOUND", "NOT_FOUND"], ["NOT_FOUND", "NOT_FOUND"]],
                    ]
                ),
            ),
            (
                tf.constant([[[100.0], [100.1], [10.123]]], dtype="float64"),
                "input_7",
                ".",
                1,
                "NOT_FOUND",
                "string",
                "int32",
                tf.constant([[[0], [1], [123]]], dtype="int32"),
            ),
            (
                tf.constant([[[100.0], [100.1], [10.123]]], dtype="float64"),
                "input_7",
                ".",
                -200,
                "0",
                "string",
                "int32",
                tf.constant([[[0], [0], [0]]], dtype="int32"),
            ),
        ],
    )
    def test_sub_string_delim(
        self,
        input_tensor,
        input_name,
        delimiter,
        index,
        default_value,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = SubStringDelimAtIndexLayer(
            name=input_name,
            delimiter=delimiter,
            index=index,
            default_value=default_value,
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
