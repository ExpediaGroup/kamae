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

from kamae.tensorflow.layers import StringCaseLayer


class TestStringCase:
    @pytest.mark.parametrize(
        "input_tensor, input_name, string_case_type, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[["Sunday Vibes"], ["Saturday Vibes"], ["Friday Vibes"]]]),
                "input_1",
                "lower",
                None,
                None,
                tf.constant([[["sunday vibes"], ["saturday vibes"], ["friday vibes"]]]),
            ),
            (
                tf.constant(["HElLO wOrLd", "adVeNtuRE TIme", "BeGiNs"]),
                "input_2",
                "upper",
                None,
                None,
                tf.constant(["HELLO WORLD", "ADVENTURE TIME", "BEGINS"]),
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
                "input_3",
                "lower",
                None,
                None,
                tf.constant(
                    [
                        [
                            ["expedia.com", "expedia.co.uk"],
                            ["expedia.ca", "expedia.ch"],
                        ],
                        [
                            ["expedia.hello", "expedia.this"],
                            ["expedia.is", "expedia.test"],
                        ],
                    ]
                ),
            ),
            (
                tf.constant(["TrUe", "FaLsE"]),
                "input_4",
                "lower",
                None,
                "bool",
                tf.constant([True, False], dtype="bool"),
            ),
        ],
    )
    def test_sub_string_delim(
        self,
        input_tensor,
        input_name,
        string_case_type,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringCaseLayer(
            name=input_name,
            string_case_type=string_case_type,
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
