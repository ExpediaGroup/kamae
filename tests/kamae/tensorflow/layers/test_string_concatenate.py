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

from kamae.tensorflow.layers import StringConcatenateLayer


class TestStringConcatenate:
    @pytest.mark.parametrize(
        "input_tensors, layer_name, separator, input_dtype, output_dtype, expected_output",
        [
            # 1D
            (
                [
                    tf.constant(["Chill", "Friday", "Vibes"]),
                    tf.constant(["Chill", "Saturday", "Vibes"]),
                    tf.constant(["Chill", "Sunday", "Vibes"]),
                ],
                "chill",
                "~",
                None,
                None,
                tf.constant(
                    [
                        "Chill~Chill~Chill",
                        "Friday~Saturday~Sunday",
                        "Vibes~Vibes~Vibes",
                    ]
                ),
            ),
            # more dims
            (
                [
                    tf.constant([[["I'm"]]]),
                    tf.constant([[["very"]]]),
                    tf.constant([[["serious"]]]),
                ],
                "sirius",
                ".",
                None,
                None,
                tf.constant([[["I'm.very.serious"]]]),
            ),
            # different dtypes
            (
                [
                    tf.constant(["Chill", "Friday", "Vibes"]),
                    tf.constant([0, 1, 2], dtype=tf.int32),
                ],
                "casting chill",
                "*(^_^)*",
                "string",
                None,
                tf.constant(["Chill*(^_^)*0", "Friday*(^_^)*1", "Vibes*(^_^)*2"]),
            ),
        ],
    )
    def test_string_concatenate(
        self,
        input_tensors,
        layer_name,
        separator,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringConcatenateLayer(
            name=layer_name,
            separator=separator,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(input_tensors)
        # then
        assert layer.name == layer_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(expected_output, output_tensor)
