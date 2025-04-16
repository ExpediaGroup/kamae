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

from kamae.tensorflow.layers.ordinal_array_encode import OrdinalArrayEncodeLayer


class TestOrdinalArrayEncode:
    @pytest.mark.parametrize(
        "input_tensor, pad_value, axis, input_name, expected_output",
        [
            (
                tf.constant([["a", "a", "b", "-1"]]),
                "-1",
                -1,
                "input_1",
                tf.constant([[0, 0, 1, -1]]),
            ),
            (
                tf.constant([["a", "a", "b", "-1"]]),
                None,
                -1,
                "input_2",
                tf.constant([[0, 0, 1, 2]]),
            ),
            (
                tf.constant([[[["-1", "a", "b", "c"]]]]),
                "-1",
                -1,
                "input_3",
                tf.constant([[[[-1, 0, 1, 2]]]]),
            ),
            (
                tf.constant([[[["-1", "-1", "-1", "-1"]]]]),
                "-1",
                -1,
                "input_4",
                tf.constant([[[[-1, -1, -1, -1]]]]),
            ),
            (
                tf.constant([[["b", "b"], ["b", "a"]]]),
                "-1",
                1,
                "input_5",
                tf.constant([[[0, 0], [0, 1]]]),
            ),
            (
                tf.constant([[["a", "-1", "b", "-1"], ["a", "a", "b", "c"]]]),
                "-1",
                1,
                "input_6",
                tf.constant([[[0, -1, 0, -1], [0, 0, 0, 0]]]),
            ),
            (
                tf.constant([["a", "a", "a"], ["b", "b", "b"]]),
                "-1",
                0,
                "input_7",
                tf.constant([[0, 0, 0], [1, 1, 1]]),
            ),
        ],
    )
    def test_ordinal_array_encoder(
        self, input_tensor, pad_value, axis, input_name, expected_output
    ):
        # when
        layer = OrdinalArrayEncodeLayer(pad_value=pad_value, axis=axis, name=input_name)
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
