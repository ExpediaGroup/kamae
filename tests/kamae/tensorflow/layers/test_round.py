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

from kamae.tensorflow.layers import RoundLayer


class TestRound:
    @pytest.mark.parametrize(
        "input_tensor, input_name, round_type, input_dtype, output_dtype, expected_output",
        [
            (
                [tf.constant([1.3455, 2.46557, 3.456754])],
                "input_1",
                "ceil",
                "float64",
                None,
                tf.constant([2.0, 3.0, 4.0], dtype="float64"),
            ),
            (
                tf.constant([5.2345, 2.678678], dtype="float64"),
                "input_2",
                "floor",
                "float32",
                "string",
                tf.constant(["5.0", "2.0"]),
            ),
            (
                tf.constant([[[1.5456, 2.35445, 30.21332]]]),
                "input_3",
                "round",
                None,
                None,
                tf.constant([[[2.0, 2.0, 30.0]]]),
            ),
        ],
    )
    def test_round(
        self,
        input_tensor,
        input_name,
        round_type,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = RoundLayer(
            name=input_name,
            round_type=round_type,
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
    def test_round_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = RoundLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            round_type="ceil",
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
