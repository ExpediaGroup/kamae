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

from kamae.tensorflow.layers import RoundToDecimalLayer


class TestRoundToDecimal:
    @pytest.mark.parametrize(
        "input_tensor, input_name, decimals, input_dtype, output_dtype, expected_output",
        [
            (
                [tf.constant([1.3455, 2.46557, 3.456754])],
                "input_1",
                1,
                "float64",
                None,
                tf.constant([1.3, 2.5, 3.5], dtype="float64"),
            ),
            (
                [tf.constant([1, 2, 3])],
                "input_1",
                1,
                "int32",
                "float16",
                tf.constant([1.0, 2.0, 3.0], dtype="float16"),
            ),
            (
                tf.constant([5.2345, 2.678678]),
                "input_2",
                3,
                None,
                None,
                tf.constant([5.234, 2.679]),
            ),
            (
                tf.constant([[[1.5456, 2.35461, 30.21332]]]),
                "input_3",
                4,
                "float32",
                "string",
                tf.constant([[["1.5456", "2.3546", "30.2133"]]]),
            ),
        ],
    )
    def test_round_to_decimal(
        self,
        input_tensor,
        input_name,
        decimals,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = RoundToDecimalLayer(
            name=input_name,
            decimals=decimals,
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
        "inputs, input_name",
        [
            (
                tf.constant([1.3455, 2.46557, 3.456754], dtype=tf.float32),
                "input_1",
            ),
            (
                tf.constant([5.2345, 2.678678], dtype=tf.float16),
                "input_2",
            ),
        ],
    )
    def test_round_to_decimal_raises_error(self, inputs, input_name):
        with pytest.raises(ValueError):
            layer = RoundToDecimalLayer(name=input_name, decimals=100)
            layer(inputs)

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
    def test_round_to_decimal_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = RoundToDecimalLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            decimals=3,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
