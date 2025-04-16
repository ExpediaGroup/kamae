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

from kamae.tensorflow.layers import ExponentLayer


class TestExponent:
    @pytest.mark.parametrize(
        "inputs, input_name, exponent, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1.0, 2.0, 3.0]),
                "input_1",
                5.0,
                "float32",
                None,
                tf.constant([1.0, 32.0, 243.0], dtype="float32"),
            ),
            (
                [tf.constant([5.0, 2.0]), tf.constant([2.0, 1.0])],
                "input_2",
                None,
                "float64",
                "float64",
                tf.constant([25.0, 2.0], dtype="float64"),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                3.0,
                "float64",
                "float32",
                tf.constant([[[3.375, 15.625, 27000.0]]]),
            ),
            (
                [
                    tf.constant([[7.0], [4.0], [3.0]]),
                    tf.constant([[1.0], [2.0], [3.1]]),
                ],
                "input_4",
                None,
                "float64",
                "float32",
                tf.constant([[7.0], [16.0], [30.135323]]),
            ),
            (
                [
                    tf.constant([[7.0], [4.0], [3.0]]),
                    tf.constant([[0.0], [2.0], [3.1]]),
                ],
                "input_4",
                None,
                "float64",
                "float32",
                tf.constant([[1.0], [16.0], [30.135323]]),
            ),
            (
                [
                    tf.constant(["2.0"]),
                    tf.constant(["5.0"]),
                ],
                "input_5",
                None,
                "float32",
                None,
                tf.constant([32.0]),
            ),
        ],
    )
    def test_exponent(
        self, inputs, input_name, exponent, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = ExponentLayer(
            name=input_name,
            exponent=exponent,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(inputs)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_near(output_tensor, expected_output)

    @pytest.mark.parametrize(
        "inputs, input_name, exponent",
        [
            (
                [tf.constant([1.0, 2.0, 3.0]), tf.constant([1.0, 2.0, 3.0])],
                "input_1",
                1.0,
            ),
            (
                [tf.constant([5.0, 2.0]), tf.constant([2.0, 10.0])],
                "input_2",
                10.0,
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                None,
            ),
        ],
    )
    def test_exponent_raises_error(self, inputs, input_name, exponent):
        with pytest.raises(ValueError):
            layer = ExponentLayer(name=input_name, exponent=exponent)
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
    def test_exponent_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = ExponentLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
