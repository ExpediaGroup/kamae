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

from kamae.tensorflow.layers import MultiplyLayer


class TestMultiply:
    @pytest.mark.parametrize(
        "inputs, input_name, multiplier, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1, 2, 3], dtype="int64"),
                "input_1",
                5.5,
                None,
                None,
                tf.constant([5.5, 11.0, 16.5], dtype="float64"),
            ),
            (
                [tf.constant([5.0, 2.0]), tf.constant([2.0, 10.0])],
                "input_2",
                None,
                "float64",
                "string",
                tf.constant(["10.0", "20.0"]),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                3.0,
                "float16",
                "float64",
                tf.constant([[[4.5, 7.5, 90.0]]], dtype="float64"),
            ),
            (
                [
                    tf.constant([[7.0], [4.0], [3.0]]),
                    tf.constant([[1.0], [2.0], [3.0]]),
                ],
                "input_4",
                None,
                "int64",
                "int16",
                tf.constant([[7], [8], [9]], dtype="int16"),
            ),
            (
                [tf.constant([5.0]), tf.constant([2.0]), tf.constant([3.0])],
                "input_4",
                None,
                None,
                None,
                tf.constant([30.0]),
            ),
        ],
    )
    def test_multiply(
        self, inputs, input_name, multiplier, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = MultiplyLayer(
            name=input_name,
            multiplier=multiplier,
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
        if expected_output.dtype.is_floating:
            tf.debugging.assert_near(output_tensor, expected_output)
        else:
            tf.debugging.assert_equal(output_tensor, expected_output)

    @pytest.mark.parametrize(
        "inputs, input_name, multiplier",
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
    def test_multiply_raises_error(self, inputs, input_name, multiplier):
        with pytest.raises(ValueError):
            layer = MultiplyLayer(name=input_name, multiplier=multiplier)
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
    def test_multiply_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = MultiplyLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
