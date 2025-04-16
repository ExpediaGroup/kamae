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

from kamae.tensorflow.layers import MinLayer


class TestMin:
    @pytest.mark.parametrize(
        "inputs, input_name, min_constant, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1, 2, 3], dtype="int32"),
                "input_1",
                1.345,
                None,
                None,
                tf.constant([1.0, 1.345, 1.345], dtype="float32"),
            ),
            (
                [tf.constant([5.0, 2.0]), tf.constant([2.0, 10.0])],
                "input_2",
                None,
                None,
                None,
                tf.constant([2.0, 2.0]),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                3.0,
                "float64",
                None,
                tf.constant([[[1.5, 2.5, 3.0]]], dtype="float64"),
            ),
            (
                [
                    tf.constant([[7.0], [4.0], [3.0]]),
                    tf.constant([[1.0], [2.0], [3.1]]),
                    tf.constant([[56.0], [2.0], [1.0]]),
                    tf.constant([[0.0], [2.0], [7.6]]),
                ],
                "input_4",
                None,
                "int32",
                "string",
                tf.constant([["0"], ["2"], ["1"]]),
            ),
        ],
    )
    def test_min(
        self,
        inputs,
        input_name,
        min_constant,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = MinLayer(
            name=input_name,
            min_constant=min_constant,
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
        tf.debugging.assert_equal(output_tensor, expected_output)

    @pytest.mark.parametrize(
        "inputs, input_name, min_constant",
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
    def test_min_raises_error(self, inputs, input_name, min_constant):
        with pytest.raises(ValueError):
            layer = MinLayer(name=input_name, min_constant=min_constant)
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
    def test_min_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = MinLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
