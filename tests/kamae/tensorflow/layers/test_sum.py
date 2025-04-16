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

from kamae.tensorflow.layers import SumLayer


class TestSum:
    @pytest.mark.parametrize(
        "inputs, input_name, addend, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["1", "2", "3"]),
                "input_1",
                1.23,
                "int64",
                None,
                tf.constant([2.23, 3.23, 4.23], dtype="float64"),
            ),
            (
                [tf.constant([5.0, 2.0]), tf.constant([2.0, 10.0])],
                "input_2",
                None,
                None,
                None,
                tf.constant([7.0, 12.0]),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                3.0,
                "float32",
                "float16",
                tf.constant([[[4.5, 5.5, 33.0]]], dtype="float16"),
            ),
            (
                [
                    tf.constant([[7.0], [4.0], [3.0]]),
                    tf.constant([[1.0], [2.0], [3.0]]),
                    tf.constant([[56.0], [23.0], [31.0]]),
                    tf.constant([[0.0], [2.0], [7.0]]),
                ],
                "input_4",
                None,
                "int32",
                "float32",
                tf.constant([[64.0], [31.0], [44.0]]),
            ),
        ],
    )
    def test_sum(
        self, inputs, input_name, addend, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = SumLayer(
            name=input_name,
            addend=addend,
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
        "inputs, input_name, addend",
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
    def test_sum_raises_error(self, inputs, input_name, addend):
        with pytest.raises(ValueError):
            layer = SumLayer(name=input_name, addend=addend)
            layer(inputs)
