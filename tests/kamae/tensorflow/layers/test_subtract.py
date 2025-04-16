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

from kamae.tensorflow.layers import SubtractLayer


class TestSubtract:
    @pytest.mark.parametrize(
        "inputs, input_name, subtrahend, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["1", "2", "3"]),
                "input_1",
                1.56,
                "int64",
                None,
                tf.constant([-0.56, 0.44, 1.44], dtype="float64"),
            ),
            (
                [tf.constant([5.0, 2.0]), tf.constant([2.0, 10.0])],
                "input_2",
                None,
                "int64",
                None,
                tf.constant([3, -8], dtype="int64"),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                3.0,
                "float32",
                None,
                tf.constant([[[-1.5, -0.5, 27.0]]]),
            ),
            (
                [
                    tf.constant([[7.0], [4.0], [3.0]]),
                    tf.constant([[1.0], [2.0], [3.1]]),
                ],
                "input_4",
                None,
                None,
                None,
                tf.constant([[6.0], [2.0], [-0.1]]),
            ),
            (
                [
                    tf.constant([8.0]),
                    tf.constant([5.0]),
                    tf.constant([1.0]),
                ],
                "input_5",
                None,
                "float64",
                "float32",
                tf.constant([2.0]),
            ),
        ],
    )
    def test_subtract(
        self, inputs, input_name, subtrahend, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = SubtractLayer(
            name=input_name,
            subtrahend=subtrahend,
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
        "inputs, input_name, subtrahend",
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
    def test_subtract_raises_error(self, inputs, input_name, subtrahend):
        with pytest.raises(ValueError):
            layer = SubtractLayer(name=input_name, subtrahend=subtrahend)
            layer(inputs)
