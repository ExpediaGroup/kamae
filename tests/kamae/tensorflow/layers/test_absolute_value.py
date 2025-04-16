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

from kamae.tensorflow.layers import AbsoluteValueLayer


class TestAbsoluteValue:
    @pytest.mark.parametrize(
        "inputs, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1.0, 2.0, 3.0], dtype="float32"),
                "input_1",
                "float32",
                "float64",
                tf.constant([1.0, 2.0, 3.0], dtype="float64"),
            ),
            (
                tf.constant([-5.0, 2.0, 2.0, -10.0], dtype="float32"),
                "input_2",
                "float64",
                "int64",
                tf.constant([5, 2, 2, 10], dtype="int64"),
            ),
            (
                tf.constant([[[1, -2, 30]]], dtype="int64"),
                "input_3",
                "int32",
                "string",
                tf.constant([[["1", "2", "30"]]]),
            ),
        ],
    )
    def test_absolute_value(
        self, inputs, input_name, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = AbsoluteValueLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
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
    def test_absolute_value_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = AbsoluteValueLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
