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

from kamae.tensorflow.layers import ExpLayer


class TestExp:
    @pytest.mark.parametrize(
        "input_tensor, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["1.0", "2.0", "3.0"]),
                "input_1",
                "float32",
                "float32",
                tf.constant([2.7182817, 7.389056, 20.085537]),
            ),
            (
                tf.constant([5.0, 2.0]),
                "input_2",
                "float64",
                None,
                tf.constant([148.4131591025766, 7.38905609893065], dtype="float64"),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                None,
                "float64",
                tf.constant(
                    [[[4.481688976287842, 12.182494163513184, 10686474223616.0]]],
                    dtype="float64",
                ),
            ),
        ],
    )
    def test_exp(
        self, input_tensor, input_name, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = ExpLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
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
        tf.debugging.assert_near(output_tensor, expected_output)

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
    def test_exp_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = ExpLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
