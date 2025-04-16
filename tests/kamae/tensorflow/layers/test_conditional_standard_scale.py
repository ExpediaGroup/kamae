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

from kamae.tensorflow.layers import ConditionalStandardScaleLayer


class TestConditionalStandardScale:
    @pytest.mark.parametrize(
        "input_tensor, input_name, mean, variance, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[[1.0, 4.0, 7.0]], [[3.0, 6.0, 4.0]]]),
                "input_1",
                [1.0, 2.0, 3.0],
                [25.0, 64.0, 16.0],
                None,
                None,
                tf.constant([[[0.0, 0.25, 1.0]], [[0.4, 0.5, 0.25]]]),
            ),
            (
                tf.constant([[[1.0, 4.0, 7.0]]]),
                "input_1",
                [1.0, 4.0, 7.0],
                [0.0, 0.0, 0.0],
                None,
                None,
                tf.constant([[[0.0, 0.0, 0.0]]]),
            ),
            (
                tf.constant([[2.0, 5.0, 1.0], [3.0, 1.0, 2.0], [4.0, 5.0, 1.0]]),
                "input_2",
                [2.0, 1.0, 8.0],
                [9.3025, 11.9716, 2.9929],
                "float64",
                "float32",
                tf.constant(
                    [
                        [0.0, 1.15606936, -4.04624277],
                        [0.3278688, 0.0, -3.4682080],
                        [0.6557377, 1.15606936, -4.04624277],
                    ],
                    dtype="float32",
                ),
            ),
            (
                tf.constant([[[[[-1.0, 7.0, 8.0]]]]]),
                "input_3",
                [1.0, 2.0, 3.0],
                [64.0, 16.0, 4.0],
                "float16",
                "float64",
                tf.constant([[[[[-0.25, 1.25, 2.5]]]]], dtype="float64"),
            ),
        ],
    )
    def test_conditional_standard_scaling(
        self,
        input_tensor,
        input_name,
        mean,
        variance,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = ConditionalStandardScaleLayer(
            name=input_name,
            mean=mean,
            variance=variance,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(input_tensor)
        conf = layer.get_config()
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_near(expected_output, output_tensor)

    @pytest.mark.parametrize(
        "input_tensor, input_name, mean, variance, skip_zeros, epsilon, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[[[-1.0, 7.0, 0.0]]]]),
                "input_3",
                [1.0, 2.0, 3.0],
                [64.0, 16.0, 4.0],
                False,
                0.0,
                None,
                None,
                tf.constant([[[[-0.25, 1.25, -1.5]]]]),
            ),
            (
                tf.constant([[[[-1.0, 7.0, 0.0]]]]),
                "input_3",
                [1.0, 2.0, 3.0],
                [64.0, 16.0, 4.0],
                True,
                0.0,
                "float64",
                "string",
                tf.constant([[[["-0.25", "1.25", "0.0"]]]]),
            ),
            (
                tf.constant([[[[-0.000001, 7.0, 0.0]]]]),
                "input_3",
                [1.0, 2.0, 3.0],
                [64.0, 16.0, 4.0],
                True,
                0.0001,
                "float64",
                "string",
                tf.constant([[[["0.0", "1.25", "0.0"]]]]),
            ),
        ],
    )
    def test_conditional_standard_scaling_with_skip_zeros(
        self,
        input_tensor,
        input_name,
        mean,
        variance,
        skip_zeros,
        epsilon,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = ConditionalStandardScaleLayer(
            name=input_name,
            mean=mean,
            variance=variance,
            skip_zeros=skip_zeros,
            epsilon=epsilon,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        conf = layer.get_config()
        output_tensor = layer(input_tensor)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        if expected_output.dtype == "string":
            tf.debugging.assert_equal(expected_output, output_tensor)
        else:
            tf.debugging.assert_near(expected_output, output_tensor)

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
    def test_conditional_standard_scale_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = ConditionalStandardScaleLayer(
            name=input_name,
            mean=[1.0, 2.0, 3.0],
            variance=[25.0, 64.0, 16.0],
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
