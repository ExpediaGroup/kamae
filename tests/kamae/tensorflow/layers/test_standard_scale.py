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

from kamae.tensorflow.layers import StandardScaleLayer


class TestStandardScale:
    @pytest.mark.parametrize(
        "input_tensor, input_name, mean, variance, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[["1.0", "4.0", "7.0"]], [["3.0", "6.0", "4.0"]]]),
                "input_1",
                [1.0, 2.0, 3.0],
                [25.0, 64.0, 16.0],
                "float64",
                "string",
                tf.constant([[["0.0", "0.25", "1.0"]], [["0.4", "0.5", "0.25"]]]),
            ),
            (
                tf.constant([[["1.0", "4.0", "7.0"]]]),
                "input_1",
                [1.0, 4.0, 7.0],
                [0.0, 0.0, 0.0],
                "float64",
                "string",
                tf.constant([[["0.0", "0.0", "0.0"]]]),
            ),
            (
                tf.constant([[2.0, 5.0, 1.0], [3.0, 1.0, 2.0], [4.0, 5.0, 1.0]]),
                "input_2",
                [2.0, 1.0, 8.0],
                [9.3025, 11.9716, 2.9929],
                None,
                None,
                tf.constant(
                    [
                        [0.0, 1.1560693979263306, -4.04624277],
                        [0.3278688, 0.0, -3.4682080],
                        [0.6557377, 1.1560693979263306, -4.04624277],
                    ]
                ),
            ),
            (
                tf.constant([[[[[-1.0, 7.0, 8.0]]]]]),
                "input_3",
                [1.0, 2.0, 3.0],
                [64.0, 16.0, 4.0],
                "float64",
                "float16",
                tf.constant([[[[[-0.25, 1.25, 2.5]]]]], dtype="float16"),
            ),
        ],
    )
    def test_standard_scaling(
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
        layer = StandardScaleLayer(
            name=input_name,
            mean=mean,
            variance=variance,
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
        "input_tensor, input_name, mean, variance, mask_value, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[[1.0, 4.0, 7.0]], [[3.0, 6.0, 4.0]]]),
                "input_1",
                [1.0, 2.0, 3.0],
                [25.0, 64.0, 16.0],
                4.0,
                None,
                None,
                tf.constant([[[0.0, 4.0, 1.0]], [[0.4, 0.5, 4.0]]]),
            ),
            (
                tf.constant([[2.0, 5.0, 1.0], [3.0, 1.0, 2.0], [4.0, 5.0, 1.0]]),
                "input_2",
                [2.0, 1.0, 8.0],
                [9.3025, 11.9716, 2.9929],
                1.0,
                "float32",
                "float64",
                tf.constant(
                    [
                        [0.0, 1.1560694, 1.0],
                        [0.32786885, 1.0, -3.46820807],
                        [0.6557377, 1.1560694, 1.0],
                    ],
                    dtype="float64",
                ),
            ),
            (
                tf.constant([[[[[-1.0, 7.0, 8.0]]]]]),
                "input_3",
                [1.0, 2.0, 3.0],
                [64.0, 16.0, 4.0],
                8.0,
                "float16",
                "float32",
                tf.constant([[[[[-0.25, 1.25, 8.0]]]]]),
            ),
        ],
    )
    def test_standard_scaling_with_masking(
        self,
        input_tensor,
        input_name,
        mean,
        variance,
        mask_value,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StandardScaleLayer(
            name=input_name,
            mean=mean,
            variance=variance,
            mask_value=mask_value,
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
        tf.debugging.assert_near(expected_output, output_tensor, atol=1e-6)

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
    def test_standard_scale_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = StandardScaleLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            mean=[1.0, 2.0, 3.0],
            variance=[4.0, 5.0, 6.0],
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
