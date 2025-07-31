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

from kamae.tensorflow.layers import MinMaxScaleLayer


class TestMinMaxScale:
    @pytest.mark.parametrize(
        "input_tensor, input_name, min, max, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[["1.0", "4.0", "7.0"]], [["3.0", "6.0", "4.0"]]]),
                "input_1",
                [1.0, 2.0, 3.0],
                [5.0, 10.0, 7.0],
                "float64",
                "string",
                tf.constant([[["0.0", "0.25", "1.0"]], [["0.5", "0.5", "0.25"]]]),
            ),
            (
                tf.constant([[["1.0", "4.0", "7.0"]]]),
                "input_1",
                [1.0, 4.0, 2.0],
                [10.0, 5.0, 7.0],
                "float64",
                "string",
                tf.constant([[["0.0", "0.0", "1.0"]]]),
            ),
            (
                tf.constant([[0.0, 5.0, 1.0], [3.0, 1.0, 2.0], [4.0, 5.0, 1.0]]),
                "input_2",
                [-25.0, -1.0, -8.0],
                [100.0, 5.0, 10.0],
                None,
                None,
                tf.constant(
                    [
                        [0.2, 1.0, 0.5],
                        [0.224, 0.33333333, 0.555555556],
                        [0.232, 1.0, 0.5],
                    ]
                ),
            ),
            (
                tf.constant([[[[[-1.0, 7.0, 8.0]]]]]),
                "input_3",
                [-1.0, 2.0, 3.0],
                [64.0, 16.0, 43.0],
                "float64",
                "float16",
                tf.constant([[[[[0.0, 0.35714285714285715, 0.125]]]]], dtype="float16"),
            ),
        ],
    )
    def test_min_max_scale(
        self,
        input_tensor,
        input_name,
        min,
        max,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = MinMaxScaleLayer(
            name=input_name,
            min=min,
            max=max,
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
        "input_tensor, input_name, min, max, mask_value, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([[[1.0, 4.0, 7.0]], [[3.0, 6.0, 4.0]]]),
                "input_1",
                [1.0, 2.0, 3.0],
                [25.0, 64.0, 16.0],
                4.0,
                None,
                None,
                tf.constant(
                    [
                        [[0.0, 4.0, 0.30769232]],
                        [[0.08333333333333333, 0.06451612903225806, 4.0]],
                    ]
                ),
            ),
            (
                tf.constant([[2.0, 5.0, 1.0], [3.0, 1.0, 2.0], [4.0, 5.0, 1.0]]),
                "input_2",
                [2.0, 1.0, -8.0],
                [5.0, 10.0, 15.0],
                1.0,
                "float32",
                "float64",
                tf.constant(
                    [
                        [0.0, 0.4444444444444444, 1.0],
                        [0.3333333333333333, 1.0, 0.43478260869565216],
                        [0.6666666666666666, 0.4444444444444444, 1.0],
                    ],
                    dtype="float64",
                ),
            ),
            (
                tf.constant([[[[[-1.0, 7.0, 8.0]]]]]),
                "input_3",
                [-1.0, 2.0, 3.0],
                [4.0, 16.0, 4.0],
                8.0,
                "float16",
                "float32",
                tf.constant([[[[[0.0, 0.35717773, 8.0]]]]]),
            ),
        ],
    )
    def test_min_max_scale_with_masking(
        self,
        input_tensor,
        input_name,
        min,
        max,
        mask_value,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = MinMaxScaleLayer(
            name=input_name,
            min=min,
            max=max,
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
    def test_min_max_scale_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = MinMaxScaleLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            min=[1.0, 2.0, 3.0],
            max=[4.0, 5.0, 6.0],
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
