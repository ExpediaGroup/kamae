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

from kamae.tensorflow.layers import ArrayCropLayer


class TestArrayCrop:
    @pytest.mark.parametrize(
        "input_tensor, dtype, array_length, pad_value, input_name, expected_output",
        [
            (
                tf.constant(["a", "a", "b", "c"]),
                tf.string,
                3,
                "-1",
                "input_0",
                tf.constant(["a", "a", "b"]),
            ),
            (
                tf.constant([["a", "a", "b", "c"]]),
                tf.string,
                3,
                "-1",
                "input_1",
                tf.constant([["a", "a", "b"]]),
            ),
            (
                tf.constant([["a", "a", "b", "c"]]),
                tf.string,
                5,
                "-1",
                "input_2",
                tf.constant([["a", "a", "b", "c", "-1"]]),
            ),
            (
                tf.constant([[1, 2, 3, 4]]),
                tf.int32,
                3,
                -1,
                "input_3",
                tf.constant([[1, 2, 3]]),
            ),
            (
                tf.constant([[1, 2, 3, 4]]),
                tf.int32,
                5,
                -1,
                "input_4",
                tf.constant([[1, 2, 3, 4, -1]]),
            ),
            (
                tf.constant([[1.0, 2.0, 3.0, 4.0]]),
                tf.float32,
                5,
                -1.0,
                "input_5",
                tf.constant([[1.0, 2.0, 3.0, 4.0, -1.0]]),
            ),
            (
                tf.constant([[["a", "a", "b", "c"], ["x", "y", "z", "z"]]]),
                tf.string,
                3,
                "-1",
                "input_6",
                tf.constant([[["a", "a", "b"], ["x", "y", "z"]]]),
            ),
            (
                tf.constant([[["a", "a", "b", "c"], ["x", "y", "z", "z"]]]),
                tf.string,
                5,
                "-1",
                "input_7",
                tf.constant([[["a", "a", "b", "c", "-1"], ["x", "y", "z", "z", "-1"]]]),
            ),
            (
                tf.constant([[[1, 2, 3, 4], [9, 2, 7, 8]]]),
                tf.int32,
                3,
                -1,
                "input_8",
                tf.constant([[[1, 2, 3], [9, 2, 7]]]),
            ),
            (
                tf.constant([[[1, 2], [9, 2]]]),
                tf.int32,
                5,
                -1,
                "input_8",
                tf.constant([[[1, 2, -1, -1, -1], [9, 2, -1, -1, -1]]]),
            ),
            (
                tf.constant([[[["a", "a", "b", "c"]]]]),
                tf.string,
                3,
                "-1",
                "input_10",
                tf.constant([[[["a", "a", "b"]]]]),
            ),
            (
                tf.constant([[[["a", "a", "b", "c"]]]]),
                tf.string,
                5,
                "-1",
                "input_11",
                tf.constant([[[["a", "a", "b", "c", "-1"]]]]),
            ),
            (
                tf.constant([[]]),
                tf.string,
                3,
                "-1",
                "input_12",
                tf.constant([["-1", "-1", "-1"]]),
            ),
            (
                tf.constant([[]]),
                tf.int32,
                3,
                -1,
                "input_13",
                tf.constant([[-1, -1, -1]]),
            ),
            (
                tf.constant([[9, 88, 11]]),
                tf.int32,
                1,
                -1,
                "input_14",
                tf.constant([[9]]),
            ),
            (
                tf.constant(
                    [[1687087026136, 1687087026136, 1687087026136, 1687087026136]]
                ),
                tf.int64,
                5,
                -1,
                "input_5",
                tf.constant(
                    [[1687087026136, 1687087026136, 1687087026136, 1687087026136, -1]]
                ),
            ),
        ],
    )
    def test_array_crop_layer(
        self, input_tensor, dtype, array_length, pad_value, input_name, expected_output
    ):
        # when
        layer = ArrayCropLayer(
            input_dtype=dtype,
            output_dtype=dtype,
            array_length=array_length,
            pad_value=pad_value,
            name=input_name,
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
        tf.debugging.assert_equal(expected_output, output_tensor)

    @pytest.mark.parametrize(
        "input_tensor, array_length, pad_value, input_name",
        [
            (
                tf.constant([["a", "a", "b", "c"]]),
                3,
                -1,
                "input_1",
            ),
            (
                tf.constant([["a", "a", "b", "c"]]),
                3,
                -1.0,
                "input_2",
            ),
            (
                tf.constant([[1, 2, 3, 4]]),
                3,
                "-1",
                "input_3",
            ),
            (
                tf.constant([[1, 2, 3, 4]]),
                3,
                1.2,
                "input_3",
            ),
        ],
    )
    def test_array_crop_throws_error_when_pad_type_does_not_match_tensor(
        self, input_tensor, array_length, pad_value, input_name
    ):
        # when
        layer = ArrayCropLayer(
            array_length=array_length, pad_value=pad_value, name=input_name
        )
        with pytest.raises(TypeError):
            _ = layer(input_tensor)

    @pytest.mark.parametrize(
        "array_length, pad_value, input_name",
        [
            (
                -3,
                "-1",
                "input_1",
            ),
            (
                0,
                -1,
                "input_2",
            ),
        ],
    )
    def test_array_crop_throws_error_with_negative_array_length(
        self, array_length, pad_value, input_name
    ):
        with pytest.raises(ValueError):
            _ = ArrayCropLayer(
                array_length=array_length, pad_value=pad_value, name=input_name
            )

    @pytest.mark.parametrize(
        "array_length, pad_value, input_name",
        [
            (
                3,
                None,
                "input_1",
            ),
        ],
    )
    def test_array_crop_throws_error_with_invalid_pad_value(
        self, array_length, pad_value, input_name
    ):
        with pytest.raises(ValueError):
            _ = ArrayCropLayer(
                array_length=array_length, pad_value=pad_value, name=input_name
            )
