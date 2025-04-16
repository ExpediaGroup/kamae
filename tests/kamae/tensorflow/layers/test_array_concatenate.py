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

from kamae.tensorflow.layers import ArrayConcatenateLayer


class TestArrayConcatenate:
    @pytest.mark.parametrize(
        "input_tensor_list, input_name, axis, auto_broadcast, input_dtype, output_dtype, expected_output",
        [
            (
                [
                    tf.constant([1, 2, 3], dtype="int32"),
                    tf.constant([4, 5, 6], dtype="int64"),
                ],
                "input_1",
                -1,
                False,
                "int32",
                None,
                tf.constant([1, 2, 3, 4, 5, 6], dtype="int32"),
            ),
            (
                [
                    tf.constant([[1, 2], [3, 4]], dtype="int64"),
                    tf.constant([[5, 6], [6, 7]], dtype="int32"),
                ],
                "input_2",
                -1,
                False,
                "float32",
                "float64",
                tf.constant([[1, 2, 5, 6], [3, 4, 6, 7]], dtype="float64"),
            ),
            (
                [tf.constant([[1, 2, 3]]), tf.constant([[4, 5, 6]])],
                "input_3",
                0,
                False,
                None,
                "int32",
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="int32"),
            ),
            # test auto broadcasting
            (
                [
                    tf.constant([[[1, 2], [3, 4]]], dtype="int64"),
                    tf.constant([[5, 6]], dtype="int32"),
                ],
                "input_2",
                -1,
                True,
                "float32",
                "float64",
                tf.constant([[[1, 2, 5, 6], [3, 4, 5, 6]]], dtype="float64"),
            ),
            (
                [
                    tf.constant([[[1, 2], [3, 4]]], dtype="int64"),
                    tf.constant([5], dtype="int32"),
                ],
                "input_2",
                -1,
                True,
                "float32",
                "float64",
                tf.constant([[[1, 2, 5], [3, 4, 5]]], dtype="float64"),
            ),
            # reverse broadcasting (broadcasting the first to the second)
            (
                [
                    tf.constant([[5, 6]], dtype="int32"),
                    tf.constant([[[1, 2], [3, 4]]], dtype="int64"),
                ],
                "input_2",
                -1,
                True,
                "float32",
                "float64",
                tf.constant([[[5, 6, 1, 2], [5, 6, 3, 4]]], dtype="float64"),
            ),
            (
                [
                    tf.constant([5], dtype="int32"),
                    tf.constant([[[1, 2], [3, 4]]], dtype="int64"),
                ],
                "input_2",
                -1,
                True,
                "float32",
                "float64",
                tf.constant([[[5, 1, 2], [5, 3, 4]]], dtype="float64"),
            ),
            # simple broadcasting
            (
                [
                    tf.constant([[[1, 2], [3, 4]]], dtype="int64"),
                    tf.constant([[[5], [6]]], dtype="int32"),
                ],
                "input_2",
                -1,
                True,
                "float32",
                "float64",
                tf.constant([[[1, 2, 5], [3, 4, 6]]], dtype="float64"),
            ),
        ],
    )
    def test_concat(
        self,
        input_tensor_list,
        input_name,
        axis,
        auto_broadcast,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = ArrayConcatenateLayer(
            name=input_name,
            axis=axis,
            auto_broadcast=auto_broadcast,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(input_tensor_list)
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
        "input_tensor_list, axis, auto_broadcast, expected_output",
        [
            (
                [
                    tf.constant([[[1, 2], [3, 4]]], dtype="int64"),
                    tf.constant([[[5], [6]]], dtype="int32"),
                ],
                -2,
                True,
                tf.constant([[[1, 2, 5], [3, 4, 6]]], dtype="float64"),
            ),
        ],
    )
    def test_concat_with_bad_axes(
        self,
        input_tensor_list,
        axis,
        auto_broadcast,
        expected_output,
    ):
        with pytest.raises(ValueError):
            _ = ArrayConcatenateLayer(
                name="test",
                axis=axis,
                auto_broadcast=auto_broadcast,
            )
