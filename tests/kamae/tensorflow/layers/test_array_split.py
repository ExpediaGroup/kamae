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

from kamae.tensorflow.layers import ArraySplitLayer


class TestArraySplit:
    @pytest.mark.parametrize(
        "input_tensor, input_name, axis, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1, 2, 3, 4, 5, 6], dtype="int32"),
                "input_1",
                -1,
                "float32",
                None,
                [
                    tf.constant([1], dtype="float32"),
                    tf.constant([2], dtype="float32"),
                    tf.constant([3], dtype="float32"),
                    tf.constant([4], dtype="float32"),
                    tf.constant([5], dtype="float32"),
                    tf.constant([6], dtype="float32"),
                ],
            ),
            (
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="float32"),
                "input_2",
                -1,
                None,
                "int32",
                [
                    tf.constant([[1], [4]], dtype="int32"),
                    tf.constant([[2], [5]], dtype="int32"),
                    tf.constant([[3], [6]], dtype="int32"),
                ],
            ),
            (
                tf.constant([[1, 2, 3], [4, 5, 6]]),
                "input_3",
                0,
                "int32",
                "string",
                [tf.constant([["1", "2", "3"]]), tf.constant([["4", "5", "6"]])],
            ),
        ],
    )
    def test_split(
        self, input_tensor, input_name, axis, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = ArraySplitLayer(
            name=input_name,
            axis=axis,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(input_tensor)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        for expected, output in zip(expected_output, output_tensor):
            assert (
                output.dtype == expected.dtype
            ), "Output tensor dtype is not the same as expected tensor dtype"
            assert (
                output.shape == expected.shape
            ), "Output tensor shape is not the same as expected tensor shape"

        tf.debugging.assert_equal(expected_output, output_tensor)
