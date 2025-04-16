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

from kamae.tensorflow.layers import IdentityLayer


class TestIdentity:
    @pytest.mark.parametrize(
        "input_tensor, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1, 2, 3], dtype="float32"),
                "input_1",
                "float64",
                None,
                tf.constant([1, 2, 3], dtype="float64"),
            ),
            (
                tf.constant([1, 2, 3], dtype="int32"),
                "input_2",
                None,
                "string",
                tf.constant(["1", "2", "3"], dtype="string"),
            ),
            (
                tf.constant(["hello", "world"], dtype="string"),
                "input_3",
                None,
                None,
                tf.constant(["hello", "world"], dtype="string"),
            ),
            (
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="float32"),
                "input_4",
                "int32",
                "int32",
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="int32"),
            ),
            (
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="int32"),
                "input_5",
                "float32",
                "float64",
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="float64"),
            ),
            (
                tf.constant([["hello", "world"], ["hello", "world"]], dtype="string"),
                "input_6",
                "string",
                None,
                tf.constant([["hello", "world"], ["hello", "world"]], dtype="string"),
            ),
        ],
    )
    def test_identity(
        self, input_tensor, input_name, input_dtype, output_dtype, expected_output
    ):
        # when
        layer = IdentityLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        output_tensor = layer(input_tensor)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            expected_output.dtype == output_tensor.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            expected_output.shape == output_tensor.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(expected_output, output_tensor)
