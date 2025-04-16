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

from kamae.tensorflow.layers import BucketizeLayer


class TestBucketize:
    @pytest.mark.parametrize(
        "input_tensor, input_name, splits, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1.0, 2.0, 3.0]),
                "input_1",
                [2.0, 3.0, 5.0],
                None,
                None,
                tf.constant([1, 2, 3], dtype="int64"),
            ),
            (
                tf.constant([5, 2], dtype="int32"),
                "input_2",
                [1.0, 5.0, 7.0, 7.5],
                "int64",
                "float32",
                tf.constant([3.0, 2.0], dtype="float32"),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                [3.0, 4.0, 5.0, 31.0],
                "float32",
                "string",
                tf.constant([[["1", "1", "4"]]]),
            ),
            (
                tf.constant([[7.0], [4.0], [3.0]]),
                "input_4",
                [2.0, 6.0, 7.0, 9.0],
                "float32",
                "int64",
                tf.constant([[4], [2], [2]], dtype="int64"),
            ),
        ],
    )
    def test_bucketizer(
        self,
        input_tensor,
        input_name,
        splits,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = BucketizeLayer(
            name=input_name,
            splits=splits,
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
    def test_bucketize_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = BucketizeLayer(
            name=input_name,
            splits=[1.0, 2.0, 3.0],
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
