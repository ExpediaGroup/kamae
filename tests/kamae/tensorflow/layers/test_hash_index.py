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

from kamae.tensorflow.layers import HashIndexLayer


class TestHashIndex:
    @pytest.mark.parametrize(
        "input_tensor, input_name, num_bins, mask_value, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_1",
                100,
                None,
                None,
                None,
                tf.constant([40, 99, 24], dtype=tf.int64),
            ),
            (
                tf.constant([[["Mon", "Tue"], ["Wed", "Thu"]]]),
                "input_2",
                500,
                "Tue",
                "string",
                "float64",
                tf.constant([[[18.0, 0.0], [276.0, 122.0]]], dtype=tf.float64),
            ),
            (
                tf.constant([[["Sun"], ["Sat"], ["Fri"]]]),
                "input_3",
                67,
                None,
                None,
                "float32",
                tf.constant([[[23.0], [48.0], [25.0]]], dtype=tf.float32),
            ),
            (
                tf.constant([[[0], [1000], [67.78]]]),
                "input_4",
                125,
                None,
                "string",
                "int64",
                tf.constant([[[106], [76], [16]]], dtype=tf.int64),
            ),
        ],
    )
    def test_hash_indexer(
        self,
        input_tensor,
        input_name,
        num_bins,
        mask_value,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = HashIndexLayer(
            name=input_name,
            num_bins=num_bins,
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
        tf.debugging.assert_equal(expected_output, output_tensor)

    @pytest.mark.parametrize(
        "inputs, input_name, input_dtype, output_dtype",
        [
            (
                tf.constant([1.0, 2.0, 3.0], dtype="float64"),
                "input_1",
                None,
                "float64",
            )
        ],
    )
    def test_hash_index_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = HashIndexLayer(
            name=input_name,
            num_bins=100,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
