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

from kamae.tensorflow.layers import BloomEncodeLayer


class TestBloomEncode:
    @pytest.mark.parametrize(
        "input_tensor, input_name, num_bins, mask_value, num_hash_fns, feature_cardinality, use_heuristic_num_bins, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([["Mon"], ["Tue"], ["Wed"]]),
                "input_1",
                100,
                None,
                5,
                None,
                False,
                None,
                "int64",
                tf.constant(
                    [[72, 59, 14, 41, 91], [77, 53, 98, 95, 54], [77, 77, 90, 45, 15]],
                    dtype=tf.int64,
                ),
            ),
            (
                tf.constant([[["Mon", "Tue"], ["Wed", "Thu"]]]),
                "input_2",
                500,
                "Tue",
                3,
                None,
                False,
                None,
                "int32",
                tf.constant(
                    [[[[363, 176, 168], [0, 0, 0]], [[420, 345, 1], [452, 220, 77]]]],
                    dtype=tf.int32,
                ),
            ),
            (
                tf.constant([[["Sun"], ["Sat"], ["Fri"]]]),
                "input_3",
                67,
                None,
                2,
                100,
                True,
                "string",
                "int16",
                tf.constant([[[14, 7], [18, 10], [4, 9]]], dtype=tf.int16),
            ),
        ],
    )
    def test_bloom_encoder(
        self,
        input_tensor,
        input_name,
        num_bins,
        mask_value,
        num_hash_fns,
        feature_cardinality,
        use_heuristic_num_bins,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = BloomEncodeLayer(
            name=input_name,
            num_bins=num_bins,
            mask_value=mask_value,
            num_hash_fns=num_hash_fns,
            feature_cardinality=feature_cardinality,
            use_heuristic_num_bins=use_heuristic_num_bins,
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
                tf.constant([1.0, 2.0, 3.0], dtype="float32"),
                "input_1",
                None,
                "float64",
            )
        ],
    )
    def test_bloom_encode_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = BloomEncodeLayer(
            name=input_name,
            num_bins=100,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
