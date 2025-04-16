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

from kamae.tensorflow.layers import StringIndexLayer


class TestStringIndex:
    @pytest.mark.parametrize(
        "input_tensor, input_name, vocabulary, mask_token, num_oov_indices, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_1",
                ["Mon", "Tue", "Wed"],
                None,
                1,
                None,
                None,
                tf.constant([1, 2, 3], dtype=tf.int64),
            ),
            (
                tf.constant([[["Mon", "Tue"], ["Wed", "Thu"]]]),
                "input_2",
                tf.constant(["Mon", "Tue", "Wed"]),
                None,
                1,
                None,
                None,
                tf.constant([[[1, 2], [3, 0]]], dtype=tf.int64),
            ),
            (
                tf.constant([[["Sun"], ["Sat"], ["Fri"]]]),
                "input_3",
                tf.constant(["Mon", "Tue", "Fri"]),
                None,
                1,
                None,
                None,
                tf.constant([[[0], [0], [3]]], dtype=tf.int64),
            ),
            (
                tf.constant(["Mon", "Tue", "Wed"]),
                "input_4",
                ["Mon", "Tue"],
                "Wed",
                2,
                None,
                None,
                tf.constant([3, 4, 0], dtype=tf.int64),
            ),
            (
                tf.constant([[["Mon", "Tue"], ["Wed", "Sat"]]]),
                "input_5",
                tf.constant(["Mon", "Wed"]),
                "Tue",
                3,
                None,
                None,
                tf.constant([[[4, 0], [5, 1]]], dtype=tf.int64),
            ),
            (
                tf.constant([[["Sun"], ["Sat"], ["Fri"]]]),
                "input_6",
                tf.constant(["Mon", "Tue"]),
                "Fri",
                6,
                None,
                None,
                tf.constant([[[4], [1], [0]]], dtype=tf.int64),
            ),
            (
                tf.constant([1, 2, 1], dtype=tf.int32),
                "input_7",
                ["1", "3"],
                None,
                1,
                "string",
                "bool",
                tf.constant([True, False, True], dtype="bool"),
            ),
        ],
    )
    def test_string_indexer(
        self,
        input_tensor,
        input_name,
        vocabulary,
        mask_token,
        num_oov_indices,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = StringIndexLayer(
            name=input_name,
            vocabulary=vocabulary,
            mask_token=mask_token,
            num_oov_indices=num_oov_indices,
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
