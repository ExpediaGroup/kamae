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

from kamae.tensorflow.layers import ListRankLayer


class TestListRank:
    @pytest.mark.parametrize(
        "input_tensor, expected_output, input_dtype, output_dtype, axis",
        [
            # No ties, no casting, default axis
            (
                tf.constant(
                    [[[1.0], [1.5], [9.0], [4.0], [6.0], [2.0], [0.5], [0.0]]],
                    dtype=tf.float32,
                ),
                tf.constant([[[6], [5], [1], [3], [2], [4], [7], [8]]], dtype=tf.int32),
                None,
                None,
                1,
            ),
            # With ties, casting, default axis
            (
                tf.constant(
                    [[[1.0], [1.5], [9.0], [4.0], [6.0], [2.0], [0.5], [0.0], [0.0]]],
                    dtype=tf.float32,
                ),
                tf.constant(
                    [[["6"], ["5"], ["1"], ["3"], ["2"], ["4"], ["7"], ["8"], ["9"]]],
                    dtype=tf.string,
                ),
                "float32",
                "string",
                1,
            ),
            # Test different axes - axis 1
            (
                tf.constant([[[1.0], [2.0]], [[4.0], [3.0]]], dtype=tf.float32),
                tf.constant([[[2], [1]], [[1], [2]]], dtype=tf.int32),
                None,
                None,
                1,
            ),
            # Test different axes - axis 0
            (
                tf.constant([[[1.0], [2.0]], [[4.0], [3.0]]], dtype=tf.float32),
                tf.constant([[[2], [2]], [[1], [1]]], dtype=tf.int32),
                None,
                None,
                0,
            ),
        ],
    )
    def test_listwise_rank(
        self, input_tensor, expected_output, input_dtype, output_dtype, axis
    ):
        # when
        name = "listwise_rank_test"
        layer = ListRankLayer(
            name=name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            axis=axis,
        )
        output_tensor = layer(input_tensor)
        # then
        assert layer.name == name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(output_tensor, expected_output)
