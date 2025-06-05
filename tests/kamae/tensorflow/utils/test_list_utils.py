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

from kamae.tensorflow.utils import get_top_n


class TestGetTopN:
    @pytest.mark.parametrize(
        "val_tensor, sort_tensor, axis, top_n, sort_order, expected_output",
        [
            (
                tf.constant([[10, 20, 30], [1, 2, 3]], dtype=tf.int32),  # shape [2, 3]
                tf.constant(
                    [[0.1, 0.4, 0.3], [0.5, 0.2, 0.9]], dtype=tf.float32
                ),  # shape [2, 3]
                1,
                2,
                "desc",
                tf.constant([[20, 30], [3, 1]], dtype=tf.int32),  # shape [2, 2]
            ),
            (
                tf.constant([[10, 20, 30], [1, 2, 3]], dtype=tf.int32),  # shape [2, 3]
                tf.constant([[0.1, 0.4, 0.3], [0.5, 0.2, 0.9]], dtype=tf.float32),
                # shape [2, 3]
                1,
                2,
                "asc",
                tf.constant([[10, 30], [2, 1]], dtype=tf.int32),  # shape [2, 2]
            ),
        ],
    )
    def test_get_top_n(
        self, val_tensor, sort_tensor, axis, top_n, sort_order, expected_output
    ):
        output = get_top_n(val_tensor, axis, sort_tensor, top_n, sort_order)
        tf.debugging.assert_equal(output, expected_output)

    @pytest.mark.parametrize(
        "val_tensor, sort_tensor, axis, top_n, sort_order, expected_output",
        [
            (
                tf.constant([[[10], [20], [30]]], dtype=tf.int32),  # shape [1, 3, 1]
                tf.constant(
                    [[[0.1], [0.4], [0.3]]], dtype=tf.float32
                ),  # shape [1, 3, 1]
                1,
                2,
                "desc",
                tf.constant([[[20], [30]]], dtype=tf.int32),  # shape [1, 2, 1]
            )
        ],
    )
    def test_get_top_with_batch_1(
        self, val_tensor, sort_tensor, axis, top_n, sort_order, expected_output
    ):
        output = get_top_n(val_tensor, axis, sort_tensor, top_n, sort_order)
        tf.debugging.assert_equal(output, expected_output)
