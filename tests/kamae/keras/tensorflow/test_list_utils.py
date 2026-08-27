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

from kamae.keras.tensorflow.utils import get_top_n, min_filter_mask


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


class TestMinFilterMask:
    @pytest.mark.parametrize(
        "dtype, min_filter_value, expected",
        [
            # Integer values cannot be compared against a float threshold directly,
            # and rounding up leaves >= unchanged.
            (tf.int32, 0.0, [False, True, True]),
            (tf.int32, 0.5, [False, False, True]),
            (tf.int32, 1.0, [False, False, True]),
            (tf.int32, -0.5, [False, True, True]),
            # Floats keep the fractional threshold as given.
            (tf.float32, 0.5, [False, False, True]),
            (tf.float64, -0.5, [False, True, True]),
        ],
    )
    def test_min_filter_mask_threshold(self, dtype, min_filter_value, expected):
        val_tensor = tf.constant([-1, 0, 1], dtype=dtype)
        output = min_filter_mask(val_tensor, min_filter_value)
        tf.debugging.assert_equal(output, tf.constant(expected, dtype=tf.bool))

    @pytest.mark.parametrize(
        "min_filter_value, expected",
        [
            # Narrowing these to int8 would wrap around, so they are answered without
            # a cast: below the dtype minimum keeps everything, above the dtype
            # maximum keeps nothing.
            (-200.0, [True, True, True]),
            (-129.0, [True, True, True]),
            (-128.0, [True, True, True]),
            (127.0, [False, False, False]),
            (128.0, [False, False, False]),
            (200.0, [False, False, False]),
        ],
    )
    def test_min_filter_mask_threshold_outside_dtype_range(
        self, min_filter_value, expected
    ):
        val_tensor = tf.constant([1, 2, 3], dtype=tf.int8)
        output = min_filter_mask(val_tensor, min_filter_value)
        tf.debugging.assert_equal(output, tf.constant(expected, dtype=tf.bool))

    def test_min_filter_mask_keeps_dtype_minimum(self):
        """The dtype minimum is a real value, not a sentinel, so a threshold at the
        bound must keep it."""
        val_tensor = tf.constant([-128, -127], dtype=tf.int8)
        output = min_filter_mask(val_tensor, -128.0)
        tf.debugging.assert_equal(output, tf.constant([True, True], dtype=tf.bool))
