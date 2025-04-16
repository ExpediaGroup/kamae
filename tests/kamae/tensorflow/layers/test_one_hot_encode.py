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

from kamae.tensorflow.layers import OneHotLayer


class TestOneHotEncode:
    @pytest.mark.parametrize(
        "input_tensor, input_name, vocabulary, num_oov_indices, drop_unseen, mask_token, expected_output",
        [
            (
                tf.constant(["1", "2", "3"]),
                "input_1",
                ["1", "2", "3"],
                1,
                False,
                None,
                tf.constant(
                    [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32
                ),
            ),
            (
                tf.constant(["1", "2", "3"]),
                "input_1",
                ["1", "2", "3"],
                1,
                True,
                None,
                tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32),
            ),
            (
                tf.constant([[["1", "2"], ["3", "4"]]]),
                "input_2",
                ["1", "2", "3"],
                1,
                False,
                None,
                tf.constant(
                    [
                        [
                            [[0, 1, 0, 0], [0, 0, 1, 0]],
                            [[0, 0, 0, 1], [1, 0, 0, 0]],
                        ]
                    ],
                    dtype=tf.float32,
                ),
            ),
            (
                tf.constant([[["1", "2"], ["3", "4"]]]),
                "input_2",
                ["1", "2", "3"],
                1,
                True,
                None,
                tf.constant(
                    [
                        [
                            [[1, 0, 0], [0, 1, 0]],
                            [[0, 0, 1], [0, 0, 0]],
                        ]
                    ],
                    dtype=tf.float32,
                ),
            ),
            (
                tf.constant([[[1], [2], [3]]]),
                "input_3",
                ["1", "2", "4"],
                1,
                False,
                None,
                tf.constant(
                    [[[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]], dtype=tf.float32
                ),
            ),
            (
                tf.constant(["1", "2", "3"]),
                "input_1",
                ["1", "2", "3"],
                1,
                False,
                "4",
                tf.constant(
                    [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                    dtype=tf.float32,
                ),
            ),
            (
                tf.constant(["1", "2", "3"]),
                "input_1",
                ["1", "2", "3"],
                1,
                True,
                "4",
                tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32),
            ),
            (
                tf.constant([[["1", "2"], ["3", "4"]]]),
                "input_2",
                ["1", "2", "3"],
                1,
                False,
                "4",
                tf.constant(
                    [
                        [
                            [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                            [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0]],
                        ]
                    ],
                    dtype=tf.float32,
                ),
            ),
            (
                tf.constant([[["1", "2"], ["3", "4"]]]),
                "input_2",
                ["1", "2", "3"],
                1,
                True,
                "5",
                tf.constant(
                    [
                        [
                            [[1, 0, 0], [0, 1, 0]],
                            [[0, 0, 1], [0, 0, 0]],
                        ]
                    ],
                    dtype=tf.float32,
                ),
            ),
            (
                tf.constant([[[1], [2], [3]]]),
                "input_3",
                ["1", "2", "4"],
                1,
                False,
                "5",
                tf.constant(
                    [[[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]],
                    dtype=tf.float32,
                ),
            ),
        ],
    )
    def test_onehot(
        self,
        input_tensor,
        input_name,
        vocabulary,
        num_oov_indices,
        drop_unseen,
        mask_token,
        expected_output,
    ):
        # when
        layer = OneHotLayer(
            name=input_name,
            vocabulary=vocabulary,
            num_oov_indices=num_oov_indices,
            drop_unseen=drop_unseen,
            mask_token=mask_token,
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
