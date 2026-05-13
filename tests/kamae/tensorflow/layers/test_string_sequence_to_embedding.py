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

from kamae.tensorflow.layers import StringSequenceToEmbeddingLayer


class TestStringSequenceToEmbedding:
    def test_default_separators_drops_trailing_one_axis(self):
        layer = StringSequenceToEmbeddingLayer(
            name="default_separators",
            seq_len=4,
            embedding_dim=3,
        )
        # Shape (1, 1) with a trailing size-1 axis: expect it to be squeezed.
        inputs = tf.constant([["1|2|3,4|5|6,0|0|0,0|0|0"]])
        expected = tf.constant(
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=tf.float32,
        )
        output = layer(inputs)
        assert output.shape == (1, 4, 3)
        tf.debugging.assert_near(expected, output)

    def test_drops_trailing_one_axis_on_rank_three_input(self):
        layer = StringSequenceToEmbeddingLayer(
            name="drop_trailing_rank_three",
            seq_len=4,
            embedding_dim=3,
        )
        # Shape (None, 1, 1) -> expect (None, 1, 4, 3).
        inputs = tf.constant([[["1|2|3,4|5|6,0|0|0,0|0|0"]]])
        assert inputs.shape == (1, 1, 1)
        output = layer(inputs)
        assert output.shape == (1, 1, 4, 3)

    def test_no_trailing_one_axis_keeps_input_shape(self):
        layer = StringSequenceToEmbeddingLayer(
            name="no_squeeze",
            seq_len=2,
            embedding_dim=2,
        )
        # Last axis size > 1 -> do NOT drop; output is input.shape + (seq_len, d).
        inputs = tf.constant([["1|2,3|4", "5|6,7|8"]])
        assert inputs.shape == (1, 2)
        output = layer(inputs)
        assert output.shape == (1, 2, 2, 2)
        expected = tf.constant(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ],
            dtype=tf.float32,
        )
        tf.debugging.assert_near(expected, output)

    def test_pads_short_sequences_and_truncates_long_ones(self):
        layer = StringSequenceToEmbeddingLayer(
            name="pad_and_truncate",
            seq_len=3,
            embedding_dim=2,
            pad_value="0",
        )
        inputs = tf.constant(
            [
                # Short: only 2 vectors supplied, last vector should be pad.
                ["1|2,3|4"],
                # Long: 4 vectors supplied, last one should be truncated.
                ["1|2,3|4,5|6,7|8"],
            ]
        )
        expected = tf.constant(
            [
                [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ],
            dtype=tf.float32,
        )
        output = layer(inputs)
        assert output.shape == (2, 3, 2)
        tf.debugging.assert_near(expected, output)

    def test_reverse_reverses_only_non_pad_portion(self):
        layer = StringSequenceToEmbeddingLayer(
            name="reverse",
            seq_len=4,
            embedding_dim=2,
            reverse=True,
        )
        inputs = tf.constant([["1|1,2|2,3|3,0|0"]])
        # Non-pad portion "1|1, 2|2, 3|3" reversed -> "3|3, 2|2, 1|1".
        expected = tf.constant(
            [
                [
                    [3.0, 3.0],
                    [2.0, 2.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
            dtype=tf.float32,
        )
        output = layer(inputs)
        tf.debugging.assert_near(expected, output)

    def test_custom_separators(self):
        layer = StringSequenceToEmbeddingLayer(
            name="custom_separators",
            seq_len=2,
            embedding_dim=3,
            separator=":",
            sequence_separator=";",
        )
        inputs = tf.constant([["1:2:3;4:5:6"]])
        expected = tf.constant(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            dtype=tf.float32,
        )
        output = layer(inputs)
        tf.debugging.assert_near(expected, output)

    def test_get_config_round_trip(self):
        layer = StringSequenceToEmbeddingLayer(
            name="round_trip",
            seq_len=5,
            embedding_dim=4,
            separator="|",
            sequence_separator=",",
            pad_value="0",
            reverse=True,
        )
        config = layer.get_config()
        assert config["seq_len"] == 5
        assert config["embedding_dim"] == 4
        assert config["separator"] == "|"
        assert config["sequence_separator"] == ","
        assert config["pad_value"] == "0"
        assert config["reverse"] is True
        recovered = StringSequenceToEmbeddingLayer.from_config(config)
        assert recovered.seq_len == 5
        assert recovered.embedding_dim == 4

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            StringSequenceToEmbeddingLayer(seq_len=0, embedding_dim=3)
        with pytest.raises(ValueError):
            StringSequenceToEmbeddingLayer(seq_len=3, embedding_dim=0)
        with pytest.raises(ValueError):
            StringSequenceToEmbeddingLayer(separator=",", sequence_separator=",")
