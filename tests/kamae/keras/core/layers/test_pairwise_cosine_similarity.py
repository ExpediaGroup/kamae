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

from kamae.keras.core.layers import PairwiseCosineSimilarityLayer


class TestPairwiseCosineSimilarity:
    @pytest.mark.parametrize(
        "query, flat_candidates, embedding_dim, expected_output",
        [
            (
                tf.constant([[1.0, 0.0, 0.0]]),
                tf.constant([[1.0, 0.0, 0.0]]),
                3,
                tf.constant([[1.0]]),
            ),
            (
                tf.constant([[1.0, 0.0, 0.0]]),
                tf.constant([[-1.0, 0.0, 0.0]]),
                3,
                tf.constant([[-1.0]]),
            ),
            (
                tf.constant([[1.0, 0.0]]),
                tf.constant([[0.0, 1.0]]),
                2,
                tf.constant([[0.0]]),
            ),
            (
                tf.constant([[1.0, 0.0, 0.0]]),
                tf.constant([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
                3,
                tf.constant([[1.0, 0.0, 0.0]]),
            ),
            (
                tf.constant([[0.0, 0.0, 0.0]]),
                tf.constant([[1.0, 0.0, 0.0]]),
                3,
                tf.constant([[0.0]]),
            ),
        ],
    )
    def test_pairwise_cosine_similarity(
        self, query, flat_candidates, embedding_dim, expected_output
    ):
        layer = PairwiseCosineSimilarityLayer(
            name="pairwise_cos", embedding_dim=embedding_dim
        )
        output_tensor = layer([query, flat_candidates])

        assert output_tensor.shape == expected_output.shape
        tf.debugging.assert_near(output_tensor, expected_output, atol=1e-6)

    def test_batch_processing(self):
        query = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        flat_candidates = tf.constant([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
        layer = PairwiseCosineSimilarityLayer(name="batch_test", embedding_dim=2)
        output_tensor = layer([query, flat_candidates])
        expected = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        tf.debugging.assert_near(output_tensor, expected, atol=1e-6)

    def test_wrong_number_of_inputs(self):
        layer = PairwiseCosineSimilarityLayer(name="error_test", embedding_dim=3)
        with pytest.raises(ValueError):
            layer([tf.constant([[1.0, 0.0, 0.0]])])

    def test_get_config(self):
        layer = PairwiseCosineSimilarityLayer(name="config_test", embedding_dim=64)
        config = layer.get_config()
        assert config["embedding_dim"] == 64
        assert config["name"] == "config_test"
