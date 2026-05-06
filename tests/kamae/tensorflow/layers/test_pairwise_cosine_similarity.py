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

import numpy as np
import tensorflow as tf

from kamae.tensorflow.layers import PairwiseCosineSimilarityLayer


class TestPairwiseCosineSimilarityLayer:
    def test_identical_vectors_give_similarity_one(self):
        # query [1, 0] vs candidate [1, 0] → cosine = 1.0
        layer = PairwiseCosineSimilarityLayer(embedding_dim=2)
        query = tf.constant([[1.0, 0.0]])
        candidates = tf.constant([[1.0, 0.0]])  # 1 candidate, dim=2

        result = layer([query, candidates]).numpy()

        np.testing.assert_array_almost_equal(result, [[1.0]])

    def test_opposite_vectors_give_similarity_minus_one(self):
        # query [1, 0] vs candidate [-1, 0] → cosine = -1.0
        layer = PairwiseCosineSimilarityLayer(embedding_dim=2)
        query = tf.constant([[1.0, 0.0]])
        candidates = tf.constant([[-1.0, 0.0]])

        result = layer([query, candidates]).numpy()

        np.testing.assert_array_almost_equal(result, [[-1.0]])

    def test_orthogonal_vectors_give_similarity_zero(self):
        # query [1, 0] vs candidate [0, 1] → cosine = 0.0
        layer = PairwiseCosineSimilarityLayer(embedding_dim=2)
        query = tf.constant([[1.0, 0.0]])
        candidates = tf.constant([[0.0, 1.0]])

        result = layer([query, candidates]).numpy()

        np.testing.assert_array_almost_equal(result, [[0.0]])

    def test_multiple_candidates_flat_packed(self):
        # query [1, 0], candidates: [1, 0] and [0, 1] packed as [1, 0, 0, 1]
        # expected: [1.0, 0.0]
        layer = PairwiseCosineSimilarityLayer(embedding_dim=2)
        query = tf.constant([[1.0, 0.0]])
        candidates = tf.constant([[1.0, 0.0, 0.0, 1.0]])

        result = layer([query, candidates]).numpy()

        np.testing.assert_array_almost_equal(result, [[1.0, 0.0]])

    def test_zero_query_vector_gives_zero_similarity(self):
        layer = PairwiseCosineSimilarityLayer(embedding_dim=2)
        query = tf.constant([[0.0, 0.0]])
        candidates = tf.constant([[1.0, 0.0]])

        result = layer([query, candidates]).numpy()

        np.testing.assert_array_almost_equal(result, [[0.0]])

    def test_batch_of_queries(self):
        # Two rows, each with query vs one candidate of dim=2
        layer = PairwiseCosineSimilarityLayer(embedding_dim=2)
        query = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        candidates = tf.constant([[1.0, 0.0], [0.0, 1.0]])  # same vectors

        result = layer([query, candidates]).numpy()

        np.testing.assert_array_almost_equal(result, [[1.0], [1.0]])
