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
import pytest
import tensorflow as tf

from kamae.spark.transformers import PairwiseCosineSimilarityTransformer


class TestPairwiseCosineSimilarityTransformer:
    @pytest.fixture(scope="class")
    def input_df(self, spark_session):
        # query: [1, 0], candidates packed flat: [1, 0, 0, 1] → 2 candidates of dim=2
        return spark_session.createDataFrame(
            [
                (
                    [1.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0],
                ),  # identical + orthogonal → [1.0, 0.0]
                (
                    [0.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                ),  # identical + orthogonal → [1.0, 0.0]
            ],
            ["query", "candidates"],
        )

    def test_returns_cosine_similarity_per_candidate(self, input_df):
        transformer = PairwiseCosineSimilarityTransformer(
            inputCols=["query", "candidates"],
            outputCol="scores",
            embeddingDim=2,
        )
        result = transformer.transform(input_df).select("scores").collect()

        np.testing.assert_array_almost_equal(result[0].scores, [1.0, 0.0])
        np.testing.assert_array_almost_equal(result[1].scores, [1.0, 0.0])

    def test_opposite_vectors_give_minus_one(self, spark_session):
        df = spark_session.createDataFrame(
            [([1.0, 0.0], [-1.0, 0.0])],
            ["query", "candidates"],
        )
        transformer = PairwiseCosineSimilarityTransformer(
            inputCols=["query", "candidates"],
            outputCol="scores",
            embeddingDim=2,
        )
        result = transformer.transform(df).select("scores").collect()

        np.testing.assert_array_almost_equal(result[0].scores, [-1.0])

    def test_wrong_number_of_input_cols_raises(self):
        with pytest.raises(ValueError):
            PairwiseCosineSimilarityTransformer(
                inputCols=["a"],
                outputCol="scores",
                embeddingDim=2,
            )

    @pytest.mark.parametrize(
        "queries, flat_candidates, embedding_dim, input_dtype, output_dtype",
        [
            # default dtypes, dim=2, 2 candidates
            (
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
                2,
                None,
                None,
            ),
            # float input, double output
            (
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
                2,
                "float",
                "double",
            ),
            # double input, float output
            (
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
                2,
                "double",
                "float",
            ),
            # dim=3, 3 candidates
            (
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                3,
                None,
                None,
            ),
            # opposite vectors → similarity -1.0
            (
                [[1.0, 0.0]],
                [[-1.0, 0.0]],
                2,
                None,
                None,
            ),
            # zero-vector query → both sides must return 0.0
            (
                [[0.0, 0.0]],
                [[1.0, 0.0, 0.0, 1.0]],
                2,
                None,
                None,
            ),
        ],
    )
    def test_spark_tf_parity(
        self,
        spark_session,
        queries,
        flat_candidates,
        embedding_dim,
        input_dtype,
        output_dtype,
    ):
        transformer = PairwiseCosineSimilarityTransformer(
            inputCols=["query", "candidates"],
            outputCol="scores",
            embeddingDim=embedding_dim,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )

        spark_df = spark_session.createDataFrame(
            list(zip(queries, flat_candidates)),
            ["query", "candidates"],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("scores")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        tf_queries = tf.constant(queries, dtype=tf.float32)
        tf_candidates = tf.constant(flat_candidates, dtype=tf.float32)
        keras_values = (
            transformer.get_tf_layer()([tf_queries, tf_candidates]).numpy().tolist()
        )

        np.testing.assert_almost_equal(
            spark_values,
            keras_values,
            decimal=4,
            err_msg="Spark and TensorFlow outputs are not equal",
        )
