import numpy as np
import pytest

from kamae.spark.transformers import PairwiseCosineSimilarityTransformer


class TestPairwiseCosineSimilarityTransformer:
    @pytest.fixture(scope="class")
    def input_df(self, spark_session):
        # query: [1, 0], candidates packed flat: [1, 0, 0, 1] → 2 candidates of dim=2
        return spark_session.createDataFrame(
            [
                ([1.0, 0.0], [1.0, 0.0, 0.0, 1.0]),  # identical + orthogonal → [1.0, 0.0]
                ([0.0, 1.0], [0.0, 1.0, 1.0, 0.0]),  # identical + orthogonal → [1.0, 0.0]
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

    def test_spark_keras_parity(self, input_df):
        import tensorflow as tf

        transformer = PairwiseCosineSimilarityTransformer(
            inputCols=["query", "candidates"],
            outputCol="scores",
            embeddingDim=2,
        )
        spark_values = (
            transformer.transform(input_df)
            .select("scores")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        query = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        candidates = tf.constant([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
        keras_values = transformer.get_tf_layer()([query, candidates]).numpy().tolist()

        np.testing.assert_array_almost_equal(spark_values, keras_values)