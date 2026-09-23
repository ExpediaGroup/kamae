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
from pyspark.sql.types import ArrayType, IntegerType, LongType, StructField, StructType

from kamae.spark.estimators import EventNgramLookupEstimator
from kamae.spark.transformers import EventNgramLookupTransformer


class TestEventNgramLookupEstimator:
    @pytest.fixture
    def id_df(self, spark_session):
        # Two events x 4 ids per row. The tuple (1, 2, 3, 4) recurs many times so its
        # n-grams clear minNgramFreq, while (9, 9, 9, 9) appears once (stays unknown).
        rows = (
            [([1, 2, 3, 4, 5, 6, 7, 8],)] * 5
            + [([1, 2, 3, 4, 0, 0, 0, 0],)] * 5
            + [([9, 9, 9, 9, 9, 9, 9, 9],)]
        )
        schema = StructType([StructField("clicks_ids", ArrayType(IntegerType()), True)])
        return spark_session.createDataFrame(rows, schema)

    def _estimator(self, **overrides) -> EventNgramLookupEstimator:
        params = dict(
            inputCols=["clicks_ids"],
            outputCols=["clicks_tokens"],
            numEventsPerInput=[2],
            tupleSize=4,
            topK=3,
            vocabSize=100,
            minNgramFreq=1,
        )
        params.update(overrides)
        return EventNgramLookupEstimator(**params)

    def test_fit_returns_transformer_with_propagated_params(self, id_df):
        estimator = self._estimator(layerName="clicks_tokens")
        transformer = estimator.fit(id_df)

        assert isinstance(transformer, EventNgramLookupTransformer)
        assert transformer.getInputCols() == ["clicks_ids"]
        assert transformer.getOutputCols() == ["clicks_tokens"]
        assert transformer.getTupleSize() == 4
        assert transformer.getTopK() == 3
        assert transformer.getNumEventsPerInput() == [2]
        # Vocabulary holds the two reserved tokens plus the learned n-grams.
        assert transformer.getVocabularySize() > 2
        # The frequent tuple was observed and pre-encoded into the lookup table.
        assert (1, 2, 3, 4) in transformer.getTupleToTokens()

    def test_layer_name_defaults_to_uid(self, id_df):
        estimator = self._estimator()
        transformer = estimator.fit(id_df)
        assert transformer.getLayerName() == estimator.uid

    def test_fitted_transformer_tokenizes_frequent_tuple(self, id_df):
        transformer = self._estimator(layerName="clicks_tokens").fit(id_df)
        tokens = (
            transformer.transform(id_df)
            .select("clicks_tokens")
            .collect()[0]["clicks_tokens"]
        )
        assert len(tokens) == 2 * 3
        # The frequent tuple maps to at least one real (non pad/unk) token id (>= 2).
        assert any(token >= 2 for token in tokens)

    def test_validates_input_output_lengths(self, id_df):
        estimator = self._estimator(outputCols=["a", "b"])
        with pytest.raises(ValueError):
            estimator.fit(id_df)

    def test_validates_events_per_input_length(self, id_df):
        estimator = self._estimator(numEventsPerInput=[2, 2])
        with pytest.raises(ValueError):
            estimator.fit(id_df)

    def test_sample_fraction_still_fits(self, id_df):
        estimator = self._estimator(sampleFraction=0.9)
        transformer = estimator.fit(id_df)
        assert isinstance(transformer, EventNgramLookupTransformer)

    def test_multiple_inputs_share_one_vocabulary(self, spark_session):
        rows = [([1, 2, 3, 4], [5, 6, 7, 8])] * 5
        schema = StructType(
            [
                StructField("clicks_ids", ArrayType(IntegerType()), True),
                StructField("prop_ids", ArrayType(IntegerType()), True),
            ]
        )
        df = spark_session.createDataFrame(rows, schema)
        estimator = self._estimator(
            inputCols=["clicks_ids", "prop_ids"],
            outputCols=["clicks_tokens", "prop_tokens"],
            numEventsPerInput=[1, 1],
        )
        transformer = estimator.fit(df)
        lookup = transformer.getTupleToTokens()
        # Tuples from both input columns end up in the single shared lookup table.
        assert (1, 2, 3, 4) in lookup
        assert (5, 6, 7, 8) in lookup

    def test_compatible_dtypes_are_integer_only(self):
        # The Keras layer consumes integer ids only, so the Spark side must not
        # advertise dtypes it cannot achieve parity on. The dtypes are compared
        # against the element type of the input column, so they name the element
        # types rather than the array that holds them.
        estimator = self._estimator()
        assert estimator.compatible_dtypes == [IntegerType(), LongType()]

    def test_use_fit_sample_defaults_to_false(self, id_df):
        # KamaeSparkPipeline calls getUseFitSample() on every estimator declaring the
        # param, so an unset default would raise. False is also the right value here:
        # the vocabulary is thresholded on exact corpus frequencies.
        estimator = self._estimator()
        assert estimator.getUseFitSample() is False

    @pytest.mark.parametrize(
        "param, value",
        [
            ("vocabSize", 2),
            ("minNgramFreq", 0),
            ("tupleSize", 0),
            ("topK", 0),
            ("numEventsPerInput", []),
            ("numEventsPerInput", [1, 0]),
        ],
    )
    def test_invalid_fit_params_raise(self, param, value):
        with pytest.raises(ValueError):
            self._estimator(**{param: value})

    def test_include_token_types_propagates_type_lookup(self, id_df):
        transformer = self._estimator(
            layerName="clicks_tokens", includeTokenTypes=True
        ).fit(id_df)

        assert transformer.getIncludeTokenTypes() is True
        type_lookup = transformer.getTokenTypeLookup()
        assert type_lookup is not None
        # pad (0) and unk (1) are always type 0.
        assert type_lookup[0] == 0
        assert type_lookup[1] == 0
        # Every learned token's type is a non-zero bitmask within [1, 2**tupleSize).
        vocab_size = transformer.getVocabularySize()
        assert len(type_lookup) == vocab_size
        for token_id in range(2, vocab_size):
            assert 1 <= type_lookup[token_id] < 2**4
