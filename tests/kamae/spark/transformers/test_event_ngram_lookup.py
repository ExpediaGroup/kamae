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
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from kamae.keras.tensorflow.layers import EventNgramLookupLayer
from kamae.spark.transformers import EventNgramLookupTransformer

# Fitted lookup table shared across the tests: two known event tuples, top_k = 3.
LOOKUP_TABLE = {(1, 2, 3, 4): [2, 3, 0], (5, 6, 7, 8): [4, 5, 0]}
# The transformer persists the table as two flat, JSON-serialisable int lists.
LOOKUP_KEYS = [i for key in LOOKUP_TABLE for i in key]
LOOKUP_VALUES = [v for value in LOOKUP_TABLE.values() for v in value]
TOP_K = 3
TUPLE_SIZE = 4
VOCAB_SIZE = 6


class TestEventNgramLookupTransformer:
    def _transformer(self, **overrides) -> EventNgramLookupTransformer:
        params = dict(
            inputCols=["clicks_ids"],
            outputCols=["clicks_tokens"],
            numEventsPerInput=[2],
            tupleSize=TUPLE_SIZE,
            topK=TOP_K,
            vocabularySize=VOCAB_SIZE,
            lookupKeys=LOOKUP_KEYS,
            lookupValues=LOOKUP_VALUES,
            layerName="clicks_tokens",
        )
        params.update(overrides)
        return EventNgramLookupTransformer(**params)

    @pytest.fixture
    def search_level_df(self, spark_session):
        # Each row is a flat int array of 2 events x 4 ids: a known tuple, an unknown
        # tuple and an all-zero (padding) tuple.
        data = [
            ([1, 2, 3, 4, 5, 6, 7, 8],),  # known, known
            ([1, 2, 3, 4, 9, 9, 9, 9],),  # known, unknown
            ([0, 0, 0, 0, 0, 0, 0, 0],),  # padding, padding
        ]
        schema = StructType([StructField("clicks_ids", ArrayType(IntegerType()), True)])
        return spark_session.createDataFrame(data, schema)

    def test_transform_known_unknown_padding(self, search_level_df):
        transformer = self._transformer()
        actual = [
            row["clicks_tokens"]
            for row in transformer.transform(search_level_df)
            .select("clicks_tokens")
            .collect()
        ]
        expected = [
            [2, 3, 0, 4, 5, 0],  # both tuples known
            [2, 3, 0, 1, 1, 1],  # known then <unk> (id 1)
            [0, 0, 0, 0, 0, 0],  # both padding (id 0)
        ]
        assert actual == expected

    def test_spark_tf_parity_search_level(self, search_level_df):
        transformer = self._transformer()
        spark_out = [
            row["clicks_tokens"]
            for row in transformer.transform(search_level_df)
            .select("clicks_tokens")
            .collect()
        ]
        layer = transformer.get_keras_layer()
        tf_in = tf.constant(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 9, 9, 9, 9],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=tf.int32,
        )
        tf_out = layer(tf_in).numpy().tolist()
        assert spark_out == tf_out

    @pytest.mark.parametrize(
        "ids, expected",
        [
            # A short row is right-padded up to numEvents * topK.
            ([1, 2, 3, 4], [2, 3, 0, 0, 0, 0]),
            # A trailing partial event cannot be a tuple, so it is dropped.
            ([1, 2, 3, 4, 5, 6], [2, 3, 0, 0, 0, 0]),
            # A row longer than numEvents is truncated.
            ([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4], [2, 3, 0, 4, 5, 0]),
            # Null and empty rows are all padding.
            (None, [0, 0, 0, 0, 0, 0]),
            ([], [0, 0, 0, 0, 0, 0]),
        ],
    )
    def test_transform_pads_and_truncates_to_fixed_length(
        self, spark_session, ids, expected
    ):
        # given
        schema = StructType([StructField("clicks_ids", ArrayType(IntegerType()), True)])
        df = spark_session.createDataFrame([(ids,)], schema)

        # when
        actual = (
            self._transformer()
            .transform(df)
            .select("clicks_tokens")
            .collect()[0]["clicks_tokens"]
        )

        # then
        assert actual == expected

    def test_transform_rejects_string_input_column(self, spark_session):
        # The Keras layer can only consume integer ids, so the Spark side declares the
        # same dtypes rather than silently accepting a column it cannot serve.
        schema = StructType([StructField("clicks_ids", ArrayType(StringType()), True)])
        df = spark_session.createDataFrame([(["1,2,3,4"],)], schema)
        with pytest.raises(Exception):
            self._transformer().transform(df).collect()

    def test_spark_tf_parity_with_list_dimension(self, search_level_df):
        # Rank-3 inputs (batch, list_size, ids) arise in listwise models; the list
        # dimension must be preserved and each item tokenized independently.
        transformer = self._transformer()
        spark_out = [
            row["clicks_tokens"]
            for row in transformer.transform(search_level_df)
            .select("clicks_tokens")
            .collect()
        ]
        layer = transformer.get_keras_layer()
        # One batch of 3 list items, mirroring the 3 Spark rows.
        tf_in = tf.constant(
            [
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [1, 2, 3, 4, 9, 9, 9, 9],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            dtype=tf.int32,
        )
        tf_out = layer(tf_in)
        assert tf_out.shape == (1, 3, 2 * TOP_K)
        assert tf_out.numpy()[0].tolist() == spark_out

    def test_get_keras_layer_returns_single_consolidated_layer(self):
        layer = self._transformer().get_keras_layer()
        assert isinstance(layer, EventNgramLookupLayer)
        assert layer.name == "clicks_tokens_tokenizer"
        assert layer.num_events_per_input == [2]
        assert layer.top_k == TOP_K
        assert layer.tuple_size == TUPLE_SIZE

    def test_multiple_inputs_one_layer_all_events(self):
        transformer = EventNgramLookupTransformer(
            inputCols=["clicks_ids", "prop_ids"],
            outputCols=["clicks_tokens", "prop_tokens"],
            numEventsPerInput=[2, 1],
            tupleSize=TUPLE_SIZE,
            topK=TOP_K,
            vocabularySize=VOCAB_SIZE,
            lookupKeys=LOOKUP_KEYS,
            lookupValues=LOOKUP_VALUES,
            layerName="tokenizer",
        )
        # A single layer tokenizes all inputs (the table is embedded once).
        layer = transformer.get_keras_layer()
        assert isinstance(layer, EventNgramLookupLayer)
        assert layer.num_events_per_input == [2, 1]

    def test_construct_layer_info(self):
        info = self._transformer().construct_layer_info()
        assert info["name"] == "clicks_tokens"
        assert info["inputs"] == ["clicks_ids"]
        assert info["outputs"] == ["clicks_tokens"]

    def test_transform_defaults(self):
        transformer = EventNgramLookupTransformer(
            inputCols=["clicks_ids"],
            outputCols=["clicks_tokens"],
            numEventsPerInput=[2],
            topK=TOP_K,
            lookupKeys=LOOKUP_KEYS,
            lookupValues=LOOKUP_VALUES,
        )
        assert transformer.getLayerName() == transformer.uid
        assert transformer.getTupleSize() == 4
        assert transformer.getIncludeTokenTypes() is False
        assert transformer.getTokenTypeLookup() is None

    def test_getters(self):
        transformer = self._transformer()
        assert transformer.getTupleSize() == TUPLE_SIZE
        assert transformer.getTopK() == TOP_K
        assert transformer.getNumEventsPerInput() == [2]
        assert transformer.getVocabularySize() == VOCAB_SIZE
        assert transformer.getTupleToTokens() == LOOKUP_TABLE
        assert transformer.getIncludeTokenTypes() is False

    def test_params_survive_save_and_load(self, spark_session, tmp_path):
        # Spark writes params to JSON metadata, so the fitted table has to stay in a
        # JSON-serialisable shape for the transformer, and any pipeline holding it, to
        # be saveable at all.
        transformer = self._transformer(
            includeTokenTypes=True, tokenTypeLookup=[0, 0, 5, 1, 8, 3]
        )
        path = str(tmp_path / "transformer")
        transformer.write().overwrite().save(path)

        reloaded = EventNgramLookupTransformer.load(path)
        assert reloaded.getLookupKeys() == LOOKUP_KEYS
        assert reloaded.getLookupValues() == LOOKUP_VALUES
        assert reloaded.getTupleToTokens() == LOOKUP_TABLE
        assert reloaded.getTokenTypeLookup() == [0, 0, 5, 1, 8, 3]

    def test_reloaded_transformer_tokenizes_identically(
        self, spark_session, search_level_df, tmp_path
    ):
        transformer = self._transformer()
        expected = transformer.transform(search_level_df).collect()

        path = str(tmp_path / "transformer")
        transformer.write().overwrite().save(path)
        reloaded = EventNgramLookupTransformer.load(path)

        assert reloaded.transform(search_level_df).collect() == expected

    def test_transform_emits_type_columns(self, search_level_df):
        # token id -> ID-level bitmask; tokens 2/3 are types 5/1, unk(1)->0, pad(0)->0.
        type_lookup = [0, 0, 5, 1, 8, 3]
        transformer = self._transformer(
            includeTokenTypes=True, tokenTypeLookup=type_lookup
        )
        rows = (
            transformer.transform(search_level_df)
            .select("clicks_tokens", "clicks_tokens_types")
            .collect()
        )
        for row in rows:
            # Each type is the bitmask gathered at the corresponding token id.
            assert row["clicks_tokens_types"] == [
                type_lookup[t] for t in row["clicks_tokens"]
            ]

    def test_construct_layer_info_appends_type_outputs(self):
        # The consolidated layer returns all token tensors then all type tensors, so
        # the declared output columns must follow the same grouping.
        info = self._transformer(
            includeTokenTypes=True, tokenTypeLookup=[0, 0, 5, 1, 8, 3]
        ).construct_layer_info()
        assert info["inputs"] == ["clicks_ids"]
        assert info["outputs"] == ["clicks_tokens", "clicks_tokens_types"]

    def test_spark_tf_parity_with_types(self, search_level_df):
        type_lookup = [0, 0, 5, 1, 8, 3]
        transformer = self._transformer(
            includeTokenTypes=True, tokenTypeLookup=type_lookup
        )
        spark_tokens, spark_types = [], []
        for row in (
            transformer.transform(search_level_df)
            .select("clicks_tokens", "clicks_tokens_types")
            .collect()
        ):
            spark_tokens.append(row["clicks_tokens"])
            spark_types.append(row["clicks_tokens_types"])

        layer = transformer.get_keras_layer()
        tf_in = tf.constant(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 9, 9, 9, 9],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=tf.int32,
        )
        tf_tokens, tf_types = layer(tf_in)
        assert tf_tokens.numpy().tolist() == spark_tokens
        assert tf_types.numpy().tolist() == spark_types
