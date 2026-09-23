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

import logging
from collections import Counter

import pytest
from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType

from kamae.spark.utils.ngram_utils import (
    EventNgramVocabulary,
    build_tuple_lookup_table,
    build_vocabulary,
    collect_ngrams_from_dataframe,
    log_vocabulary_by_length,
    tokenize_events,
)
from kamae.spark.utils.ngram_worker_functions import (
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    encode_tuple,
    extract_ngrams_from_column_worker,
    extract_tuples_from_column_worker,
)


class TestCollectNgramsFromDataFrame:
    @pytest.fixture(scope="class")
    def id_df(self, spark_session):
        schema = StructType([StructField("clicks_ids", ArrayType(IntegerType()), True)])
        return spark_session.createDataFrame([([1, 2, 3, 4],)], schema)

    def test_raises_when_an_input_column_is_missing(self, id_df):
        # Silently training on whichever columns happen to exist would produce a
        # vocabulary that does not cover every column being tokenized.
        with pytest.raises(ValueError, match="not found on the DataFrame"):
            collect_ngrams_from_dataframe(id_df, ["clicks_ids", "absent_col"], 4)

    def test_counts_across_all_given_columns(self, id_df):
        counter = collect_ngrams_from_dataframe(id_df, ["clicks_ids"], 4)
        # One event of 4 non-zero levels -> 2**4 - 1 = 15 combinations, each seen once.
        assert len(counter) == 15
        assert set(counter.values()) == {1}

    def test_drops_ngrams_below_min_frequency_before_collecting(self, spark_session):
        # The two events share L0_1, L1_2 and L2_3, so only the 2**3 - 1 = 7
        # combinations of those levels are seen twice.
        schema = StructType([StructField("clicks_ids", ArrayType(IntegerType()), True)])
        df = spark_session.createDataFrame([([1, 2, 3, 4],), ([1, 2, 3, 5],)], schema)

        counter = collect_ngrams_from_dataframe(df, ["clicks_ids"], 4, min_ngram_freq=2)

        assert len(counter) == 7
        assert set(counter.values()) == {2}
        assert all("L3_4" not in ngram and "L3_5" not in ngram for ngram in counter)

    def test_raises_when_an_input_column_is_a_nested_array(self, spark_session):
        # A nested array would be chunked over its sub-arrays rather than over its ids.
        schema = StructType(
            [StructField("nested_ids", ArrayType(ArrayType(IntegerType())), True)]
        )
        nested_df = spark_session.createDataFrame([([[1, 2, 3, 4]],)], schema)
        with pytest.raises(ValueError, match="array nesting level"):
            collect_ngrams_from_dataframe(nested_df, ["nested_ids"], 4)


class TestExtractNgramsFromColumnWorker:
    @pytest.mark.parametrize(
        "col_value, event_size, expected",
        [
            # Every combination of the present levels, so level-skipping n-grams
            # such as (L0, L2) are included.
            (
                [1, 2, 0],
                3,
                [("L0_1",), ("L1_2",), ("L0_1", "L1_2")],
            ),
            # Zero ids are dropped before combining, so L1 never appears.
            (
                [1, 0, 3],
                3,
                [("L0_1",), ("L2_3",), ("L0_1", "L2_3")],
            ),
            # An all-zero event contributes nothing.
            ([0, 0, 0], 3, []),
            (None, 3, []),
            ([], 3, []),
            # A null id is an absent ID level, exactly like a 0, rather than becoming
            # an "L1_None" token that could never match the vocabulary.
            (
                [1, None, 3],
                3,
                [("L0_1",), ("L2_3",), ("L0_1", "L2_3")],
            ),
            ([None, None, None], 3, []),
        ],
    )
    def test_extracts_within_event_combinations(self, col_value, event_size, expected):
        actual = extract_ngrams_from_column_worker(col_value, event_size)
        assert sorted(actual) == sorted(expected)

    def test_never_crosses_event_boundaries(self):
        # Two single-level events: no n-gram may contain ids from both.
        ngrams = extract_ngrams_from_column_worker([1, 0, 0, 5, 0, 0], 3)
        assert all(
            not ({"L0_1"} <= set(ngram) and {"L0_5"} <= set(ngram)) for ngram in ngrams
        )

    def test_raises_when_the_ids_are_not_a_whole_number_of_events(self):
        with pytest.raises(ValueError, match="whole number of events"):
            extract_ngrams_from_column_worker([1, 2, 3, 4, 5, 6], 4)


class TestExtractTuplesFromColumnWorker:
    @pytest.mark.parametrize(
        "col_value, expected",
        [
            ([1, 2, 3, 4, 5, 6, 7, 8], [(1, 2, 3, 4), (5, 6, 7, 8)]),
            # All-zero events are padding and are not vocabulary entries.
            ([1, 2, 3, 4, 0, 0, 0, 0], [(1, 2, 3, 4)]),
            # A partially-zero event is still a real event.
            ([1, 0, 0, 0], [(1, 0, 0, 0)]),
            ([0, 0, 0, 0], []),
            (None, []),
        ],
    )
    def test_extracts_non_zero_tuples(self, col_value, expected):
        assert extract_tuples_from_column_worker(col_value, 4) == expected


class TestBuildVocabulary:
    def test_assigns_ids_by_descending_frequency_from_two(self):
        counter = Counter({("L0_1",): 100, ("L1_2",): 50, ("L0_1", "L1_2"): 10})
        ngram_to_id = build_vocabulary(counter, vocab_size=100, min_ngram_freq=1)
        # Ids start at 2 (0 and 1 are reserved) and ascend as frequency descends, so
        # a smaller id always means a more frequent n-gram.
        assert ngram_to_id == {
            ("L0_1",): 2,
            ("L1_2",): 3,
            ("L0_1", "L1_2"): 4,
        }

    def test_drops_ngrams_below_min_frequency(self):
        counter = Counter({("L0_1",): 100, ("L1_2",): 5})
        ngram_to_id = build_vocabulary(counter, vocab_size=100, min_ngram_freq=10)
        assert ngram_to_id == {("L0_1",): 2}

    def test_vocab_size_reserves_two_special_tokens(self):
        counter = Counter({("L0_1",): 100, ("L1_2",): 90, ("L2_3",): 80})
        ngram_to_id = build_vocabulary(counter, vocab_size=4, min_ngram_freq=1)
        # vocab_size=4 leaves room for 2 learned n-grams alongside pad and unk.
        assert len(ngram_to_id) == 2

    def test_ties_broken_deterministically_by_ngram(self):
        counter = Counter({("L1_2",): 10, ("L0_1",): 10})
        first = build_vocabulary(counter, vocab_size=100, min_ngram_freq=1)
        second = build_vocabulary(
            Counter({("L0_1",): 10, ("L1_2",): 10}), vocab_size=100, min_ngram_freq=1
        )
        assert first == second


class TestEventNgramVocabularyBuildTypeLookup:
    def _vocabulary(self, ngrams):
        return EventNgramVocabulary(ngrams=ngrams)

    @pytest.mark.parametrize(
        "ngram, expected_bitmask",
        [
            (("L0_1",), 0b0001),
            (("L1_2",), 0b0010),
            (("L3_4",), 0b1000),
            # A level-skipping n-gram sets exactly the bits of the levels it spans.
            (("L0_1", "L2_3"), 0b0101),
            (("L0_1", "L1_2", "L2_3", "L3_4"), 0b1111),
        ],
    )
    def test_bitmask_names_the_spanned_levels(self, ngram, expected_bitmask):
        vocabulary = self._vocabulary({ngram: 2})
        assert vocabulary.build_type_lookup()[2] == expected_bitmask

    def test_special_tokens_have_type_zero(self):
        lookup = self._vocabulary({("L0_1",): 2}).build_type_lookup()
        assert lookup[PAD_TOKEN_ID] == 0
        assert lookup[UNK_TOKEN_ID] == 0

    def test_supports_more_than_four_levels(self):
        # The bitmask is parsed from the level integer, so it is not capped at L3.
        vocabulary = self._vocabulary({("L4_9", "L6_1"): 2})
        assert vocabulary.build_type_lookup()[2] == (1 << 4) | (1 << 6)


class TestEncodeTuple:
    # ("L0_1",) is the most frequent, then the pair, then ("L1_2",).
    NGRAMS = {("L0_1",): 2, ("L0_1", "L1_2"): 3, ("L1_2",): 4}

    def test_returns_matching_ngrams_most_frequent_first(self):
        assert encode_tuple((1, 2), self.NGRAMS, top_k=3) == [2, 3, 4]

    def test_truncates_to_top_k(self):
        assert encode_tuple((1, 2), self.NGRAMS, top_k=2) == [2, 3]

    def test_pads_when_fewer_matches_than_top_k(self):
        assert encode_tuple((1, 0), self.NGRAMS, top_k=3) == [
            2,
            PAD_TOKEN_ID,
            PAD_TOKEN_ID,
        ]

    def test_all_zero_tuple_has_no_encoding(self):
        assert encode_tuple((0, 0), self.NGRAMS, top_k=3) is None

    def test_unmatched_tuple_has_no_encoding_so_it_is_left_out_of_the_table(self):
        # A tuple with no n-gram in the vocabulary carries no token, so it is omitted
        # from the lookup table and resolves to <unk> at inference by the same rule as
        # a tuple that was never seen while fitting.
        assert encode_tuple((7, 8), self.NGRAMS, top_k=4) is None
        assert tokenize_events([7, 8], {}, num_events=1, tuple_size=2, top_k=4) == [
            UNK_TOKEN_ID,
            PAD_TOKEN_ID,
            PAD_TOKEN_ID,
            PAD_TOKEN_ID,
        ]


class TestTokenizeEvents:
    LOOKUP = {(1, 2, 3, 4): [2, 3], (5, 6, 7, 8): [4, 5]}

    @pytest.mark.parametrize(
        "ids, expected",
        [
            ([1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5]),
            # A non-zero tuple missing from the table is unknown, not padding, and is
            # marked by a single <unk> rather than filling every token slot.
            ([1, 2, 3, 4, 9, 9, 9, 9], [2, 3, UNK_TOKEN_ID, PAD_TOKEN_ID]),
            # An all-zero event is padding.
            ([1, 2, 3, 4, 0, 0, 0, 0], [2, 3, 0, 0]),
            # Short rows are right-padded to num_events * top_k.
            ([1, 2, 3, 4], [2, 3, 0, 0]),
            # Long rows are truncated.
            ([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4], [2, 3, 4, 5]),
            (None, [0, 0, 0, 0]),
            ([], [0, 0, 0, 0]),
            # Null ids read as 0, so an all-null event is padding and a partially-null
            # event is looked up as if the null levels were absent.
            ([None, None, None, None, 1, 2, 3, 4], [0, 0, 2, 3]),
            ([1, 2, 3, None, 0, 0, 0, 0], [UNK_TOKEN_ID, PAD_TOKEN_ID, 0, 0]),
        ],
    )
    def test_tokenizes_to_fixed_length(self, ids, expected):
        actual = tokenize_events(ids, self.LOOKUP, num_events=2, tuple_size=4, top_k=2)
        assert actual == expected
        assert len(actual) == 4

    def test_raises_when_the_ids_are_not_a_whole_number_of_events(self):
        with pytest.raises(ValueError, match="whole number of events"):
            tokenize_events(
                [1, 2, 3, 4, 5, 6], self.LOOKUP, num_events=2, tuple_size=4, top_k=2
            )

    def test_empty_lookup_table_maps_every_event_to_unk(self):
        assert tokenize_events(
            [1, 2, 3, 4], {}, num_events=1, tuple_size=4, top_k=2
        ) == [UNK_TOKEN_ID, PAD_TOKEN_ID]


class TestLogVocabularyByLength:
    def test_reports_the_share_of_each_n_gram_length(self, caplog):
        # A vocabulary dominated by 1-grams means the n-gram combinations are
        # earning little, which is what this reports.
        vocabulary = {
            ("L0_1",): 2,
            ("L0_2",): 3,
            ("L0_1", "L1_5"): 4,
            ("L0_1", "L1_5", "L2_9"): 5,
            ("L0_1", "L1_5", "L2_9", "L3_2"): 6,
        }
        with caplog.at_level(logging.INFO, logger="kamae.spark.utils.ngram_utils"):
            log_vocabulary_by_length(vocabulary)

        message = caplog.text
        assert "1-gram 2 (40.0%)" in message
        assert "2-gram 1 (20.0%)" in message
        assert "3-gram 1 (20.0%)" in message
        assert "4-gram 1 (20.0%)" in message

    def test_reports_an_empty_vocabulary_without_dividing_by_zero(self, caplog):
        with caplog.at_level(logging.INFO, logger="kamae.spark.utils.ngram_utils"):
            log_vocabulary_by_length({})
        assert "empty vocabulary" in caplog.text

    def test_build_vocabulary_reports_the_distribution(self, caplog):
        counter = Counter(
            {("L0_1",): 10, ("L0_1", "L1_2"): 8, ("L0_1", "L1_2", "L2_3"): 6}
        )
        with caplog.at_level(logging.INFO, logger="kamae.spark.utils.ngram_utils"):
            build_vocabulary(ngram_counter=counter, vocab_size=100, min_ngram_freq=1)
        assert "Vocabulary by n-gram length" in caplog.text
