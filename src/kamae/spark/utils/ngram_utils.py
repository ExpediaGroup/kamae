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

"""
N-gram vocabulary utilities for event-level discrete-ID tokenization.

Each event is a fixed-size tuple of IDs (e.g. 4 levels ``L0, L1, L2, L3``)
and n-grams are learned only within event boundaries, never across events. The public
helpers are ``collect_ngrams_from_dataframe`` (distributed n-gram counting),
``build_vocabulary`` (frequency selection) and ``build_tuple_lookup_table`` (pre-compute
the tuple -> top-k tokens table for O(1) inference).
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame

from kamae.spark.utils.ngram_worker_functions import (
    NUM_RESERVED_TOKENS,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    encode_tuple,
    extract_ngrams_from_column_worker,
    extract_tuples_from_column_worker,
)

logger = logging.getLogger(__name__)

DEFAULT_EVENT_SIZE = 4  # Default discrete ID values per event (e.g. L0, L1, L2, L3)


class EventNgramVocabulary:
    """
    Vocabulary of n-grams learned from within-event patterns only.

    Each event is a tuple of IDs and n-grams are learned only within event
    boundaries. Token id 0 is reserved for ``<pad>`` and 1 for ``<unk>``; learned
    n-grams are assigned ids from 2 upwards, contiguously and in descending frequency
    order, so a smaller token id always means a more frequent n-gram.
    """

    def __init__(self, ngrams: Dict[Tuple[str, ...], int]) -> None:
        """
        Initializes the n-gram vocabulary.

        :param ngrams: Mapping from n-gram tuple to token id, as returned by
        ``build_vocabulary``.
        """
        self.ngrams = ngrams

    @property
    def vocab_size(self) -> int:
        """
        Returns the total number of token ids, including the two reserved tokens.

        Token ids are contiguous from ``NUM_RESERVED_TOKENS``, so this is also one past
        the largest token id.

        :returns: Number of token ids in the vocabulary.
        """
        return len(self.ngrams) + NUM_RESERVED_TOKENS

    def build_type_lookup(self) -> List[int]:
        """
        Builds a per-token "type" lookup: token id -> bitmask of contributing ID levels.

        Each n-gram is a tuple of level-prefixed ids (``L<level>_<id>``). The type
        encodes *which* ID levels the n-gram spans as a bitmask with bit ``level`` set
        for each present level (e.g. an ``(L0, L2)`` skip-gram -> ``0b0101 = 5``). It is
        compact categorical feature the model can embed alongside the token itself. The
        reserved ``<pad>`` and ``<unk>`` tokens map to ``0``.

        The bitmask is parametric in the number of ID levels: the level integer is
        parsed from each n-gram part, so it works for any ``tupleSize`` (and
        multi-digit levels).
        The type cardinality is ``2 ** tupleSize``.

        :returns: List indexed by token id giving each token's level bitmask.
        """
        lookup = [0] * self.vocab_size
        for ngram, token_id in self.ngrams.items():
            bitmask = 0
            for part in ngram:
                # part is "L<level>_<id>"; take the integer level after the "L".
                bitmask |= 1 << int(part[1:].split("_")[0])
            lookup[token_id] = bitmask
        return lookup


def collect_ngrams_from_dataframe(
    df: DataFrame,
    input_columns: List[str],
    event_size: int = DEFAULT_EVENT_SIZE,
) -> Counter:
    """
    Extracts and counts within-event n-grams across the given columns, distributed.

    All columns are counted in a single pass: every n-gram of every column of a row is
    emitted by one ``flatMap``, so the dataset is read once no matter how many columns
    are tokenized. Counts are aggregated with ``reduceByKey`` (map-side combine), so
    only the distinct n-grams reach the driver.

    :param df: Input DataFrame with the discrete-ID columns.
    :param input_columns: Column names holding discrete ID values.
    :param event_size: Number of discrete ID values per event.
    :returns: Counter mapping each n-gram tuple to its corpus frequency.
    :raises ValueError: If any of the input columns is missing from the DataFrame.
    """
    missing_cols = [c for c in input_columns if c not in df.columns]
    if missing_cols:
        # Training the vocabulary on a subset of the requested columns would silently
        # produce a vocabulary that does not cover every column being tokenized.
        raise ValueError(
            f"Input columns not found on the DataFrame: {missing_cols}. "
            f"Available columns: {df.columns}"
        )

    logger.info(f"Collecting n-grams from {len(input_columns)} columns in one pass...")

    ngram_counts = (
        df.select(*input_columns)
        .rdd.flatMap(
            lambda row: [
                ngram
                for col_value in row
                for ngram in extract_ngrams_from_column_worker(col_value, event_size)
            ]
        )
        .map(lambda ngram: (ngram, 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )

    all_ngrams = Counter(dict(ngram_counts))
    logger.info(f"Found {len(all_ngrams):,} unique n-grams across all columns")

    return all_ngrams


def log_vocabulary_by_length(ngram_to_id: Dict[Tuple[str, ...], int]) -> None:
    """
    Logs how the vocabulary splits across n-gram lengths.

    A vocabulary dominated by 1-grams means the tokenizer is mostly learning
    single ID levels and the n-gram combinations are earning little, which is the
    signal for tuning ``vocab_size`` and ``min_ngram_freq``. Counting is one pass
    over the kept n-grams, so this is cheap enough to always report.

    :param ngram_to_id: The fitted vocabulary, n-gram tuple to token id.
    :returns: None - the distribution is logged.
    """
    if not ngram_to_id:
        logger.info("Vocabulary by n-gram length: empty vocabulary")
        return
    by_length = Counter(len(ngram) for ngram in ngram_to_id)
    total = len(ngram_to_id)
    parts = [
        f"{length}-gram {by_length[length]:,} "
        f"({100.0 * by_length[length] / total:.1f}%)"
        for length in sorted(by_length)
    ]
    logger.info(f"Vocabulary by n-gram length: {' | '.join(parts)}")


def build_vocabulary(
    ngram_counter: Counter,
    vocab_size: int = 50000,
    min_ngram_freq: int = 10,
) -> Dict[Tuple[str, ...], int]:
    """
    Builds the vocabulary by filtering on frequency and keeping the top n-grams.

    N-grams below ``min_ngram_freq`` are dropped, the rest are sorted by descending
    frequency (ties broken by the n-gram itself, for determinism) and the top
    ``vocab_size - NUM_RESERVED_TOKENS`` are assigned contiguous ids from
    ``NUM_RESERVED_TOKENS`` upwards. Because ids ascend as frequency descends,
    "the top-k most frequent matching n-grams" is later just an ascending sort of ids.

    :param ngram_counter: Counter mapping n-gram tuples to frequencies.
    :param vocab_size: Target vocabulary size including the two reserved tokens.
    :param min_ngram_freq: Minimum frequency for an n-gram to be included.
    :returns: Mapping from each kept n-gram tuple to its token id.
    """
    filtered_ngrams = {
        ngram: freq for ngram, freq in ngram_counter.items() if freq >= min_ngram_freq
    }
    logger.info(
        f"After min_freq={min_ngram_freq}: {len(filtered_ngrams):,} n-grams "
        f"(from {len(ngram_counter):,} total)"
    )

    # Sort by frequency descending, then by the n-gram itself so that equal-frequency
    # n-grams are ordered deterministically across runs.
    top_ngrams = sorted(filtered_ngrams.items(), key=lambda x: (-x[1], x[0]))[
        : vocab_size - NUM_RESERVED_TOKENS
    ]
    ngram_to_id = {
        ngram: token_id
        for token_id, (ngram, _) in enumerate(top_ngrams, start=NUM_RESERVED_TOKENS)
    }

    logger.info(
        f"Final vocabulary size: {len(ngram_to_id) + NUM_RESERVED_TOKENS:,} "
        f"({len(ngram_to_id):,} n-grams + {NUM_RESERVED_TOKENS} reserved tokens)"
    )
    log_vocabulary_by_length(ngram_to_id)

    return ngram_to_id


def build_tuple_lookup_table(
    df: DataFrame,
    input_columns: List[str],
    vocabulary: EventNgramVocabulary,
    top_k: int,
    event_size: int = DEFAULT_EVENT_SIZE,
) -> Dict[Tuple[int, ...], List[int]]:
    """
    Pre-computes the ``id_tuple -> top-k token ids`` lookup table, distributed.

    All input columns are read in one pass and deduped, so each distinct event tuple is
    encoded exactly once, on the workers, via ``encode_tuple``. That encoding is a pure
    function of the tuple and the fitted vocabulary, so the assembled table does not
    depend on how the work was partitioned.

    :param df: DataFrame with the discrete-ID columns.
    :param input_columns: Column names holding discrete ID values.
    :param vocabulary: Fitted ``EventNgramVocabulary``.
    :param top_k: Number of tokens per tuple.
    :param event_size: Number of discrete ID values per tuple.
    :returns: Mapping from each event tuple to its list of ``top_k`` token ids.
    """
    if not input_columns:
        return {}

    logger.info("Building tuple->tokens lookup table (distributed)...")

    # The fitted n-gram vocabulary is bounded by vocab_size (not by the number of
    # tuples), so it is captured by the encoding closure and shipped to the workers.
    ngrams = vocabulary.ngrams

    tuple_to_tokens = (
        df.select(*input_columns)
        .rdd.flatMap(
            lambda row: [
                id_tuple
                for col_value in row
                for id_tuple in extract_tuples_from_column_worker(col_value, event_size)
            ]
        )
        .distinct()
        .map(lambda id_tuple: (id_tuple, encode_tuple(id_tuple, ngrams, top_k)))
        .collectAsMap()
    )

    logger.info(f"Encoded {len(tuple_to_tokens):,} unique tuples")

    return tuple_to_tokens


def tokenize_events(
    ids: Optional[List[int]],
    lookup_table: Dict[Tuple[int, ...], List[int]],
    num_events: int,
    tuple_size: int,
    top_k: int,
) -> List[int]:
    """
    Tokenizes one row's discrete ID values into a flat token array via the lookup table.

    Splits the row into per-event tuples and maps each to its tokens: an all-zero or
    incomplete event yields padding, a tuple in the table yields its stored tokens, and
    any other non-zero tuple yields ``<unk>``. This is the single source of truth for
    the row-level tokenization used by the Spark transform UDF, and is mirrored by the
    TensorFlow ``EventNgramLookupLayer``.

    :param ids: One row's discrete ID values as a flat int array, split into
    consecutive events of ``tuple_size`` ids. May be `None` or empty, and individual
    ids may be null (read as ``0``).
    :param lookup_table: Fitted mapping from event tuple to its top-k token list.
    :param num_events: Number of events the output is padded/truncated to.
    :param tuple_size: Number of discrete ID values per event tuple.
    :param top_k: Number of tokens per event tuple.
    :returns: Flat token array of length ``num_events * top_k``.
    """
    pad_tokens = [PAD_TOKEN_ID] * top_k
    unk_tokens = [UNK_TOKEN_ID] * top_k

    if not ids:
        return pad_tokens * num_events

    # A null id is read as 0 (an absent ID level), matching the fitting path and the
    # dense TensorFlow input, rather than producing a tuple that can never be found.
    event_tuples = [
        tuple(
            0 if id_value is None else id_value for id_value in ids[i : i + tuple_size]
        )
        for i in range(0, len(ids), tuple_size)
    ]

    all_tokens: List[int] = []
    for event_tuple in event_tuples:
        if len(event_tuple) != tuple_size or all(v == 0 for v in event_tuple):
            all_tokens.extend(pad_tokens)
        else:
            all_tokens.extend(lookup_table.get(event_tuple, unk_tokens))

    # Pad/truncate to the fixed output length.
    expected_length = num_events * top_k
    if len(all_tokens) < expected_length:
        all_tokens.extend([PAD_TOKEN_ID] * (expected_length - len(all_tokens)))
    return all_tokens[:expected_length]
