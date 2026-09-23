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
Worker functions for n-gram extraction in Spark RDD operations.

These run inside ``flatMap`` / ``map`` on the Spark workers and are pure Python over
plain ints and tuples (standard library only). Importing this module still imports the
``kamae`` package, and with it TensorFlow, so kamae must be installed on the workers,
as for kamae's other UDF-based transformers.
"""

from itertools import combinations
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
NUM_RESERVED_TOKENS = 2  # <pad> and <unk>; learned n-grams get ids from here upwards


def iter_events(col_value: Any, event_size: int) -> Iterator[List[int]]:
    """
    Yields each event of a column value as a list of ``event_size`` int ids.

    The column value is a flat integer array chunked into consecutive events, so its
    length must be a whole number of events. A null id is read as ``0``, i.e. as an
    absent ID level.

    :param col_value: Column value containing discrete ID values.
    :param event_size: Number of discrete ID values per event.
    :raises ValueError: If the number of ids is not a multiple of ``event_size``.
    :returns: Iterator over per-event lists of int ids.
    """
    if col_value is None:
        return
    if len(col_value) % event_size != 0:
        raise ValueError(
            f"Number of discrete ID values must be a whole number of events: got "
            f"{len(col_value)} ids, which is not a multiple of the event size "
            f"{event_size}."
        )
    for i in range(0, len(col_value), event_size):
        yield [
            0 if id_value is None else id_value
            for id_value in col_value[i : i + event_size]
        ]


def event_ngrams(event_ids: Sequence[int]) -> Iterator[Tuple[str, ...]]:
    """
    Yields every within-event n-gram of a single event.

    The event's non-zero ids are level-prefixed (``L<level>_<id>``) and every
    combination of length 1..n is emitted, so n-grams that skip ID levels are included.
    Zero ids denote absent ID levels and take no part in any n-gram. This is the single
    definition of an n-gram, used both when counting them and when encoding a tuple
    against a fitted vocabulary.

    :param event_ids: One event's ids, one per ID level.
    :returns: Iterator over the event's n-gram tuples.
    """
    present = [
        f"L{level}_{id_value}"
        for level, id_value in enumerate(event_ids)
        if id_value != 0
    ]
    for length in range(1, len(present) + 1):
        yield from combinations(present, length)


def extract_ngrams_from_column_worker(
    col_value: Any, event_size: int
) -> List[Tuple[str, ...]]:
    """
    Extracts all within-event n-grams from a single row's column value.

    N-grams are never formed across event boundaries.

    :param col_value: Column value containing discrete ID values.
    :param event_size: Number of discrete ID values per event.
    :returns: List of n-gram tuples.
    """
    return [
        ngram
        for event_ids in iter_events(col_value, event_size)
        for ngram in event_ngrams(event_ids)
    ]


def extract_tuples_from_column_worker(
    col_value: Any, event_size: int
) -> List[Tuple[int, ...]]:
    """
    Extracts all non-zero event tuples from a single row's column value.

    :param col_value: Column value containing discrete ID values.
    :param event_size: Number of discrete ID values per tuple.
    :returns: List of ID tuples (all-zero events are skipped).
    """
    return [
        tuple(event_ids)
        for event_ids in iter_events(col_value, event_size)
        if any(id_value != 0 for id_value in event_ids)
    ]


def encode_tuple(
    id_tuple: Tuple[int, ...],
    ngrams: Dict[Tuple[str, ...], int],
    top_k: int,
) -> Optional[List[int]]:
    """
    Encodes one event tuple into its top-k token ids against a fitted vocabulary.

    Keeps the tuple's n-grams that are present in ``ngrams`` and returns the ``top_k``
    smallest ids; ids are assigned by descending frequency, so a smaller id is a more
    frequent n-gram.

    Returns `None` when the tuple has no token to contribute, i.e. when it is all-zero
    or when every one of its n-grams missed the vocabulary. Such a tuple is left out of
    the lookup table, so at inference it takes the same table-miss path as a tuple that
    was never seen during fitting. Unknown is therefore decided in one place, by one
    rule, rather than once here and again at inference.

    Runs inside a Spark ``map`` on the workers, and is the single source of truth for
    per-tuple encoding used by ``build_tuple_lookup_table``.

    :param id_tuple: Event tuple of IDs.
    :param ngrams: Fitted mapping from n-gram tuple to token id.
    :param top_k: Number of tokens to return.
    :returns: List of token ids of length ``top_k`` (padded if needed), or `None` if the
    tuple matched no n-gram.
    """
    token_ids = sorted(
        ngrams[ngram] for ngram in event_ngrams(id_tuple) if ngram in ngrams
    )
    if not token_ids:
        return None

    tokens = token_ids[:top_k]
    return tokens + [PAD_TOKEN_ID] * (top_k - len(tokens))
