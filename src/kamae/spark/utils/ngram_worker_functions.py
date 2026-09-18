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

These functions run on Spark workers, so they depend only on the Python standard
library (no TensorFlow / kamae imports).
"""

from itertools import combinations
from typing import Any, Dict, Iterator, List, Tuple

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
NUM_RESERVED_TOKENS = 2  # <pad> and <unk>; learned n-grams get ids from here upwards


def _iter_events(col_value: Any, event_size: int) -> Iterator[List[int]]:
    """
    Yields each event of a column value as a list of ``event_size`` int ids.

    The column value is a flat integer array chunked into consecutive events. A
    trailing chunk with fewer than ``event_size`` ids is skipped, as it cannot be a
    complete event. A null id is read as ``0``, i.e. as an absent ID level, which is
    what the dense TensorFlow input carries for a missing id.

    :param col_value: Column value containing discrete ID values.
    :param event_size: Number of discrete ID values per event.
    :returns: Iterator over per-event lists of int ids.
    """
    if col_value is None:
        return
    for i in range(0, len(col_value), event_size):
        chunk = col_value[i : i + event_size]
        if len(chunk) == event_size:
            yield [0 if id_value is None else id_value for id_value in chunk]


def extract_ngrams_from_column_worker(
    col_value: Any, event_size: int = 4
) -> List[Tuple[str, ...]]:
    """
    Extracts all within-event n-grams from a single row's column value.

    Each event's non-zero ids are level-prefixed (``L<level>_<id>``) and every
    combination of length 1..n is emitted (so n-grams that skip ID levels are
    included). N-grams are never formed across event boundaries.

    :param col_value: Column value containing discrete ID values.
    :param event_size: Number of discrete ID values per event.
    :returns: List of n-gram tuples.
    """
    ngrams_list = []
    for event_ids in _iter_events(col_value, event_size):
        event_strs = [f"L{j}_{v}" for j, v in enumerate(event_ids) if v != 0]
        for length in range(1, len(event_strs) + 1):
            for combo in combinations(range(len(event_strs)), length):
                ngrams_list.append(tuple(event_strs[i] for i in combo))
    return ngrams_list


def extract_tuples_from_column_worker(
    col_value: Any, event_size: int = 4
) -> List[Tuple[int, ...]]:
    """
    Extracts all non-zero event tuples from a single row's column value.

    :param col_value: Column value containing discrete ID values.
    :param event_size: Number of discrete ID values per tuple.
    :returns: List of ID tuples (all-zero events are skipped).
    """
    tuples_list = []
    for event_ids in _iter_events(col_value, event_size):
        if any(v != 0 for v in event_ids):
            tuples_list.append(tuple(event_ids))
    return tuples_list


def encode_tuple(
    id_tuple: Tuple[int, ...],
    ngrams: Dict[Tuple[str, ...], int],
    top_k: int,
) -> List[int]:
    """
    Encodes one event tuple into its top-k token ids against a fitted vocabulary.

    Level-prefixes the non-zero ids (``L<level>_<id>``), enumerates every combination
    of length 1..n (so n-grams skipping ID levels are considered), keeps those present
    in ``ngrams``, and returns the ``top_k`` smallest ids (ids are assigned by
    frequency, so smaller id = more frequent). An all-zero tuple yields padding; a
    non-zero tuple whose every combination missed the vocabulary yields ``<unk>``, one
    per present ID level.

    Runs inside a Spark ``map`` on the workers, and is the single source of truth for
    per-tuple encoding used by ``build_tuple_lookup_table``.

    :param id_tuple: Event tuple of IDs.
    :param ngrams: Fitted mapping from n-gram tuple to token id.
    :param top_k: Number of tokens to return.
    :returns: List of token ids of length ``top_k`` (padded if needed).
    """
    event = [f"L{j}_{v}" for j, v in enumerate(id_tuple) if v != 0]
    if not event:
        return [PAD_TOKEN_ID] * top_k

    candidate_ids = []
    for length in range(len(event), 0, -1):
        for combo in combinations(range(len(event)), length):
            ngram = tuple(event[i] for i in combo)
            if ngram in ngrams:
                candidate_ids.append(ngrams[ngram])

    if not candidate_ids:
        num_unk = min(len(event), top_k)
        return [UNK_TOKEN_ID] * num_unk + [PAD_TOKEN_ID] * (top_k - num_unk)

    # Ids are unique per n-gram, so sorting ascending gives the most frequent first.
    candidate_ids.sort()
    tokens = candidate_ids[:top_k]
    return tokens + [PAD_TOKEN_ID] * (top_k - len(tokens))
