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
TensorFlow layer that tokenizes event-level discrete ID values via a lookup table.

Companion layer to the ``EventNgramLookupEstimator`` / ``EventNgramLookupTransformer``.
Each input row holds a sequence of events, and each event is a fixed-size tuple of
discrete ID values (e.g. 4 levels ``L0, L1, L2, L3``). The layer splits each input into
per-event tuples and maps every tuple to its pre-computed top-k token list, so that
inference is a pure ``O(1)`` table lookup identical to the Spark ``_transform`` path.

A single layer instance tokenizes ALL input columns (one per ``num_events_per_input``
entry), so the fitted lookup table is embedded once regardless of the number of inputs.
It returns one token tensor per input, in input order. When a ``token_type_lookup`` is
supplied, it additionally returns one "type" tensor per input (each token id mapped to a
bitmask of the ID levels its n-gram spans), grouped after all the token tensors:
``[tokens_0, ..., tokens_{N-1}, types_0, ..., types_{N-1}]``.

Each event tuple is bijectively packed into a single ``int64`` key and resolved by a
Keras ``IntegerLookup`` sublayer plus a gathered values tensor, applied fully
vectorised (no per-element ``map_fn`` / ``cond`` and no string ops in the serving
path). The table is persisted in ``get_config`` as parallel ``lookup_keys`` /
``lookup_values`` lists so the layer reloads standalone from a ``.keras`` file and
exports cleanly to SavedModel.

Token id conventions (shared with the vocabulary):
    - ``0`` = padding (``<pad>``): emitted for all-zero / missing events.
    - ``1`` = unknown (``<unk>``): emitted for a non-zero tuple absent from the table.
"""

from typing import Any, Dict, List, Optional, Union

import keras
import tensorflow as tf
from keras import KerasTensor

import kamae
from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class EventNgramLookupLayer(BaseLayer):
    """
    Tokenizes discrete ID values using a pre-computed tuple to top-k token lookup table.

    For each event position the output is the tuple's ``top_k`` token ids. Unknown
    (non-zero) tuples map to ``<unk>`` and all-zero / padding events map to ``<pad>``,
    matching the Spark transformer exactly.

    One layer tokenizes all input columns (``num_events_per_input`` gives the event
    count of each). Each input may be rank-2 ``(batch, num_events * tuple_size)`` or
    rank-3 ``(batch, list_size, num_events * tuple_size)``; the list dimension (e.g.
    per listwise item) is preserved. The id axis is padded/truncated to
    ``num_events * tuple_size`` before being split into tuples.

    Outputs one token tensor per input. With ``token_type_lookup`` set it also outputs
    one type tensor per input (same shape as the tokens), grouped after the tokens.
    """

    supported_backends = TENSORFLOW_ONLY
    jit_compatible = False

    def __init__(
        self,
        num_events_per_input: List[int],
        top_k: int,
        tuple_size: int,
        lookup_keys: Optional[List[List[int]]] = None,
        lookup_values: Optional[List[List[int]]] = None,
        token_type_lookup: Optional[List[int]] = None,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the EventNgramLookupLayer layer.

        The lookup table is supplied as the parallel ``lookup_keys`` / ``lookup_values``
        lists, which is both what the transformer passes and what ``get_config``
        persists, so construction and reload take the same path.

        :param num_events_per_input: Number of events for each input column, in input
        order (e.g. ``[10, 10, 1]``). Its length is the number of inputs the layer
        tokenizes.
        :param top_k: Number of tokens emitted per event tuple.
        :param tuple_size: Number of ID levels per event tuple.
        :param lookup_keys: Lookup table keys: one ``tuple_size``-long int list per
        event tuple. Defaults to `None` (an empty table, so every tuple is unknown).
        :param lookup_values: Lookup table values: one ``top_k``-long token list per
        key, positionally aligned with ``lookup_keys``. Defaults to `None`.
        :param token_type_lookup: Optional per-token-id list mapping each token to a
        bitmask of the ID levels its n-gram spans. When given, the layer also emits
        a type tensor per input. Defaults to `None` (tokens only).
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.num_events_per_input = [int(n) for n in num_events_per_input]
        self.top_k = int(top_k)
        self.tuple_size = int(tuple_size)

        self._lookup_keys = lookup_keys if lookup_keys is not None else []
        self._lookup_values = lookup_values if lookup_values is not None else []
        self._build_lookup_table(self._lookup_keys, self._lookup_values)

        # Optional per-token type lookup (bitmask of ID levels; 0 = pad/unk).
        self._token_type_lookup = token_type_lookup
        if token_type_lookup is not None:
            self._type_tensor = tf.constant(token_type_lookup, dtype=tf.int32)

    def _build_lookup_table(
        self, keys: List[List[int]], values: List[List[int]]
    ) -> None:
        """
        Builds the ``IntegerLookup`` sublayer and gathered values tensor.

        Each tuple key is bijectively packed into a single ``int64`` by allotting
        ``key_bits`` bits per ID level, where ``key_bits`` is the bit length of the
        largest ID in the table. Packing lets one ``IntegerLookup`` index the whole
        table, which keeps string ops out of the serving path. The lookup maps a
        miss/OOV to index 0 and the i-th key to index ``i + 1``; ``_call`` shifts back
        by one to gather from the values tensor.

        :param keys: Tuple keys as a list of int lists.
        :param values: Token lists (one per key), each of length ``top_k``.
        :raises ValueError: If an ID is negative, or the packed key would not fit in a
        signed ``int64``.
        """
        key_ids = [int(x) for k in keys for x in k]
        min_id = min(key_ids, default=0)
        if min_id < 0:
            raise ValueError(
                f"Discrete ID values must be non-negative, but the lookup table "
                f"contains {min_id}. Each event tuple is packed into a single "
                f"non-negative int64 key, which a negative ID cannot represent."
            )
        max_id = max(key_ids, default=1)
        self.key_bits = max(1, max_id.bit_length())
        if self.tuple_size * self.key_bits > 63:
            raise ValueError(
                f"Cannot pack event tuples into a signed int64: tuple_size "
                f"({self.tuple_size}) * key_bits ({self.key_bits}, from a largest ID "
                f"of {max_id}) = {self.tuple_size * self.key_bits} bits, which exceeds "
                f"63. Reduce tuple_size or the ID cardinality."
            )
        self.key_powers = [1 << (j * self.key_bits) for j in range(self.tuple_size)]
        packed_keys = [
            sum(int(x) * p for x, p in zip(k, self.key_powers)) for k in keys
        ]

        # Keep at least one value row so the gather in _call stays valid even for a
        # (degenerate) empty table.
        self.values_tensor = (
            tf.constant(values, dtype=tf.int32)
            if values
            else tf.zeros([1, self.top_k], dtype=tf.int32)
        )
        # An empty table uses one dummy key that real packed keys, always
        # non-negative, never match, so every lookup resolves to OOV (index 0). It must
        # be -2: IntegerLookup strips a leading run of its own special tokens, and -1 is
        # its default oov_token, so a vocabulary of just [-1] is left empty.
        self.key_to_index = keras.layers.IntegerLookup(
            vocabulary=packed_keys if packed_keys else [-2],
            num_oov_indices=1,
            mask_token=None,
            name=f"{self.name}_key_lookup",
        )
        self.unk_pattern = tf.constant([UNK_TOKEN_ID] * self.top_k, dtype=tf.int32)
        self.pad_pattern = tf.constant([PAD_TOKEN_ID] * self.top_k, dtype=tf.int32)

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return ["int32", "int64"]

    def _tokenize(self, inputs: KerasTensor, num_events: int) -> KerasTensor:
        """
        Tokenizes one input tensor into its flat token ids.

        :param inputs: ID tensor, rank-2 ``(batch, num_events * tuple_size)`` or rank-3
        ``(batch, list_size, num_events * tuple_size)``.
        :param num_events: Number of events this input is padded/truncated to.
        :returns: Token id tensor of shape ``(batch, num_events * top_k)`` for a
        rank-2 input, or ``(batch, list_size, num_events * top_k)`` for rank-3.
        """
        output_length = num_events * self.top_k
        input_rank = len(inputs.shape)
        if input_rank == 3:
            batch_size = tf.shape(inputs)[0]
            list_or_1 = tf.shape(inputs)[1]
            inputs_flat = tf.reshape(inputs, [-1, tf.shape(inputs)[2]])
            restore_list = True
        else:
            inputs_flat = inputs
            restore_list = False

        # Pad/truncate the id axis to num_events * tuple_size, then split into tuples.
        # Padding by the full expected length before slicing covers both the short and
        # the long case without branching on the input width.
        expected_length = num_events * self.tuple_size
        inputs_padded = tf.pad(inputs_flat, [[0, 0], [0, expected_length]])[
            :, :expected_length
        ]
        all_tuples = tf.reshape(inputs_padded, [-1, self.tuple_size])

        # All-zero tuples are padding; everything else is looked up (miss -> UNK).
        is_padding = tf.reduce_all(tf.equal(all_tuples, 0), axis=1)

        # Pack each tuple into one int64 key. IDs outside [0, max_id] cannot be
        # represented, so they are clamped for packing and then forced to miss --
        # clamping alone would let an out-of-range ID alias onto a valid key.
        max_id = (1 << self.key_bits) - 1
        ids_64 = tf.cast(all_tuples, tf.int64)
        in_range = tf.reduce_all((ids_64 >= 0) & (ids_64 <= max_id), axis=1)
        powers = tf.constant(self.key_powers, dtype=tf.int64)
        tuple_keys = tf.reduce_sum(tf.clip_by_value(ids_64, 0, max_id) * powers, axis=1)

        # Shift back by one (IntegerLookup uses index 0 for OOV) to index values_tensor.
        indices = self.key_to_index(tuple_keys)
        found = (indices > 0) & in_range
        gathered = tf.gather(self.values_tensor, tf.maximum(indices - 1, 0))

        # tf.where broadcasts, so the (top_k,) unk/pad constants apply as they are,
        # with no per-tuple copy materialised.
        tokens = tf.where(found[:, None], gathered, self.unk_pattern)
        tokens = tf.where(is_padding[:, None], self.pad_pattern, tokens)

        if restore_list:
            return tf.reshape(tokens, [batch_size, list_or_1, output_length])
        return tf.reshape(tokens, [-1, output_length])

    def _call(
        self, inputs: Union[KerasTensor, List[KerasTensor]], **kwargs: Any
    ) -> Union[KerasTensor, List[KerasTensor]]:
        """
        Tokenizes every input, and optionally derives their per-token types.

        :param inputs: List of ID tensors (one per ``num_events_per_input`` entry). A
        single input may be passed as a bare tensor.
        :returns: One token tensor per input. If a ``token_type_lookup`` was supplied,
        the per-input type tensors are appended after the token tensors. A single token
        tensor (one input, no types) is returned unwrapped.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        tokens = [
            self._tokenize(inp, num_events)
            for inp, num_events in zip(inputs, self.num_events_per_input)
        ]

        if self._token_type_lookup is None:
            return tokens[0] if len(tokens) == 1 else tokens

        # Type = per-token bitmask of contributing ID levels, gathered by token id.
        types = [tf.gather(self._type_tensor, tok) for tok in tokens]
        return tokens + types

    def compute_output_shape(
        self, input_shape: Union[tuple, List[tuple]]
    ) -> Union[tuple, List[tuple]]:
        """
        Declares the output shape(s) for functional model building.

        :param input_shape: A single input shape, or a list of input shapes (one per
        input column).
        :returns: The matching output shape(s), with each id axis replaced by
        ``num_events * top_k``. When types are emitted, the per-input type shapes are
        appended after the token shapes.
        """
        single = not isinstance(input_shape[0], (list, tuple))
        shapes = [input_shape] if single else list(input_shape)

        token_shapes = []
        for shape, num_events in zip(shapes, self.num_events_per_input):
            output_length = num_events * self.top_k
            if len(shape) == 3:
                token_shapes.append((shape[0], shape[1], output_length))
            elif len(shape) == 2:
                token_shapes.append((shape[0], output_length))
            else:
                token_shapes.append(shape)

        if self._token_type_lookup is not None:
            # One type tensor per input, each the shape of its tokens, appended after
            # all the token tensors.
            return token_shapes + token_shapes
        return token_shapes[0] if single else token_shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the EventNgramLookupLayer layer.
        Used for saving and loading the layer from a model.

        Persists the lookup table as parallel ``lookup_keys`` / ``lookup_values`` lists
        (and the optional ``token_type_lookup``) so the layer is reconstructed by
        ``__init__`` without a custom ``from_config``.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "num_events_per_input": self.num_events_per_input,
                "top_k": self.top_k,
                "tuple_size": self.tuple_size,
                "lookup_keys": self._lookup_keys,
                "lookup_values": self._lookup_values,
                "token_type_lookup": self._token_type_lookup,
            }
        )
        return config
