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

import keras
import pytest
import tensorflow as tf

from kamae.keras.tensorflow.layers import EventNgramLookupLayer

# Fitted table shared across the tests: two known 4-tuples, top_k = 3, held as the
# parallel key/value lists the layer takes.
LOOKUP_KEYS = [[1, 2, 3, 4], [5, 6, 7, 8]]
LOOKUP_VALUES = [[2, 3, 0], [4, 5, 0]]
TOP_K = 3
TUPLE_SIZE = 4


def _layer(**overrides) -> EventNgramLookupLayer:
    params = dict(
        name="event_ngram_lookup",
        num_events_per_input=[2],
        top_k=TOP_K,
        tuple_size=TUPLE_SIZE,
        lookup_keys=LOOKUP_KEYS,
        lookup_values=LOOKUP_VALUES,
    )
    params.update(overrides)
    return EventNgramLookupLayer(**params)


class TestEventNgramLookupLayer:
    def test_layer_name_and_dtypes(self):
        layer = _layer()
        assert layer.name == "event_ngram_lookup", "Layer name is not set properly"
        assert layer.compatible_dtypes == ["int32", "int64"]

    @pytest.mark.parametrize("input_dtype", [tf.int32, tf.int64])
    def test_known_unknown_and_padding_tuples(self, input_dtype):
        # given
        inputs = tf.constant(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],  # known, known
                [1, 2, 3, 4, 9, 9, 9, 9],  # known, unknown -> unk
                [0, 0, 0, 0, 0, 0, 0, 0],  # padding, padding -> pad
            ],
            dtype=input_dtype,
        )

        # when
        output = _layer()(inputs)

        # then
        assert output.shape == (3, 2 * TOP_K)
        tf.debugging.assert_equal(
            output,
            tf.constant(
                [[2, 3, 0, 4, 5, 0], [2, 3, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
                dtype=output.dtype,
            ),
        )

    def test_preserves_list_dimension_for_rank_three_input(self):
        # given: (batch=2, list_size=2, num_events * tuple_size=8)
        inputs = tf.constant(
            [
                [[1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0]],
                [[5, 6, 7, 8, 1, 2, 3, 4], [9, 9, 9, 9, 0, 0, 0, 0]],
            ],
            dtype=tf.int32,
        )

        # when
        output = _layer()(inputs)

        # then
        assert output.shape == (2, 2, 2 * TOP_K)
        tf.debugging.assert_equal(
            output,
            tf.constant(
                [
                    [[2, 3, 0, 4, 5, 0], [0, 0, 0, 0, 0, 0]],
                    [[4, 5, 0, 2, 3, 0], [1, 0, 0, 0, 0, 0]],
                ],
                dtype=output.dtype,
            ),
        )

    @pytest.mark.parametrize(
        "inputs, expected",
        [
            # Short id axis is zero-padded up to num_events * tuple_size.
            ([[1, 2, 3, 4]], [[2, 3, 0, 0, 0, 0]]),
            # Long id axis is truncated to num_events * tuple_size.
            ([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]], [[2, 3, 0, 4, 5, 0]]),
        ],
    )
    def test_pads_and_truncates_the_id_axis(self, inputs, expected):
        output = _layer()(tf.constant(inputs, dtype=tf.int32))
        tf.debugging.assert_equal(output, tf.constant(expected, dtype=output.dtype))

    def test_tokenizes_every_input_with_its_own_event_count(self):
        # given: two inputs with different numbers of events, one shared table.
        layer = _layer(num_events_per_input=[2, 1])
        clicks = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tf.int32)
        prop = tf.constant([[5, 6, 7, 8]], dtype=tf.int32)

        # when
        tokens = layer([clicks, prop])

        # then
        assert len(tokens) == 2
        assert tokens[0].shape == (1, 2 * TOP_K)
        assert tokens[1].shape == (1, 1 * TOP_K)

    def test_emits_a_type_tensor_per_input_after_the_token_tensors(self):
        # given: token id -> ID-level bitmask.
        type_lookup = [0, 0, 5, 1, 8, 3]
        layer = _layer(token_type_lookup=type_lookup)
        inputs = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tf.int32)

        # when
        tokens, types = layer(inputs)

        # then
        assert types.shape == tokens.shape
        tf.debugging.assert_equal(
            types,
            tf.constant(
                [[type_lookup[t] for t in tokens.numpy()[0].tolist()]],
                dtype=types.dtype,
            ),
        )

    @pytest.mark.parametrize("tuple_size", [1, 2, 3, 5])
    def test_supports_tuple_sizes_other_than_four(self, tuple_size):
        # given
        layer = _layer(
            num_events_per_input=[1],
            tuple_size=tuple_size,
            lookup_keys=[list(range(1, tuple_size + 1))],
            lookup_values=[[2] * TOP_K],
        )

        # when
        output = layer(tf.constant([list(range(1, tuple_size + 1))], dtype=tf.int32))

        # then
        tf.debugging.assert_equal(
            output, tf.constant([[2] * TOP_K], dtype=output.dtype)
        )

    def test_empty_table_maps_every_non_zero_tuple_to_unk(self):
        layer = _layer(lookup_keys=[], lookup_values=[])
        output = layer(tf.constant([[1, 2, 3, 4, 0, 0, 0, 0]], dtype=tf.int32))
        tf.debugging.assert_equal(
            output, tf.constant([[1, 0, 0, 0, 0, 0]], dtype=output.dtype)
        )

    def test_out_of_range_id_misses_instead_of_aliasing_a_valid_key(self):
        # given: key_bits is sized from the largest id in the table (8 -> 4 bits), so
        # id 17 does not fit. Clamping alone would turn (17, 2, 3, 4) into (15, 2, 3, 4)
        # and could alias onto a real key, so out-of-range tuples must miss.
        layer = _layer(num_events_per_input=[1])
        assert layer.key_bits == 4

        # when
        output = layer(tf.constant([[17, 2, 3, 4]], dtype=tf.int32))

        # then
        tf.debugging.assert_equal(output, tf.constant([[1, 0, 0]], dtype=output.dtype))

    def test_negative_id_misses_instead_of_aliasing_a_valid_key(self):
        # given: a table key whose first ID level is 0, which is what clamping a
        # negative id would produce, so a negative id must miss rather than resolve
        # to this key's tokens.
        layer = _layer(
            num_events_per_input=[1],
            lookup_keys=[[0, 6, 7, 8]],
            lookup_values=[[2, 3, 0]],
        )

        # when
        output = layer(tf.constant([[-3, 6, 7, 8]], dtype=tf.int32))

        # then
        tf.debugging.assert_equal(output, tf.constant([[1, 0, 0]], dtype=output.dtype))

    def test_raises_when_the_table_contains_a_negative_id(self):
        # Packing allots a fixed number of bits per ID level, so a negative id would
        # collide with the non-negative tuple that shares its remaining levels.
        with pytest.raises(ValueError, match="non-negative"):
            _layer(
                num_events_per_input=[1],
                lookup_keys=[[-1, 6, 7, 8]],
                lookup_values=[[2, 3, 0]],
            )

    def test_raises_when_keys_do_not_fit_in_an_int64(self):
        # 16 levels of a 2**32-sized id space needs far more than 63 bits.
        with pytest.raises(ValueError, match="int64"):
            _layer(
                num_events_per_input=[1],
                tuple_size=16,
                lookup_keys=[[2**32] * 16],
                lookup_values=[[2] * TOP_K],
            )

    def test_get_config_round_trips_the_table(self):
        # given
        layer = _layer(token_type_lookup=[0, 0, 5, 1, 8, 3])
        inputs = tf.constant([[1, 2, 3, 4, 9, 9, 9, 9]], dtype=tf.int32)
        expected_tokens, expected_types = layer(inputs)

        # when: the table is persisted as parallel key/value lists, so the rebuilt
        # layer must tokenize identically without a custom from_config.
        rebuilt = EventNgramLookupLayer.from_config(layer.get_config())
        tokens, types = rebuilt(inputs)

        # then
        tf.debugging.assert_equal(tokens, expected_tokens)
        tf.debugging.assert_equal(types, expected_types)

    def test_symbolic_outputs_are_int32_with_the_tokenized_shape(self):
        # Keras infers the symbolic outputs by tracing _call, so they carry the real
        # int32 dtype and a downstream integer-only layer can be chained onto them.
        layer = _layer(
            num_events_per_input=[2, 1], token_type_lookup=[0, 0, 5, 1, 8, 3]
        )
        clicks = keras.Input(shape=(8,), dtype="int32")
        prop = keras.Input(shape=(None, 4), dtype="int64")

        outputs = layer([clicks, prop])

        assert [(output.dtype, output.shape) for output in outputs] == [
            ("int32", (None, 2 * TOP_K)),
            ("int32", (None, None, TOP_K)),
            ("int32", (None, 2 * TOP_K)),
            ("int32", (None, None, TOP_K)),
        ]

    def test_raises_when_inputs_do_not_match_num_events_per_input(self):
        # Inputs are paired with their event counts, so a mismatch would otherwise
        # silently drop the unpaired inputs' outputs.
        layer = _layer(num_events_per_input=[2, 1])
        with pytest.raises(ValueError, match="num_events_per_input"):
            layer(tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tf.int32))
