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
EventNgramLookupTransformer: tokenizes discrete IDs with a pre-computed lookup table.

Applies the ``tuple -> top-k tokens`` table fitted by ``EventNgramLookupEstimator``.
Both the Spark path (``_transform``) and the TensorFlow path (``get_keras_layer``) are
plain ``O(1)`` lookups, so the same event tuple always maps to the same tokens and the
two paths produce identical output.

When ``includeTokenTypes`` is set, each token column ``<col>`` is accompanied by a
parallel ``<col>_types`` column giving each token's ID-level bitmask (a compact
categorical feature for the model). The type is a pure function of the token id, so it
is derived by gathering the fitted ``tokenTypeLookup`` at the token ids.

Token id conventions (shared with the vocabulary and the TF layer):
    - ``0`` = padding (``<pad>``): emitted for all-zero / missing events.
    - ``1`` = unknown (``<unk>``): emitted for a non-zero tuple absent from the table.
"""

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
from typing import Any, Dict, List, Optional, Tuple

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, IntegerType, StructField, StructType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import EventNgramLookupLayer
from kamae.spark.params import EventNgramLookupParams, MultiInputMultiOutputParams
from kamae.spark.utils.ngram_utils import tokenize_events

from .base import BaseTransformer


class EventNgramLookupTransformer(
    BaseTransformer,
    MultiInputMultiOutputParams,
    EventNgramLookupParams,
):
    """
    Tokenizes discrete ID values using the pre-computed ``tuple -> top-k tokens`` table.

    Each input column holds a sequence of events; every event is a ``tupleSize``-long
    group of IDs. For each event the transformer emits the tuple's ``topK``
    token ids (padding for all-zero events, ``<unk>`` for non-zero tuples absent from
    the table), producing a flat ``numEvents * topK`` array per input column. When
    ``includeTokenTypes`` is set, a parallel ``<col>_types`` column of the same shape is
    also produced, giving each token's ID-level bitmask.
    """

    supported_backends = TENSORFLOW_ONLY
    jit_compatible = False

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCols: Optional[List[str]] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        numEventsPerInput: Optional[List[int]] = None,
        tupleSize: int = 4,
        topK: Optional[int] = None,
        vocabularySize: Optional[int] = None,
        lookupKeys: Optional[List[int]] = None,
        lookupValues: Optional[List[int]] = None,
        includeTokenTypes: bool = False,
        tokenTypeLookup: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the EventNgramLookupTransformer transformer.

        :param inputCols: Input column names holding discrete ID values.
        :param outputCols: Output column names for the token arrays.
        :param inputDtype: Data type to cast the input columns to before transforming.
        :param outputDtype: Data type to cast the output columns to after transforming.
        :param layerName: Name of the layer. Used as the name of the Keras layer in the
        keras model. If not set, we use the uid of the Spark transformer.
        :param numEventsPerInput: Number of events per input column, e.g. [10, 10, 1].
        :param tupleSize: Number of discrete ID values (ID levels) per event tuple.
        :param topK: Number of tokens emitted per event tuple.
        :param vocabularySize: Total number of unique tokens in the fitted vocabulary.
        :param lookupKeys: Fitted lookup table keys, as the event tuples flattened to a
        single int list (``tupleSize`` ids per tuple).
        :param lookupValues: Fitted lookup table values, as the token lists flattened to
        a single int list (``topK`` tokens per tuple), aligned with ``lookupKeys``.
        :param includeTokenTypes: If `True`, also emit a ``<col>_types`` column per
        input giving each token's ID-level bitmask. Defaults to `False`.
        :param tokenTypeLookup: Per-token-id list of ID-level bitmasks (from the fitted
        vocabulary). Required when ``includeTokenTypes`` is `True`.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(
            numEventsPerInput=None,
            tupleSize=4,
            topK=None,
            vocabularySize=None,
            lookupKeys=None,
            lookupValues=None,
            includeTokenTypes=False,
            tokenTypeLookup=None,
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [IntegerType(), ArrayType(IntegerType())]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Tokenizes each input column with a per-column UDF over ``tokenize_events``.

        Each input value is a flat integer array, split into consecutive events of
        ``tupleSize`` ids. The output is a flat integer array of length
        ``numEvents * topK``. When ``includeTokenTypes`` is set, a ``<col>_types``
        column is added by gathering the type bitmask at each token id.

        :param dataset: Input DataFrame.
        :returns: DataFrame with the tokenized output columns, and the optional type
        columns.
        """
        tuple_size = self.getTupleSize()
        top_k = self.getTopK()
        lookup_table = self.getTupleToTokens()
        include_types = self.getIncludeTokenTypes()
        type_lookup = self.getTokenTypeLookup() if include_types else None

        token_array_type = ArrayType(IntegerType())
        for input_col, output_col, num_events in zip(
            self.getInputCols(),
            self.getOutputCols(),
            self.getNumEventsPerInput(),
        ):
            # num_events is bound per column via the default argument so each UDF
            # captures its own value rather than the last loop iteration's.
            if not include_types:
                tokenize_udf = F.udf(
                    lambda ids, num_events=num_events: tokenize_events(
                        ids, lookup_table, num_events, tuple_size, top_k
                    ),
                    token_array_type,
                )
                dataset = dataset.withColumn(output_col, tokenize_udf(F.col(input_col)))
                continue

            # A token's type is a pure function of its id, so both are produced by one
            # UDF returning a struct, keeping the column to a single Python round-trip.
            def _tokenize_with_types(
                ids: Optional[List[int]], num_events: int = num_events
            ) -> Tuple[List[int], List[int]]:
                """Tokenizes one row and gathers each token's type bitmask.

                :param ids: One row's discrete ID values.
                :param num_events: Number of events this column is sized to.
                :returns: Tuple of (token ids, type bitmasks), same length.
                """
                tokens = tokenize_events(
                    ids, lookup_table, num_events, tuple_size, top_k
                )
                return tokens, [type_lookup[token] for token in tokens]

            tokenize_udf = F.udf(
                _tokenize_with_types,
                StructType(
                    [
                        StructField("tokens", token_array_type),
                        StructField("types", token_array_type),
                    ]
                ),
            )
            struct_col = f"{output_col}__tokens_and_types"
            dataset = (
                dataset.withColumn(struct_col, tokenize_udf(F.col(input_col)))
                .withColumn(output_col, F.col(f"{struct_col}.tokens"))
                .withColumn(f"{output_col}_types", F.col(f"{struct_col}.types"))
                .drop(struct_col)
            )

        return dataset

    def get_keras_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the Keras layer for the EventNgramLookup transformer.

        Returns a single ``EventNgramLookupLayer`` that tokenizes all input columns (so
        the fitted lookup table is embedded once), for in-graph tokenization on
        TensorFlow (needed when reading raw files via TFParquet). The layer outputs one
        token tensor per input, plus one type tensor per input when
        ``includeTokenTypes`` is set.

        :returns: The consolidated ``EventNgramLookupLayer``.
        """
        # The layer takes the table as nested key/value lists, so the flat params are
        # reshaped back into one list per tuple.
        tuple_size = self.getTupleSize()
        top_k = self.getTopK()
        keys = self.getLookupKeys() or []
        values = self.getLookupValues() or []
        return EventNgramLookupLayer(
            num_events_per_input=self.getNumEventsPerInput(),
            top_k=top_k,
            tuple_size=tuple_size,
            lookup_keys=[
                keys[i : i + tuple_size] for i in range(0, len(keys), tuple_size)
            ],
            lookup_values=[values[i : i + top_k] for i in range(0, len(values), top_k)],
            token_type_lookup=(
                self.getTokenTypeLookup() if self.getIncludeTokenTypes() else None
            ),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            name=f"{self.getLayerName()}_tokenizer",
        )

    def construct_layer_info(self) -> Dict[str, Any]:
        """
        Constructs the layer info dictionary, appending the token-type output columns.

        Overrides the base method because the consolidated layer emits more outputs
        than there are output columns when ``includeTokenTypes`` is set: it returns the
        per-input type tensors after the token tensors, so the ``<col>_types`` columns
        must be appended in that same order for the pipeline graph to zip them up
        correctly.

        :returns: Dictionary with the layer name, Keras layer, inputs and outputs.
        """
        inputs, token_cols = self.get_layer_inputs_outputs()

        outputs = list(token_cols)
        if self.getIncludeTokenTypes():
            outputs += [f"{col}_types" for col in token_cols]

        return {
            "name": self.getOrDefault("layerName"),
            "layer": self.get_keras_layer(),
            "inputs": inputs,
            "outputs": outputs,
        }
