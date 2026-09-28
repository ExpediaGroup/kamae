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
The Spark path (``_transform``) and the Keras layer (``get_keras_layer``) apply the
same table, so both produce identical tokens.

When ``includeTokenTypes`` is set, each token column ``<col>`` is accompanied by a
parallel ``<col>_types`` column giving each token's ID-level bitmask.

Token id conventions (shared with the vocabulary and the Keras layer):
    - ``0`` = padding (``<pad>``): emitted for all-zero / missing events.
    - ``1`` = unknown (``<unk>``): a non-zero tuple absent from the table yields a
      single ``<unk>`` followed by padding.
"""

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
from functools import partial
from typing import Dict, List, Optional, Tuple

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DataType,
    IntegerType,
    LongType,
    StructField,
    StructType,
)

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import EventNgramLookupLayer
from kamae.spark.params import EventNgramLookupParams, MultiInputMultiOutputParams
from kamae.spark.utils import (
    tokenize_events,
    tokenize_events_with_types,
    validate_event_column_lengths,
    validate_event_id_columns,
)

from .base import BaseTransformer


class EventNgramLookupTransformerParams(Params):
    """
    Mixin class containing the fitted state of the EventNgramLookupTransformer.

    These are produced by the ``EventNgramLookupEstimator``, so they live on the
    transformer only. The lookup table is held as two parallel flat int lists because
    Spark ML writes params to JSON, where a dict keyed by id tuples cannot be written.
    """

    vocabularySize = Param(
        Params._dummy(),
        "vocabularySize",
        "Number of token ids in the fitted vocabulary, including the reserved pad and "
        "unk tokens.",
        typeConverter=TypeConverters.toInt,
    )

    lookupKeys = Param(
        Params._dummy(),
        "lookupKeys",
        "Flattened event tuples of the fitted lookup table, tupleSize ids per tuple.",
        typeConverter=TypeConverters.toListInt,
    )

    lookupValues = Param(
        Params._dummy(),
        "lookupValues",
        "Flattened token lists of the fitted lookup table, topK tokens per tuple, "
        "positionally aligned with lookupKeys.",
        typeConverter=TypeConverters.toListInt,
    )

    tokenTypeLookup = Param(
        Params._dummy(),
        "tokenTypeLookup",
        "Per-token-id list mapping each token to a bitmask of the ID levels its n-gram "
        "spans (0 for pad/unk). Used only when includeTokenTypes is True.",
        typeConverter=TypeConverters.toListInt,
    )

    def setVocabularySize(self, value: int) -> "EventNgramLookupTransformerParams":
        """
        Sets the vocabularySize parameter.

        :param value: Number of token ids in the fitted vocabulary.
        :returns: Instance of class mixed in.
        """
        return self._set(vocabularySize=value)

    def getVocabularySize(self) -> int:
        """
        Gets the vocabularySize parameter.

        :returns: Number of token ids in the fitted vocabulary.
        """
        return self.getOrDefault(self.vocabularySize)

    def setLookupKeys(self, value: List[int]) -> "EventNgramLookupTransformerParams":
        """
        Sets the lookupKeys parameter.

        :param value: Flattened event tuples, tupleSize ids per tuple.
        :returns: Instance of class mixed in.
        """
        return self._set(lookupKeys=value)

    def getLookupKeys(self) -> Optional[List[int]]:
        """
        Gets the lookupKeys parameter.

        :returns: Flattened event tuples, or None if not set.
        """
        return self.getOrDefault(self.lookupKeys)

    def setLookupValues(self, value: List[int]) -> "EventNgramLookupTransformerParams":
        """
        Sets the lookupValues parameter.

        :param value: Flattened token lists, topK tokens per tuple.
        :returns: Instance of class mixed in.
        """
        return self._set(lookupValues=value)

    def getLookupValues(self) -> Optional[List[int]]:
        """
        Gets the lookupValues parameter.

        :returns: Flattened token lists, or None if not set.
        """
        return self.getOrDefault(self.lookupValues)

    def setTokenTypeLookup(
        self, value: List[int]
    ) -> "EventNgramLookupTransformerParams":
        """
        Sets the tokenTypeLookup parameter.

        :param value: Per-token-id list of ID-level bitmasks.
        :returns: Instance of class mixed in.
        """
        return self._set(tokenTypeLookup=value)

    def getTokenTypeLookup(self) -> Optional[List[int]]:
        """
        Gets the tokenTypeLookup parameter.

        :returns: Per-token-id list of ID-level bitmasks, or None.
        """
        return self.getOrDefault(self.tokenTypeLookup)

    def getTupleToTokens(self) -> Dict[Tuple[int, ...], List[int]]:
        """
        Rebuilds the ``event tuple -> top-k token list`` table from the flat params.

        :returns: Mapping from each event tuple to its list of token ids.
        """
        keys = self.getLookupKeys() or []
        values = self.getLookupValues() or []
        tuple_size = self.getTupleSize()
        top_k = self.getTopK()
        return {
            tuple(keys[i : i + tuple_size]): values[j : j + top_k]
            for i, j in zip(
                range(0, len(keys), tuple_size), range(0, len(values), top_k)
            )
        }


class EventNgramLookupTransformer(
    BaseTransformer,
    MultiInputMultiOutputParams,
    EventNgramLookupParams,
    EventNgramLookupTransformerParams,
):
    """
    Tokenizes discrete ID values using the pre-computed ``tuple -> top-k tokens`` table.

    Each input column holds a sequence of events; every event is a ``tupleSize``-long
    group of IDs. For each event the transformer emits the tuple's ``topK`` token ids,
    producing a flat ``numEvents * topK`` array per input column. When
    ``includeTokenTypes`` is set, a parallel ``<col>_types`` column of the same shape is
    also produced, giving each token's ID-level bitmask.

    A row whose ids are not a whole number of events raises, as in the Keras layer.
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
        :param vocabularySize: Number of token ids in the fitted vocabulary.
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
        return [IntegerType(), LongType()]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Tokenizes each input column with a per-column UDF.

        Each input value is a flat integer array, split into consecutive events of
        ``tupleSize`` ids. The output is a flat integer array of length
        ``numEvents * topK``. When ``includeTokenTypes`` is set, the same UDF also
        returns each token's type bitmask, written to the ``<col>_types`` column.

        :param dataset: Input DataFrame.
        :raises ValueError: If the input columns, output columns and event counts
        differ in length, or a column is missing or is not a single-level array.
        :returns: DataFrame with the tokenized output columns, and the optional type
        columns.
        """
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()
        num_events_per_input = self.getNumEventsPerInput()
        validate_event_column_lengths(input_cols, output_cols, num_events_per_input)
        validate_event_id_columns(dataset, input_cols)

        table_kwargs = {
            "lookup_table": self.getTupleToTokens(),
            "tuple_size": self.getTupleSize(),
            "top_k": self.getTopK(),
        }
        token_array_type = ArrayType(IntegerType())
        if not self.getIncludeTokenTypes():
            for input_col, output_col, num_events in zip(
                input_cols, output_cols, num_events_per_input
            ):
                tokenize_udf = F.udf(
                    partial(tokenize_events, num_events=num_events, **table_kwargs),
                    token_array_type,
                )
                dataset = dataset.withColumn(output_col, tokenize_udf(F.col(input_col)))
            return dataset

        # Tokens and types come from one UDF returning a struct, so each row crosses
        # into Python once.
        struct_type = StructType(
            [
                StructField("tokens", token_array_type),
                StructField("types", token_array_type),
            ]
        )
        for input_col, output_col, type_col, num_events in zip(
            input_cols,
            output_cols,
            self.getTokenTypeCols(output_cols),
            num_events_per_input,
        ):
            tokenize_udf = F.udf(
                partial(
                    tokenize_events_with_types,
                    token_type_lookup=self.getTokenTypeLookup(),
                    num_events=num_events,
                    **table_kwargs,
                ),
                struct_type,
            )
            struct_col = f"{output_col}__tokens_and_types"
            # The type columns are not in outputCols, so the base class's output cast
            # does not reach them; cast them here to match the Keras layer.
            casted_types, _ = self._cast_output_columns(
                [F.col(f"{struct_col}.types")], [token_array_type]
            )[0]
            dataset = (
                dataset.withColumn(struct_col, tokenize_udf(F.col(input_col)))
                .withColumn(output_col, F.col(f"{struct_col}.tokens"))
                .withColumn(type_col, casted_types)
                .drop(struct_col)
            )
        return dataset

    def get_keras_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the Keras layer for the EventNgramLookup transformer.

        Returns a single ``EventNgramLookupLayer`` that tokenizes all input columns, so
        the fitted lookup table is embedded once. The layer outputs one token tensor
        per input, plus one type tensor per input when ``includeTokenTypes`` is set.

        :returns: The consolidated ``EventNgramLookupLayer``.
        """
        lookup_table = self.getTupleToTokens()
        return EventNgramLookupLayer(
            num_events_per_input=self.getNumEventsPerInput(),
            top_k=self.getTopK(),
            tuple_size=self.getTupleSize(),
            lookup_keys=[list(key) for key in lookup_table],
            lookup_values=list(lookup_table.values()),
            token_type_lookup=(
                self.getTokenTypeLookup() if self.getIncludeTokenTypes() else None
            ),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            name=self.getLayerName(),
        )

    def get_layer_inputs_outputs(self) -> Tuple[List[str], List[str]]:
        """
        Gets the input and output column names, including the token-type columns.

        Overrides the base method because, with ``includeTokenTypes`` set, the layer
        returns the per-input type tensors after the token tensors, so the
        ``<col>_types`` columns are appended in that same order.

        :returns: Tuple of the input column names and the output column names.
        """
        inputs, token_cols = super().get_layer_inputs_outputs()
        return inputs, token_cols + self.getTokenTypeCols(token_cols)
