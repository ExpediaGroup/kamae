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
EventNgramLookupEstimator: learns an event-level n-gram vocabulary and lookup table.

Discrete ID values arrive as sequences of fixed-size events (e.g. 4 ID levels
``L0, L1, L2, L3``). The estimator counts every within-event n-gram (all
combinations of length 1..``tupleSize``, so skipped ID levels are included) across
the corpus, keeps the ``vocabSize`` most frequent above ``minNgramFreq``, and
pre-computes, for every observed event tuple, its ``topK`` token ids. The fitted
lookup table is handed to the ``EventNgramLookupTransformer`` for O(1) tokenization.
"""

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
from typing import List, Optional

from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, IntegerType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.spark.params import (
    EventNgramLookupParams,
    MultiInputMultiOutputParams,
    SampleFractionParams,
)
from kamae.spark.transformers import EventNgramLookupTransformer
from kamae.spark.utils.ngram_utils import (
    EventNgramVocabulary,
    build_tuple_lookup_table,
    build_vocabulary,
    collect_ngrams_from_dataframe,
)

from .base import BaseEstimator


class EventNgramLookupEstimatorParams(Params):
    """
    Mixin class containing the fit-time parameters of the EventNgramLookupEstimator.

    These govern vocabulary selection only, so they are not carried by the fitted
    transformer and live here rather than in the shared params module.
    """

    vocabSize = Param(
        Params._dummy(),
        "vocabSize",
        "Target vocabulary size when training, including the reserved pad and unk "
        "tokens.",
        typeConverter=TypeConverters.toInt,
    )

    minNgramFreq = Param(
        Params._dummy(),
        "minNgramFreq",
        "Minimum corpus frequency for an n-gram to enter the vocabulary.",
        typeConverter=TypeConverters.toInt,
    )

    def setVocabSize(self, value: int) -> "EventNgramLookupEstimatorParams":
        """
        Sets the vocabSize parameter.

        :param value: Target vocabulary size when training.
        :raises ValueError: If the vocabulary size does not leave room for at least
        one learned n-gram alongside the two reserved tokens.
        :returns: Instance of class mixed in.
        """
        if value < 3:
            raise ValueError(
                f"vocabSize must be at least 3 to hold the reserved pad and unk "
                f"tokens plus one n-gram. Got {value}"
            )
        return self._set(vocabSize=value)

    def getVocabSize(self) -> int:
        """
        Gets the vocabSize parameter.

        :returns: Target vocabulary size when training.
        """
        return self.getOrDefault(self.vocabSize)

    def setMinNgramFreq(self, value: int) -> "EventNgramLookupEstimatorParams":
        """
        Sets the minNgramFreq parameter.

        :param value: Minimum n-gram frequency threshold.
        :raises ValueError: If the threshold is not a positive integer.
        :returns: Instance of class mixed in.
        """
        if value < 1:
            raise ValueError(f"minNgramFreq must be a positive integer. Got {value}")
        return self._set(minNgramFreq=value)

    def getMinNgramFreq(self) -> int:
        """
        Gets the minNgramFreq parameter.

        :returns: Minimum n-gram frequency threshold.
        """
        return self.getOrDefault(self.minNgramFreq)


class EventNgramLookupEstimator(
    BaseEstimator,
    MultiInputMultiOutputParams,
    EventNgramLookupParams,
    EventNgramLookupEstimatorParams,
    SampleFractionParams,
):
    """
    Estimator that learns an event-level n-gram vocabulary and its lookup table.

    Fitting counts within-event n-grams across all input columns, selects the most
    frequent ones into a shared vocabulary, and pre-computes each observed event
    tuple's top-k token ids. It returns an ``EventNgramLookupTransformer`` carrying
    that lookup table. Set ``sampleFraction`` to train on a random subset of rows.
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
        topK: int = 10,
        vocabSize: int = 50000,
        minNgramFreq: int = 10,
        sampleFraction: Optional[float] = None,
        useFitSample: bool = False,
        includeTokenTypes: bool = False,
    ) -> None:
        """
        Initializes the EventNgramLookupEstimator.

        :param inputCols: Input column names holding discrete ID values.
        :param outputCols: Output column names for the token arrays.
        :param inputDtype: Data type to cast the input columns to before fitting.
        :param outputDtype: Data type the transformer casts its output columns to.
        :param layerName: Name of the layer. Used as the name of the Keras layer in the
        keras model. If not set, we use the uid of the Spark estimator.
        :param numEventsPerInput: Number of events per input column, e.g. [10, 10, 1].
        :param tupleSize: Number of discrete ID values (ID levels) per event tuple.
        :param topK: Number of tokens emitted per event tuple.
        :param vocabSize: Target vocabulary size (including the reserved pad and unk
        tokens).
        :param minNgramFreq: Minimum corpus frequency for an n-gram to be kept.
        :param sampleFraction: Optional fraction of rows to sample before fitting
        (exclusive 0.0-1.0). None (default) uses all rows. Note that sampling biases
        n-gram frequencies against the minNgramFreq threshold, so the learned
        vocabulary is approximate.
        :param useFitSample: If True, fit on the enclosing pipeline's shared sample
        when fitSampleFraction is set. Default False. Leaving this False is
        recommended: this estimator is a vocabulary builder thresholded on exact
        corpus frequencies, not a sample-robust statistic like a mean or quantile.
        :param includeTokenTypes: If `True`, the transformer also emits a parallel
        ``<col>_types`` column per input giving each token's ID-level bitmask. Defaults
        to `False`.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(
            numEventsPerInput=None,
            tupleSize=4,
            topK=10,
            vocabSize=50000,
            minNgramFreq=10,
            sampleFraction=None,
            useFitSample=False,
            includeTokenTypes=False,
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the estimator input columns.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the estimator.
        """
        return [IntegerType(), ArrayType(IntegerType())]

    def _fit(self, dataset: DataFrame) -> EventNgramLookupTransformer:
        """
        Trains the n-gram vocabulary and returns the fitted transformer.

        :param dataset: Input DataFrame with the discrete-ID columns.
        :returns: An ``EventNgramLookupTransformer`` carrying the fitted lookup table.
        :raises ValueError: If ``inputCols``, ``outputCols`` and ``numEventsPerInput``
        do not all have the same length.
        """
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()
        num_events_per_input = self.getNumEventsPerInput()

        if len(input_cols) != len(output_cols):
            raise ValueError(
                f"inputCols and outputCols must have the same length. Got "
                f"{len(input_cols)} inputs and {len(output_cols)} outputs."
            )
        if num_events_per_input is None or len(input_cols) != len(num_events_per_input):
            n_events = 0 if num_events_per_input is None else len(num_events_per_input)
            raise ValueError(
                f"numEventsPerInput must have one entry per input column. Got "
                f"{len(input_cols)} inputs and {n_events} event counts."
            )

        tuple_size = self.getTupleSize()
        top_k = self.getTopK()

        # Count within-event n-grams, then keep the most frequent as the vocabulary.
        ngram_counter = collect_ngrams_from_dataframe(
            df=dataset,
            input_columns=input_cols,
            event_size=tuple_size,
        )
        vocabulary = EventNgramVocabulary(
            ngrams=build_vocabulary(
                ngram_counter=ngram_counter,
                vocab_size=self.getVocabSize(),
                min_ngram_freq=self.getMinNgramFreq(),
            )
        )

        # Pre-compute the top-k tokens for every event tuple observed in the corpus.
        tuple_to_tokens = build_tuple_lookup_table(
            df=dataset,
            input_columns=input_cols,
            vocabulary=vocabulary,
            top_k=top_k,
            event_size=tuple_size,
        )

        # Optionally derive the per-token ID-level bitmask ("type") lookup.
        include_token_types = self.getIncludeTokenTypes()
        token_type_lookup = (
            vocabulary.build_type_lookup() if include_token_types else None
        )

        # Flatten the table into two parallel int lists so that it is JSON-serialisable
        # and the fitted pipeline can be saved. One pass over the items keeps the keys
        # and values positionally aligned.
        lookup_keys: List[int] = []
        lookup_values: List[int] = []
        for id_tuple, tokens in tuple_to_tokens.items():
            lookup_keys.extend(id_tuple)
            lookup_values.extend(tokens)

        return EventNgramLookupTransformer(
            inputCols=input_cols,
            outputCols=output_cols,
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            layerName=self.getLayerName(),
            numEventsPerInput=num_events_per_input,
            tupleSize=tuple_size,
            topK=top_k,
            vocabularySize=vocabulary.vocab_size,
            lookupKeys=lookup_keys,
            lookupValues=lookup_values,
            includeTokenTypes=include_token_types,
            tokenTypeLookup=token_type_lookup,
        )
