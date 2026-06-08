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

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
import re
from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringSequenceToEmbeddingLayer

from .base import BaseTransformer


class StringSequenceToEmbeddingParams(Params):
    """
    Mixin class containing the parameters required to parse a delimited
    string of embedding vectors into a dense float matrix.
    """

    seqLen = Param(
        Params._dummy(),
        "seqLen",
        "Maximum number of vectors per sequence.",
        typeConverter=TypeConverters.toInt,
    )

    embeddingDim = Param(
        Params._dummy(),
        "embeddingDim",
        "Dimensionality of each embedding vector.",
        typeConverter=TypeConverters.toInt,
    )

    separator = Param(
        Params._dummy(),
        "separator",
        "Separator between floats within a vector.",
        typeConverter=TypeConverters.toString,
    )

    sequenceSeparator = Param(
        Params._dummy(),
        "sequenceSeparator",
        "Separator between vectors in a sequence.",
        typeConverter=TypeConverters.toString,
    )

    padValue = Param(
        Params._dummy(),
        "padValue",
        "String used to pad short sequences.",
        typeConverter=TypeConverters.toString,
    )

    reverse = Param(
        Params._dummy(),
        "reverse",
        "Reverse the non-pad portion of each sequence along the sequence axis.",
        typeConverter=TypeConverters.toBoolean,
    )

    def getSeqLen(self) -> int:
        return self.getOrDefault(self.seqLen)

    def setSeqLen(self, value: int) -> "StringSequenceToEmbeddingParams":
        if value < 1:
            raise ValueError("seqLen must be >= 1.")
        return self._set(seqLen=value)

    def getEmbeddingDim(self) -> int:
        return self.getOrDefault(self.embeddingDim)

    def setEmbeddingDim(self, value: int) -> "StringSequenceToEmbeddingParams":
        if value < 1:
            raise ValueError("embeddingDim must be >= 1.")
        return self._set(embeddingDim=value)

    def getSeparator(self) -> str:
        return self.getOrDefault(self.separator)

    def setSeparator(self, value: str) -> "StringSequenceToEmbeddingParams":
        return self._set(separator=value)

    def getSequenceSeparator(self) -> str:
        return self.getOrDefault(self.sequenceSeparator)

    def setSequenceSeparator(self, value: str) -> "StringSequenceToEmbeddingParams":
        return self._set(sequenceSeparator=value)

    def getPadValue(self) -> str:
        return self.getOrDefault(self.padValue)

    def setPadValue(self, value: str) -> "StringSequenceToEmbeddingParams":
        return self._set(padValue=value)

    def getReverse(self) -> bool:
        return self.getOrDefault(self.reverse)

    def setReverse(self, value: bool) -> "StringSequenceToEmbeddingParams":
        return self._set(reverse=value)


class StringSequenceToEmbeddingTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    StringSequenceToEmbeddingParams,
):
    """
    Spark transformer that parses a delimited string of pre-computed
    embedding vectors into a nested array of floats with shape
    ``(seq_len, embedding_dim)``.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        seqLen: int = 10,
        embeddingDim: int = 32,
        separator: str = "|",
        sequenceSeparator: str = ",",
        padValue: str = "0",
        reverse: bool = False,
    ) -> None:
        """
        Initialises a StringSequenceToEmbeddingTransformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow
        layer in the keras model. If not set, we use the uid of the Spark
        transformer.
        :param seqLen: Maximum number of vectors per sequence. Defaults to 10.
        :param embeddingDim: Dimensionality of each embedding vector.
        Defaults to 32.
        :param separator: Separator between floats within a vector.
        Defaults to ``"|"``.
        :param sequenceSeparator: Separator between vectors in a sequence.
        Defaults to ``","``.
        :param padValue: String used to pad short sequences.
        Defaults to ``"0"``.
        :param reverse: If True, reverse the non-pad portion of each sequence.
        Defaults to False.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(seqLen=10)
        self._setDefault(embeddingDim=32)
        self._setDefault(separator="|")
        self._setDefault(sequenceSeparator=",")
        self._setDefault(padValue="0")
        self._setDefault(reverse=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [StringType()]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Parses the input string column into a
        nested array column with shape ``(seq_len, embedding_dim)``.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        if self.getSeparator() == self.getSequenceSeparator():
            raise ValueError("separator and sequenceSeparator must be different.")

        seq_len = self.getSeqLen()
        embedding_dim = self.getEmbeddingDim()
        pad_value = self.getPadValue()
        reverse = self.getReverse()
        total_floats = seq_len * embedding_dim
        # Build a single regex pattern that matches either delimiter so we can
        # split in one pass.
        split_pattern = (
            f"[{re.escape(self.getSeparator())}"
            f"{re.escape(self.getSequenceSeparator())}]"
        )

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def parse_sequence(x: Column) -> Column:
            # Split the input string into flat float tokens.
            tokens = F.split(x, pattern=split_pattern)
            # Replace empty tokens with the pad value.
            tokens = F.transform(
                tokens,
                lambda t: F.when(t == F.lit(""), pad_value).otherwise(t),
            )
            # Truncate to at most ``total_floats`` tokens.
            tokens = F.slice(tokens, 1, total_floats)
            # Pad with pad_value to exactly ``total_floats`` tokens.
            tokens = F.concat(
                tokens,
                F.array_repeat(
                    F.lit(pad_value),
                    F.greatest(F.lit(total_floats) - F.size(tokens), F.lit(0)),
                ),
            )
            # Cast each token to float.
            float_tokens = F.transform(tokens, lambda t: t.cast("float"))

            # Reshape flat array of length seq_len * embedding_dim into a
            # nested array of shape (seq_len, embedding_dim).
            vectors = F.transform(
                F.sequence(F.lit(0), F.lit(seq_len - 1)),
                lambda i: F.slice(float_tokens, i * embedding_dim + 1, embedding_dim),
            )

            if not reverse:
                return vectors

            # Count the number of non-pad vectors (a vector is pad iff all
            # of its components are zero). Reverse only that prefix.
            abs_sums = F.transform(
                vectors,
                lambda v: F.aggregate(
                    v,
                    F.lit(0.0),
                    lambda acc, value: acc + F.abs(value),
                ),
            )
            non_pad_count = F.aggregate(
                abs_sums,
                F.lit(0),
                lambda acc, s: acc + F.when(s > F.lit(0.0), 1).otherwise(0),
            )
            reversed_prefix = F.reverse(F.slice(vectors, 1, non_pad_count))
            suffix = F.slice(vectors, non_pad_count + 1, F.lit(seq_len) - non_pad_count)
            return F.concat(reversed_prefix, suffix)

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=parse_sequence,
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the StringSequenceToEmbedding transformer.

        :returns: Tensorflow keras layer equivalent to this transformer.
        """
        return StringSequenceToEmbeddingLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            seq_len=self.getSeqLen(),
            embedding_dim=self.getEmbeddingDim(),
            separator=self.getSeparator(),
            sequence_separator=self.getSequenceSeparator(),
            pad_value=self.getPadValue(),
            reverse=self.getReverse(),
        )
