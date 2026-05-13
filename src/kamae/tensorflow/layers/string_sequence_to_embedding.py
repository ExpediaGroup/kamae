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

import re
from typing import Any, Dict, List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class StringSequenceToEmbeddingLayer(BaseLayer):
    """
    Parses a delimited string that encodes a sequence of pre-computed
    embedding vectors into a dense float tensor.

    Each input element is a single string encoding up to ``seq_len``
    fixed-dimension vectors. Vectors are separated by ``sequence_separator``
    (default ``","``) and floats within a vector are separated by
    ``separator`` (default ``"|"``).

    Strings with fewer than ``seq_len * embedding_dim`` floats are padded
    with ``pad_value``; strings with more are truncated. Optionally, the
    non-pad portion of each sequence can be reversed along the sequence
    axis.

    Example:
        layer = StringSequenceToEmbeddingLayer(seq_len=4, embedding_dim=3)
        x = tf.constant([["1|2|3,4|5|6,0|0|0,0|0|0"]])
        layer(x).shape  # (1, 1, 4, 3)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        seq_len: int = 10,
        embedding_dim: int = 32,
        separator: str = "|",
        sequence_separator: str = ",",
        pad_value: str = "0",
        reverse: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialises the StringSequenceToEmbeddingLayer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param seq_len: Maximum number of vectors per sequence. Defaults to 10.
        :param embedding_dim: Dimensionality of each embedding vector.
        Defaults to 32.
        :param separator: Float separator within a vector. Defaults to ``"|"``.
        :param sequence_separator: Separator between vectors.
        Defaults to ``","``.
        :param pad_value: String used to pad short sequences. Defaults to
        ``"0"``.
        :param reverse: If True, reverse the non-pad portion of each
        sequence along the sequence axis. Defaults to False.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1.")
        if separator == sequence_separator:
            raise ValueError("separator and sequence_separator must be different.")
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.separator = separator
        self.sequence_separator = sequence_separator
        self.pad_value = pad_value
        self.reverse = reverse

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Parses each string element into a ``(seq_len, embedding_dim)`` float
        matrix. The resulting tensor has the input shape with ``seq_len`` and
        ``embedding_dim`` appended as trailing dimensions.

        :param inputs: String tensor of arbitrary shape.
        :returns: Float32 tensor with shape
        ``input.shape + (seq_len, embedding_dim)``.
        """
        input_dynamic_shape = tf.shape(inputs)
        flat = tf.reshape(inputs, [-1])

        # Unify the two separators so a single split yields all floats.
        unified = tf.strings.regex_replace(
            flat, re.escape(self.separator), self.sequence_separator
        )

        total_floats = self.seq_len * self.embedding_dim
        split = tf.strings.split(unified, sep=self.sequence_separator)
        dense = split.to_tensor(
            default_value=self.pad_value, shape=[None, total_floats]
        )

        floats = tf.strings.to_number(dense, out_type=tf.float32)
        result = tf.reshape(floats, [-1, self.seq_len, self.embedding_dim])

        if self.reverse:
            # A row is considered padding iff all of its components are 0.
            row_norms = tf.reduce_sum(tf.abs(result), axis=-1)
            seq_lengths = tf.reduce_sum(tf.cast(row_norms > 0, tf.int32), axis=-1)
            result = tf.reverse_sequence(result, seq_lengths, seq_axis=1, batch_axis=0)

        output_shape = tf.concat(
            [
                input_dynamic_shape,
                tf.constant(
                    [self.seq_len, self.embedding_dim], dtype=input_dynamic_shape.dtype
                ),
            ],
            axis=0,
        )
        return tf.reshape(result, output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringSequenceToEmbedding layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "embedding_dim": self.embedding_dim,
                "separator": self.separator,
                "sequence_separator": self.sequence_separator,
                "pad_value": self.pad_value,
                "reverse": self.reverse,
            }
        )
        return config
