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

import keras
import tensorflow as tf
from keras import KerasTensor

import kamae
from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input


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

    supported_backends = TENSORFLOW_ONLY
    jit_compatible = False

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
        try:
            float(pad_value)
        except (TypeError, ValueError):
            raise ValueError(f"pad_value must be a numeric string, got {pad_value!r}.")
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.separator = separator
        self.sequence_separator = sequence_separator
        self.pad_value = pad_value
        self.reverse = reverse

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return ["string"]

    @enforce_single_tensor_input
    def _call(self, inputs: KerasTensor, **kwargs: Any) -> KerasTensor:
        """
        Parses each string element into a ``(seq_len, embedding_dim)`` float
        matrix. The resulting tensor has the input shape with ``seq_len`` and
        ``embedding_dim`` appended as trailing dimensions. If the input has a
        trailing size-1 axis, it is dropped so the output is
        ``input.shape[:-1] + (seq_len, embedding_dim)``. This matches the
        convention used by ``StringToStringListLayer``.

        :param inputs: String tensor of arbitrary shape.
        :returns: Float32 tensor with shape
        ``input.shape + (seq_len, embedding_dim)`` or, if the input has a
        trailing size-1 axis, ``input.shape[:-1] + (seq_len, embedding_dim)``.
        """
        input_dynamic_shape = tf.shape(inputs)
        input_static_shape = inputs.shape.as_list()
        drop_trailing_axis = (
            len(input_static_shape) >= 1 and input_static_shape[-1] == 1
        )

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
        # Replace any empty tokens (from leading/trailing/repeated separators
        # or entirely empty inputs) with the pad value so tf.strings.to_number
        # does not fail on the empty string.
        dense = tf.where(tf.equal(dense, ""), self.pad_value, dense)

        floats = tf.strings.to_number(dense, out_type=tf.float32)
        result = tf.reshape(floats, [-1, self.seq_len, self.embedding_dim])

        if self.reverse:
            # Reverse only the vectors actually supplied in the input, leaving
            # any padding we appended at the tail untouched. The number of
            # supplied vectors is derived positionally (the count of non-empty
            # vectors in the original input, capped at ``seq_len``) so it does
            # not depend on the numeric value of ``pad_value``.
            vector_groups = tf.strings.split(flat, sep=self.sequence_separator)
            non_empty = tf.cast(tf.not_equal(vector_groups, ""), tf.int32)
            seq_lengths = tf.minimum(tf.reduce_sum(non_empty, axis=-1), self.seq_len)
            result = tf.reverse_sequence(result, seq_lengths, seq_axis=1, batch_axis=0)

        leading_shape = (
            input_dynamic_shape[:-1] if drop_trailing_axis else input_dynamic_shape
        )
        output_shape = tf.concat(
            [
                leading_shape,
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
