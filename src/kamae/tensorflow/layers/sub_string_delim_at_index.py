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

from typing import Any, Dict, List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class SubStringDelimAtIndexLayer(BaseLayer):
    """
    Layer which splits a string tensor by a delimiter and
    returns the substring at the specified index. If the delimiter is the empty
    string, the string is split into bytes/characters.
    If the index is negative, start counting from the end of the string.
    If the index is out of bounds, the default value is returned.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        delimiter: str = "_",
        index: int = 0,
        default_value: str = "",
        **kwargs,
    ) -> None:
        """
        Initialise the SubStringDelimAtIndexLayer layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param delimiter: String to split on. Defaults to `"_"`.
        :param index: Index of the substring to return. Defaults to `0`.
        If the index is negative, start counting from the end of the string.
        :param default_value: Value to return if index is out of bounds.
        Defaults to `""`.
        Defaults to `""`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.delimiter = delimiter
        self.index = index
        self.default_value = default_value

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @staticmethod
    def resolve_negative_indices(
        ragged_tensor: tf.RaggedTensor, index: int
    ) -> tf.Tensor:
        """
        Resolves negative indices to positive indices.

        :param ragged_tensor: Ragged tensor
        :param index: The index to resolve.
        :returns: The resolved index.
        """
        if index >= 0:
            raise ValueError("Index should be negative to resolve. Got positive index.")
        ragged_row_lengths = ragged_tensor.row_lengths(axis=-1)
        # Positive index is the length of the row + index. So that index = -1
        # resolves to the last dimension
        return tf.math.add(ragged_row_lengths, index)

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Splits the input string tensor by the delimiter and returns the substring
        at the specified index. If the index is out of bounds, the default value
        is returned.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if an iterable of tensors is passed
        in.

        :param inputs: Input tensor.
        :returns: Tensor with the substring at the specified index.
        """
        input_shape = tf.shape(inputs)
        # If the delimiter is empty, we split on bytes/characters.
        # Otherwise, we use the standard string split.
        ragged_strings_split = (
            tf.strings.split(inputs, sep=self.delimiter)
            if self.delimiter != ""
            else tf.strings.bytes_split(inputs)
        )

        if self.index >= 0:
            # The index is fully qualified, therefore, add the index + 1 to the shape
            # and then pad the ragged tensor to that shape. If the index is
            # out of bounds, it returns the default value
            index_shape = tf.constant([self.index + 1])
            input_shape = tf.concat([input_shape, index_shape], axis=0)
            return ragged_strings_split.to_tensor(
                default_value=self.default_value, shape=input_shape
            )[..., self.index]
        else:
            # The index is negative, so we need to resolve the positive index from it.
            resolved_index_tensor = self.resolve_negative_indices(
                ragged_tensor=ragged_strings_split, index=self.index
            )
            if isinstance(resolved_index_tensor, tf.RaggedTensor):
                # The resolved indices can be ragged or a regular tensor, however
                # are always rectangular since we only have a single ragged dimension,
                # and we have found the required index within this.
                resolved_index_tensor = resolved_index_tensor.to_tensor(
                    shape=tf.shape(inputs)
                )

            # Pad the ragged tensor to the maximum row_length of the ragged tensor
            # This could be different for each batch, however we return a single index
            # from it, and thus we will have consistent output shapes per batch.
            max_ragged_dim = tf.cast(
                tf.reduce_max(ragged_strings_split.row_lengths(axis=-1)), dtype=tf.int32
            )
            input_shape = tf.concat(
                [input_shape, tf.expand_dims(max_ragged_dim, axis=0)], axis=0
            )
            padded_tensor = ragged_strings_split.to_tensor(
                default_value=self.default_value, shape=input_shape
            )
            # Expand the indices to match the shape of the input
            expanded_indices = tf.expand_dims(resolved_index_tensor, axis=-1)
            # Replace negative indices with zeros temporarily, we will send these to the
            # default value as they are out of bounds
            non_negative_expanded_indices = tf.where(
                expanded_indices < 0,
                tf.constant(0, dtype=expanded_indices.dtype),
                expanded_indices,
            )
            # Gather the resolved indices from the padded tensor, send any negative
            # indices to the default value
            gathered_tensor = tf.where(
                expanded_indices >= 0,
                tf.gather(padded_tensor, non_negative_expanded_indices, batch_dims=-1),
                tf.constant(self.default_value),
            )
            # Squeeze out the extra dimension
            return tf.squeeze(gathered_tensor, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the config of the SubStringDelimAtIndex layer.
        Used for saving and loading from a model.

        Specifically adds the `delimiter`, `index` and `default_value` to the config.

        :returns: Dictionary of the config of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "delimiter": self.delimiter,
                "index": self.index,
                "default_value": self.default_value,
            }
        )
        return config
