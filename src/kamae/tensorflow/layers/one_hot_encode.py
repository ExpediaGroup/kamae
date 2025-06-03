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

import warnings
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class OneHotEncodeLayer(BaseLayer):
    """
    Performs a one-hot encoding of a string input tensor.

    Encodes each individual element in the input into an
    array the same size as the vocabulary, containing a 1 at the element
    index. If the last dimension is size 1, will encode on that
    dimension. If the last dimension is not size 1, will append a new
    dimension for the encoded output.
    """

    def __init__(
        self,
        vocabulary: Union[str, List[str]],
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        mask_token: str = None,
        num_oov_indices: int = 1,
        drop_unseen: bool = False,
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        """
        Intialises the OneHotLayer layer.

        :param vocabulary: Either an array of strings or a string path to a
        text file. If passing an array, can pass a tuple, list, 1D numpy array,
        or 1D tensor containing the string vocbulary terms. If passing a file
        path, the file should contain one line per term in the vocabulary.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param mask_token: A token that represents masked inputs. The token is included
        in vocabulary and mapped to index 0. If set to None, no mask term will be added.
        Defaults to `None`.
        :param num_oov_indices: The number of out-of-vocabulary indices to use. The
        out-of-vocabulary indices are used to represent unseen labels and are placed at
        the beginning of the one-hot encoding. Defaults to 1.
        :param drop_unseen: Whether to drop unseen label indices. If set to True, the
        layer will not add an extra dimension for unseen labels in the one-hot
        encoding. Defaults to False.
        :param encoding: The text encoding to use to interpret the input strings.
        Defaults to `"utf-8"`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.num_oov_indices = num_oov_indices
        self.vocabulary = vocabulary
        self.drop_unseen = drop_unseen
        self.mask_token = mask_token
        self.encoding = encoding
        self.lookup_layer = tf.keras.layers.StringLookup(
            vocabulary=self.vocabulary,
            output_mode="int",
            num_oov_indices=self.num_oov_indices,
            mask_token=self.mask_token,
            encoding=self.encoding,
        )

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.int16, tf.int32, tf.int64, tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs the one-hot encoding on the input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input tensor to one-hot encode.
        :returns: One-hot encoded input tensor.
        """
        casted_inputs = (
            tf.strings.as_string(inputs, scientific=False)
            if inputs.dtype != tf.string
            else inputs
        )
        indexed_inputs = self.lookup_layer(casted_inputs)
        mask_offset = 1 if self.mask_token is not None else 0

        # If last dimension to encode is 1,
        # remove it after one-hot encoding.
        # E.g. (None, None, 1) -> (None, None, 1, N) -> (None, None, N)
        # But (None, None, M) -> (None, None, M, N)
        ohe_depth = len(self.vocabulary) + self.num_oov_indices + mask_offset
        encoded_inputs = (
            tf.squeeze(tf.one_hot(indexed_inputs, ohe_depth), axis=-2)
            if indexed_inputs.get_shape()[-1] == 1
            else tf.one_hot(indexed_inputs, ohe_depth)
        )

        # If drop unseen, slice off the first num_oov_indices + mask_offset columns
        if self.drop_unseen:
            encoded_inputs = encoded_inputs[..., (self.num_oov_indices + mask_offset) :]

        return encoded_inputs

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the OneHot layer.
        Used for saving and loading from a model.

        Specifically adds the `vocabulary`, `num_oov_indices`, `mask_token`, and
        `encoding` to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "vocabulary": self.vocabulary,
                "num_oov_indices": self.num_oov_indices,
                "drop_unseen": self.drop_unseen,
                "mask_token": self.mask_token,
                "encoding": self.encoding,
            }
        )
        return config


# TODO: Remove this alias in next breaking change,
#  it is maintained for backwards compatibility
class OneHotLayer(OneHotEncodeLayer):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OneHotLayer is deprecated and will be removed in a future release. "
            "Use OneHotEncodeLayer instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        super().__init__(*args, **kwargs)
